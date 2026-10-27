"""Durable Finney/SN126/M0 publication for the hourly light validator.

This module deliberately accepts a wallet only at the final SDK boundary.  All
earlier work is unsigned, persisted, and independently guarded against a live
chain snapshot.  A previous unknown submission is a hard stop, never a reason
to make a fresh call.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from forge.inference.prelaunch import GENESIS, canonical, digest

BURN_UID = 0
NETUID = 126
MECHANISM = 0
NETWORK = "finney"
U16_MAX = 65_535
SERVICE_BPS = 500
BURN_OWNER_COLDKEY = "5DqrUa2z6E9taJdY8FGiPCrtCswsEjHjPbVo5xcTw2GqvKZm"


class PublicationError(ValueError):
    """A live guard failed; caller must retain the window and not retry blindly."""


def _chain_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        try:
            return bytes.fromhex(value.removeprefix("0x"))
        except ValueError as error:
            raise PublicationError("light_live_ciphertext_encoding") from error
    try:
        return bytes(value)
    except (TypeError, ValueError) as error:
        raise PublicationError("light_live_ciphertext_encoding") from error


def _call_hash(value: bytes) -> str:
    return hashlib.blake2b(value, digest_size=32).hexdigest()


def relative_vector(worker_uids: dict[int, int]) -> dict[int, int]:
    """Return 95% burn plus a proportional 5% service pool.

    ``SetWeights`` max-upscales rather than sum-normalizes.  Supplying relative
    95/5 values is therefore intentional: one worker produces u16 values
    ``[65535, 3449]`` and retains the 5/95 ratio after quantization.
    """

    if not worker_uids:
        return {BURN_UID: 95}
    if BURN_UID in worker_uids or any(uid < 0 or merit <= 0 for uid, merit in worker_uids.items()):
        raise PublicationError("light_live_worker_or_burn_uid")
    total = sum(worker_uids.values())
    return {BURN_UID: 95 * total, **{uid: 5 * merit for uid, merit in worker_uids.items()}}


def eligible_workers(artifact: dict[str, Any]) -> dict[int, int]:
    """Map the signed simulation's eligible service to one UID per hotkey."""

    rows = artifact.get("miners")
    if not isinstance(rows, list):
        raise PublicationError("light_live_artifact")
    output: dict[int, int] = {}
    hotkeys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PublicationError("light_live_artifact")
        merit, uid, hotkey = row.get("eligibleVideos"), row.get("uid"), row.get("hotkey")
        if type(merit) is not int or type(uid) is not int or not isinstance(hotkey, str):
            raise PublicationError("light_live_artifact")
        if merit > 0:
            if uid in output or hotkey in hotkeys:
                raise PublicationError("light_live_one_worker_per_hotkey")
            output[uid] = merit
            hotkeys.add(hotkey)
    return output


@dataclass(frozen=True)
class LiveIntent:
    sequence: int
    feed_digest: str
    window: dict[str, int]
    registry_hash: str
    weights: dict[int, int]
    targets: dict[int, tuple[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "attelier/light-live-intent/v1",
            "network": NETWORK,
            "netuid": NETUID,
            "mechanism": MECHANISM,
            "sequence": self.sequence,
            "feedDigest": self.feed_digest,
            "window": self.window,
            "registryHash": self.registry_hash,
            "relativeWeights": [[uid, value] for uid, value in sorted(self.weights.items())],
            # This is deliberately the reviewed identity tuple, not merely a
            # UID. UIDs can be recycled between the signed feed and finality.
            "targets": [
                [uid, hotkey, coldkey] for uid, (hotkey, coldkey) in sorted(self.targets.items())
            ],
        }

    def intent_hash(self) -> str:
        return digest(self.as_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LiveIntent:
        try:
            targets = {
                int(uid): (str(hotkey), str(coldkey)) for uid, hotkey, coldkey in value["targets"]
            }
            weights = {int(uid): int(weight) for uid, weight in value["relativeWeights"]}
            return cls(
                sequence=int(value["sequence"]),
                feed_digest=str(value["feedDigest"]),
                window=dict(value["window"]),
                registry_hash=str(value["registryHash"]),
                weights=weights,
                targets=targets,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PublicationError("light_live_journal_intent") from error


class Publisher(Protocol):
    def guard(self, intent: LiveIntent) -> dict[str, Any]: ...

    def submit(self, intent: LiveIntent, guard: dict[str, Any]) -> dict[str, Any]: ...

    def reconcile(self, intent: LiveIntent, receipt: dict[str, Any]) -> dict[str, Any]: ...


class LiveJournal:
    """One durable intent per feed sequence; old uncertainty blocks new writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS live_intent ("
                "sequence INTEGER PRIMARY KEY, intent TEXT NOT NULL, status TEXT NOT NULL, "
                "receipt TEXT, updated_at INTEGER NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)

    def prepare(self, intent: LiveIntent) -> dict[str, Any]:
        encoded = canonical(intent.as_dict()).decode("ascii")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT intent, status, receipt FROM live_intent WHERE sequence=?",
                (intent.sequence,),
            ).fetchone()
            # A restart must be able to recover its own durable SUBMITTING
            # receipt. Only a *different* unresolved window blocks this one.
            if row is not None:
                if row[0] != encoded:
                    db.execute("ROLLBACK")
                    raise PublicationError("light_live_sequence_conflict")
                db.execute("COMMIT")
                return {"status": row[1], "receipt": json.loads(row[2]) if row[2] else None}
            unresolved = db.execute(
                "SELECT sequence, intent, status, receipt FROM live_intent WHERE status IN "
                "('SUBMITTING','UNKNOWN_AFTER_SUBMIT') ORDER BY sequence LIMIT 1"
            ).fetchone()
            if unresolved is not None:
                db.execute("COMMIT")
                return {
                    "status": "BLOCKED_BY_PENDING",
                    "pendingIntent": json.loads(unresolved[1]),
                    "pendingStatus": unresolved[2],
                    "pendingReceipt": json.loads(unresolved[3]) if unresolved[3] else {},
                }
            db.execute(
                "INSERT INTO live_intent(sequence,intent,status,receipt,updated_at) "
                "VALUES (?,?, 'PREPARED',NULL,?)",
                (intent.sequence, encoded, int(time.time())),
            )
            db.execute("COMMIT")
        return {"status": "PREPARED", "receipt": None}

    def update(self, intent: LiveIntent, status: str, receipt: dict[str, Any] | None) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE live_intent SET status=?, receipt=?, updated_at=? "
                "WHERE sequence=? AND intent=?",
                (
                    status,
                    json.dumps(receipt, sort_keys=True) if receipt else None,
                    int(time.time()),
                    intent.sequence,
                    canonical(intent.as_dict()).decode("ascii"),
                ),
            )

    def reveal_pending(self) -> list[tuple[LiveIntent, dict[str, Any]]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT intent, receipt FROM live_intent "
                "WHERE status IN ('COMMITTED','REVEAL_PENDING')"
            ).fetchall()
        return [
            (LiveIntent.from_dict(json.loads(intent)), json.loads(receipt) if receipt else {})
            for intent, receipt in rows
        ]


class LivePublisher:
    """Coordinates a prepared live intent without hiding ambiguous submissions."""

    def __init__(self, journal: LiveJournal, publisher: Publisher) -> None:
        self.journal, self.publisher = journal, publisher

    def process(self, body: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
        # CR period can exceed the feed cadence. Keep old confirmed commits
        # observable without blocking a newer hourly commit.
        for old_intent, old_receipt in self.journal.reveal_pending():
            reconciled = self.publisher.reconcile(old_intent, old_receipt)
            self.journal.update(old_intent, reconciled["status"], reconciled)
        ledger, registry = body["ledger"], body["registry"]
        weights = relative_vector(eligible_workers(artifact))
        registry_rows = registry.get("miners") if isinstance(registry, dict) else None
        if not isinstance(registry_rows, list):
            raise PublicationError("light_live_registry")
        targets: dict[int, tuple[str, str]] = {}
        for row in registry_rows:
            if not isinstance(row, dict):
                raise PublicationError("light_live_registry")
            uid, hotkey, coldkey = row.get("uid"), row.get("hotkey"), row.get("coldkey")
            if type(uid) is not int or not isinstance(hotkey, str) or not isinstance(coldkey, str):
                raise PublicationError("light_live_registry")
            if uid in weights and uid != BURN_UID:
                if uid in targets:
                    raise PublicationError("light_live_registry")
                targets[uid] = (hotkey, coldkey)
        if set(weights) - {BURN_UID} != set(targets):
            raise PublicationError("light_live_target_not_frozen")
        # The registry pins the subnet owner coldkey. UID 0 is verified against
        # it separately because it is intentionally not a service miner row.
        owner_coldkey = registry.get("subnetOwnerColdkey") if isinstance(registry, dict) else None
        if not isinstance(owner_coldkey, str):
            raise PublicationError("light_live_registry")
        targets[BURN_UID] = ("", owner_coldkey)
        intent = LiveIntent(
            sequence=ledger["sequence"],
            feed_digest=digest(body),
            window=ledger["window"],
            registry_hash=digest(registry),
            weights=weights,
            targets=targets,
        )
        current = self.journal.prepare(intent)
        if current["status"] == "BLOCKED_BY_PENDING":
            pending = LiveIntent.from_dict(current["pendingIntent"])
            receipt = self.publisher.reconcile(pending, current["pendingReceipt"])
            self.journal.update(pending, receipt["status"], receipt)
            if receipt["status"] in {"SUBMITTING", "UNKNOWN_AFTER_SUBMIT"}:
                raise PublicationError("light_live_unreconciled_submission")
            current = self.journal.prepare(intent)
        if current["status"] in {"SUBMITTING", "UNKNOWN_AFTER_SUBMIT"}:
            receipt = self.publisher.reconcile(intent, current["receipt"] or {})
            self.journal.update(intent, receipt["status"], receipt)
            if receipt["status"] in {"SUBMITTING", "UNKNOWN_AFTER_SUBMIT"}:
                raise PublicationError("light_live_unreconciled_submission")
            return {"intent": intent.as_dict(), "status": receipt["status"], "receipt": receipt}
        if current["status"] in {"COMMITTED", "REVEAL_PENDING"}:
            receipt = self.publisher.reconcile(intent, current["receipt"] or {})
            self.journal.update(intent, receipt["status"], receipt)
            return {"intent": intent.as_dict(), "status": receipt["status"], "receipt": receipt}
        if current["status"] == "REVEALED":
            return {"intent": intent.as_dict(), **current}
        guard = self.publisher.guard(intent)
        prepared = None
        persisted: dict[str, Any] | None = None
        try:
            if hasattr(self.publisher, "prepare_submission"):
                prepared, persisted = self.publisher.prepare_submission(intent, guard)
                self.journal.update(intent, "SUBMITTING", persisted)
            else:
                self.journal.update(intent, "SUBMITTING", {"guard": guard})
        except Exception:
            if hasattr(self.publisher, "abort_prepared"):
                self.publisher.abort_prepared()
            raise
        try:
            submitted = (
                self.publisher.submit(intent, guard, prepared)
                if prepared is not None
                else self.publisher.submit(intent, guard)
            )
            receipt = {**persisted, **submitted} if persisted is not None else submitted
        except Exception as error:
            unknown = {**(persisted or {"guard": guard}), "error": type(error).__name__}
            self.journal.update(intent, "UNKNOWN_AFTER_SUBMIT", unknown)
            raise PublicationError("light_live_unknown_after_submit") from error
        status = receipt.get("status", "COMMITTED")
        self.journal.update(intent, status, receipt)
        return {"intent": intent.as_dict(), "status": status, "receipt": receipt}


class SdkM0Publisher:
    """Thin, explicit Bittensor 11.0.1 boundary for the productive service.

    It deliberately has no retry loop.  The caller journals ``SUBMITTING``
    before this method can sign, and treats every exception as ambiguous.
    """

    def __init__(
        self, chain: Any, wallet: Any, validator_hotkey: str, *, lock_dir: Path | None = None
    ) -> None:
        self.chain, self.wallet, self.validator_hotkey = chain, wallet, validator_hotkey
        self.lock_dir = lock_dir
        self._submission_lock: Any = None

    @staticmethod
    def _value(value: Any) -> Any:
        return getattr(value, "value", value)

    def abort_prepared(self) -> None:
        if self._submission_lock is not None:
            self._submission_lock.close()
            self._submission_lock = None

    def guard(self, intent: LiveIntent) -> dict[str, Any]:
        if (getattr(self.chain, "network", None), getattr(self.chain, "netuid", None)) != (
            NETWORK,
            NETUID,
        ):
            raise PublicationError("light_live_chain_namespace")
        from bittensor import storage

        block = self.chain.finalized_height()
        block_hash = self.chain.finalized_block_hash(block)
        raw = self.chain.subtensor._client._substrate.raw
        genesis = str(self.chain.subtensor._call(raw.get_block_hash(0)))
        if genesis != GENESIS:
            raise PublicationError("light_live_genesis")
        at = self.chain.subtensor.at(block)

        def query(name: str, args: list[Any] | None = None) -> Any:
            try:
                item = getattr(storage.SubtensorModule, name)
            except AttributeError as error:
                raise PublicationError(f"light_live_storage_{name}") from error
            return self._value(at.query(item, [NETUID] if args is None else args))

        owner = str(query("SubnetOwner"))
        burn_hotkey = self.chain.uid_snapshot(block).get(BURN_UID)
        owners = self.chain.owner_snapshot(block)
        validator_uid = next(
            (
                uid
                for uid, hotkey in self.chain.uid_snapshot(block).items()
                if hotkey == self.validator_hotkey
            ),
            None,
        )
        if (
            validator_uid is None
            or burn_hotkey is None
            or owners.get(BURN_UID) != (burn_hotkey, owner)
        ):
            raise PublicationError("light_live_identity")
        # These are all fail-closed: a runtime metadata mismatch is not an
        # excuse to assume the owner UID still burns its miner emission.
        if (
            int(query("MechanismCountCurrent")) != 1
            or bool(query("CommitRevealWeightsEnabled")) is not True
            or int(query("WeightsSetRateLimit")) != 100
            or int(query("Tempo")) != 360
            or int(query("CommitRevealWeightsVersion", [])) != 4
            or int(query("WeightsVersionKey")) != 0
            or int(query("MaxWeightsLimit")) != U16_MAX
            or str(query("RecycleOrBurn")) != "Burn"
            or str(query("Owner", [burn_hotkey])) != owner
        ):
            raise PublicationError("light_live_runtime_guard")
        # This item is present in the Finney metadata but intentionally is not
        # exposed by bittensor 11.0.1's typed storage namespace. Read it at the
        # same pinned finalized hash; SubnetMechanism is dTAO-related and is not
        # a Yuma selector.
        consensus = self.chain.subtensor._call(
            raw.query("SubtensorModule", "SubnetEpochConsensus", [NETUID], block_hash)
        )
        if str(consensus) != "Yuma":
            raise PublicationError("light_live_runtime_guard")
        graph = self.chain.subtensor.at(block).read("metagraph", netuid=NETUID)
        permits = graph.get("validator_permit", [])
        if validator_uid >= len(permits) or permits[validator_uid] is not True:
            raise PublicationError("light_live_validator_permit")
        snapshot = self.chain.uid_snapshot(block)
        current_owners = self.chain.owner_snapshot(block)
        for uid, frozen in intent.targets.items():
            if uid not in snapshot or uid not in current_owners:
                raise PublicationError("light_live_target_deregistered")
            current_hotkey, current_coldkey = current_owners[uid]
            if uid == BURN_UID:
                if current_coldkey != frozen[1] or current_hotkey != burn_hotkey:
                    raise PublicationError("light_live_burn_identity")
            elif (current_hotkey, current_coldkey) != frozen:
                raise PublicationError("light_live_target_identity_changed")
        return {
            "block": block,
            "blockHash": block_hash,
            "validatorUid": validator_uid,
            "burnUid": BURN_UID,
            "burnHotkey": burn_hotkey,
            "ownerColdkey": owner,
        }

    def prepare_submission(
        self, intent: LiveIntent, guard: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from bittensor.intents.weights import SetWeights, _conform, _Preflight

        if self.lock_dir is not None:
            safe = "".join(c for c in self.validator_hotkey if c.isalnum() or c in "-_")
            self.lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock = open(self.lock_dir / f"m0-{safe}.lock", "a+", encoding="utf-8")  # noqa: SIM115
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                lock.close()
                raise PublicationError("light_live_global_signer_locked") from error
            self._submission_lock = lock
        uids = sorted(intent.weights)
        plan = self.chain.subtensor.plan(
            SetWeights(
                netuid=NETUID,
                uids=uids,
                weights=[intent.weights[uid] for uid in uids],
                mechid=MECHANISM,
            ),
            self.wallet,
        )
        if not plan.ok or plan.signer != "hotkey" or plan.signer_address != self.validator_hotkey:
            raise PublicationError("light_live_sdk_plan")
        expected_uids, expected_weights = _conform(
            uids,
            [intent.weights[uid] for uid in uids],
            _Preflight(guard["validatorUid"], True, 0, U16_MAX),
            NETUID,
        )
        if (
            expected_uids != uids
            or expected_weights[0] != U16_MAX
            or any(
                abs(
                    expected_weights[index] / U16_MAX
                    - intent.weights[uids[index]] / intent.weights[0]
                )
                > 2 / U16_MAX
                for index in range(1, len(uids))
            )
        ):
            raise PublicationError("light_live_sdk_relative_vector")
        substrate = self.chain.subtensor._client._substrate
        call = self.chain.subtensor._call(substrate.decode_scale("Call", plan.call.data))
        args = {item["name"]: item["value"] for item in call.get("call_args", [])}
        if (
            call.get("call_module") != "SubtensorModule"
            or call.get("call_function") != "commit_timelocked_mechanism_weights"
            or args.get("netuid") != NETUID
            or args.get("mecid") != MECHANISM
            # SDK 11.0.1 CRv4 owns this protocol tag.  ``WeightsVersionKey``
            # remains the SetWeights version_key (0 for this subnet).
            or args.get("commit_reveal_version") != 4
        ):
            raise PublicationError("light_live_sdk_call")
        call_bytes = bytes(plan.call.data)
        persisted = {
            "status": "SUBMITTING",
            "guard": guard,
            "preparedCallHex": "0x" + call_bytes.hex(),
            "preparedCallHash": _call_hash(call_bytes),
            "preparedCallSha256": hashlib.sha256(call_bytes).hexdigest(),
            "ciphertextSha256": hashlib.sha256(_chain_bytes(args["commit"])).hexdigest(),
            "conformedUids": expected_uids,
            "conformedWeights": expected_weights,
            "revealRound": args.get("reveal_round"),
            "commitRevealVersion": args.get("commit_reveal_version"),
            "uids": args.get("uids", args.get("dests")),
            "weights": args.get("weights"),
            "preparedAtBlock": guard["block"],
        }
        # Return the opaque in-memory SDK plan separately. The durable receipt
        # is evidence for reconciliation, never a permit to re-encrypt/retry.
        return {"plan": plan}, persisted

    def submit(
        self, intent: LiveIntent, guard: dict[str, Any], prepared: dict[str, Any]
    ) -> dict[str, Any]:
        from bittensor.settings import DEFAULT_ERA_PERIOD
        from bittensor.signing import resolve_signer

        plan = prepared["plan"]
        substrate = self.chain.subtensor._client._substrate
        try:
            result = self.chain.subtensor._call(
                substrate.submit(
                    plan.call,
                    resolve_signer(self.wallet, "hotkey"),
                    period=DEFAULT_ERA_PERIOD,
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )
            )
            if not result.success:
                raise PublicationError("light_live_chain_rejected")
            return {
                "status": "COMMITTED",
                "extrinsicId": str(result.extrinsic_id),
                "guard": guard,
                "preparedCallSha256": hashlib.sha256(bytes(plan.call.data)).hexdigest(),
            }
        finally:
            self.abort_prepared()

    def reconcile(self, intent: LiveIntent, receipt: dict[str, Any]) -> dict[str, Any]:
        """Recover finalized commit evidence, then prove the automatic reveal.

        Absence is pending, not permission to create another ciphertext.
        """
        from bittensor import storage

        result = dict(receipt)
        head = self.chain.finalized_height()
        extrinsic_id = result.get("extrinsicId")
        if not extrinsic_id:
            start = int(result.get("submissionScannedThrough", result.get("preparedAtBlock", 0)))
            for height in range(start + 1, min(head, start + 120) + 1):
                candidates = [
                    index
                    for index, extrinsic in enumerate(
                        self.chain.subtensor.block_info(height).extrinsics
                    )
                    if extrinsic.get("address") == self.validator_hotkey
                    and extrinsic.get("call", {}).get("call_hash", "").removeprefix("0x")
                    in {result.get("preparedCallHash"), result.get("preparedCallSha256")}
                ]
                result["submissionScannedThrough"] = height
                if len(candidates) > 1:
                    raise PublicationError("light_live_ambiguous_inclusion")
                if candidates:
                    extrinsic_id = f"{height}-{candidates[0]:04d}"
                    result.update(
                        status="COMMITTED", extrinsicId=extrinsic_id, recoveredInclusion=True
                    )
                    break
            if not extrinsic_id:
                return {**result, "status": "UNKNOWN_AFTER_SUBMIT"}
        included, index = (int(part) for part in str(extrinsic_id).split("-", 1))
        if head < included:
            return {**result, "status": "REVEAL_PENDING"}
        extrinsic = self.chain.subtensor.block_info(included).extrinsics[index]
        call, args = (
            extrinsic["call"],
            {arg["name"]: arg["value"] for arg in extrinsic["call"].get("call_args", [])},
        )
        if (
            extrinsic.get("address") != self.validator_hotkey
            or call.get("call_module") != "SubtensorModule"
            or call.get("call_function") != "commit_timelocked_mechanism_weights"
            or args.get("netuid") != NETUID
            or args.get("mecid") != MECHANISM
            or hashlib.sha256(_chain_bytes(args.get("commit", b""))).hexdigest()
            != result.get("ciphertextSha256")
            or args.get("reveal_round") != result.get("revealRound")
            or args.get("commit_reveal_version") != result.get("commitRevealVersion")
        ):
            raise PublicationError("light_live_finalized_call_mismatch")
        events = [
            event
            for event in self.chain.subtensor.at(included).query(storage.System.Events)
            if event.get("extrinsic_idx") == index
        ]
        if not any(
            e.get("module_id") == "System" and e.get("event_id") == "ExtrinsicSuccess"
            for e in events
        ):
            raise PublicationError("light_live_finalized_failure")
        result.update(
            status="COMMITTED",
            submissionBlock=included,
            submissionBlockHash=self.chain.finalized_block_hash(included),
            submissionEvents=events,
        )
        # A committed item must remain in the pending queue until its unique
        # reveal, so disappearance without applied weights is never success.
        pending = self.chain.subtensor.at(head).query_map(
            storage.SubtensorModule.TimelockedWeightCommits, [NETUID]
        )
        queued = [
            entry
            for _, entries in pending
            for entry in (entries or [])
            if entry[0] == self.validator_hotkey
            and hashlib.sha256(_chain_bytes(entry[2])).hexdigest() == result["ciphertextSha256"]
        ]
        observed = (
            self.chain.subtensor.at(head)
            .read("weights", netuid=NETUID, mechid=MECHANISM)
            .get(result["guard"]["validatorUid"], {})
        )
        requested = {uid: value for uid, value in intent.weights.items()}
        total = sum(requested.values())
        matches = set(observed) == set(requested) and all(
            abs(observed[uid] - value / total) <= 2 / U16_MAX for uid, value in requested.items()
        )
        if matches and not queued:
            result.update(
                status="REVEALED",
                revealedAtBlock=head,
                revealedAtHash=self.chain.finalized_block_hash(head),
                revealedWeights=observed,
            )
        else:
            result.update(
                status="REVEAL_PENDING",
                pendingCommits=len(queued),
                observedWeights=observed,
                checkedAtBlock=head,
            )
        return result
