"""Pinned native CRV3 reads for the qualified testnet publication signer.

The timelocked weight call contains ciphertext, not its uid/value vector. A
matching later weight row is insufficient unless the reader can connect that
ciphertext to the unique automatic reveal that wrote the row. This module only
reads chain data and its local SQLite witness.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bittensor import storage
from bittensor.settings import GLOBAL_MAX_SUBNET_COUNT

from forge.chain.launch_publication import FinalizedWeights, QualifiedPublicationContext
from forge.chain.reward_publication import WeightIntent, WeightSubmissionReceipt


def _hex_bytes(value: object, label: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        try:
            return bytes.fromhex(value.removeprefix("0x"))
        except ValueError as error:
            raise ValueError(f"{label} is not hexadecimal bytes") from error
    if isinstance(value, list) and all(type(item) is int and 0 <= item <= 255 for item in value):
        return bytes(value)
    raise ValueError(f"{label} is not bytes")


def _blake2_256(value: bytes) -> str:
    return hashlib.blake2b(value, digest_size=32).hexdigest()


@dataclass(frozen=True)
class TimelockedCommitWitness:
    """Public CRV3 ciphertext identity prepared before the signer broadcasts."""

    intent_hash: str
    context_hash: str
    mechanism_index: int
    ciphertext_hash: str
    reveal_round: int
    commit_reveal_version: int
    prepared_at_block: int = 0


class _WitnessStore:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        with self._db() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS native_timelocked_commit_witnesses (
                intent_hash TEXT PRIMARY KEY, witness_json TEXT NOT NULL,
                scan_cursor INTEGER NOT NULL, reconciled_receipt_json TEXT,
                reconcile_status TEXT NOT NULL DEFAULT 'PENDING')"""
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(native_timelocked_commit_witnesses)")
            }
            if "scan_cursor" not in columns:
                db.execute(
                    "ALTER TABLE native_timelocked_commit_witnesses "
                    "ADD COLUMN scan_cursor INTEGER NOT NULL DEFAULT 0"
                )
            if "reconciled_receipt_json" not in columns:
                db.execute(
                    "ALTER TABLE native_timelocked_commit_witnesses "
                    "ADD COLUMN reconciled_receipt_json TEXT"
                )
            if "reconcile_status" not in columns:
                db.execute(
                    "ALTER TABLE native_timelocked_commit_witnesses "
                    "ADD COLUMN reconcile_status TEXT NOT NULL DEFAULT 'PENDING'"
                )

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def put(self, witness: TimelockedCommitWitness) -> None:
        encoded = json.dumps(asdict(witness), sort_keys=True, separators=(",", ":"))
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO native_timelocked_commit_witnesses "
                "(intent_hash,witness_json,scan_cursor) VALUES (?,?,?)",
                (witness.intent_hash, encoded, witness.prepared_at_block),
            )
            row = db.execute(
                "SELECT witness_json FROM native_timelocked_commit_witnesses WHERE intent_hash=?",
                (witness.intent_hash,),
            ).fetchone()
        if row is None or row["witness_json"] != encoded:
            raise ValueError("prepared timelocked witness changed")

    def get(self, intent_hash: str) -> TimelockedCommitWitness:
        with self._db() as db:
            row = db.execute(
                "SELECT witness_json FROM native_timelocked_commit_witnesses WHERE intent_hash=?",
                (intent_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("no prepared timelocked witness for this intent")
        try:
            return TimelockedCommitWitness(**json.loads(row["witness_json"]))
        except (TypeError, ValueError) as error:
            raise ValueError("persisted timelocked witness is invalid") from error

    def scan_cursor(self, intent_hash: str) -> int:
        with self._db() as db:
            row = db.execute(
                "SELECT scan_cursor FROM native_timelocked_commit_witnesses WHERE intent_hash=?",
                (intent_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("no prepared timelocked witness for this intent")
        return int(row["scan_cursor"])

    def recovery(self, intent_hash: str) -> tuple[str, WeightSubmissionReceipt | None]:
        with self._db() as db:
            row = db.execute(
                "SELECT reconcile_status, reconciled_receipt_json "
                "FROM native_timelocked_commit_witnesses WHERE intent_hash=?",
                (intent_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("no prepared timelocked witness for this intent")
        receipt = (
            WeightSubmissionReceipt(**json.loads(row["reconciled_receipt_json"]))
            if row["reconciled_receipt_json"]
            else None
        )
        return str(row["reconcile_status"]), receipt

    def record_scan(
        self,
        intent_hash: str,
        block: int,
        receipt: WeightSubmissionReceipt | None,
        *,
        ambiguous: bool = False,
    ) -> None:
        """Atomically retain an exact finding before skipping its scanned blocks."""
        if receipt is not None and ambiguous:
            raise ValueError("a reconciliation result cannot be both exact and ambiguous")
        encoded = (
            json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":")) if receipt else None
        )
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE native_timelocked_commit_witnesses SET scan_cursor=MAX(scan_cursor, ?), "
                "reconciled_receipt_json=COALESCE(reconciled_receipt_json, ?), "
                "reconcile_status=CASE WHEN ? THEN 'AMBIGUOUS' ELSE reconcile_status END "
                "WHERE intent_hash=?",
                (block, encoded, ambiguous, intent_hash),
            )


class TestnetNativePublicationReader:
    __test__ = False

    def __init__(
        self, chain: Any, *, validator_hotkey: str, validator_uid: int, witness_path: str | Path
    ) -> None:
        self.chain = chain
        self.validator_hotkey = validator_hotkey
        self.validator_uid = validator_uid
        self.witnesses = _WitnessStore(Path(witness_path))

    def _raw(self) -> Any:
        return self.chain.subtensor._client._substrate.raw

    @staticmethod
    def _mechanism_index(context: QualifiedPublicationContext) -> int:
        return context.mechanism * GLOBAL_MAX_SUBNET_COUNT + context.netuid

    def _runtime_hash(self, block: int) -> str:
        code = self.chain.subtensor._call(
            self._raw().rpc_request(
                "state_getStorage", ["0x3a636f6465", self.chain.finalized_block_hash(block)]
            )
        )
        return hashlib.sha256(bytes.fromhex(str(code).removeprefix("0x"))).hexdigest()

    def _context(self, context: QualifiedPublicationContext, block: int) -> None:
        if context.network != "test" or (self.chain.network, self.chain.netuid) != (
            context.network,
            context.netuid,
        ):
            raise ValueError("native reader belongs to another testnet subnet")
        genesis = self.chain.subtensor._call(self._raw().get_block_hash(0))
        if str(genesis).removeprefix("0x") != context.genesis_hash:
            raise ValueError("testnet genesis differs from the qualified context")
        if self._runtime_hash(block) != context.runtime_hash:
            raise ValueError("runtime :code differs from the qualified context")

    @staticmethod
    def _arguments(call: Any) -> dict[str, Any]:
        if isinstance(call, Mapping):
            if "call_args" in call:
                arguments = call["call_args"]
            elif "args" in call:
                arguments = call["args"]
            elif "value" in call:
                return TestnetNativePublicationReader._arguments(call["value"])
            else:
                arguments = call
        else:
            value = getattr(call, "value", None)
            if value is not None:
                return TestnetNativePublicationReader._arguments(value)
            arguments = getattr(call, "call_args", None)
        if isinstance(arguments, Mapping):
            return dict(arguments)
        if isinstance(arguments, list) and all(isinstance(item, Mapping) for item in arguments):
            return {item.get("name"): item.get("value") for item in arguments}
        raise ValueError("timelocked call arguments are unavailable")

    @staticmethod
    def _event(event: Any) -> tuple[str, str, list[Any]]:
        value = event.get("event", event) if isinstance(event, Mapping) else event
        if not isinstance(value, Mapping):
            raise ValueError("chain event is invalid")
        module = value.get("module") or value.get("event_module") or value.get("pallet")
        name = value.get("name") or value.get("event_id") or value.get("event")
        attributes = value.get("attributes") or value.get("params") or []
        if isinstance(attributes, Mapping):
            attributes = list(attributes.values())
        if (
            not isinstance(module, str)
            or not isinstance(name, str)
            or not isinstance(attributes, list)
        ):
            raise ValueError("chain event fields are invalid")
        return module, name, attributes

    def _events(self, block: int) -> list[Any]:
        events = self.chain.subtensor.at(block).query(storage.System.Events)
        if isinstance(events, list):
            return events
        info = self.chain.subtensor.block_info(block)
        legacy_events = getattr(info, "events", None)
        if legacy_events is None and isinstance(info, Mapping):
            legacy_events = info.get("events")
        if not isinstance(legacy_events, list):
            raise ValueError("archive block has no decoded events")
        return legacy_events

    def _commit_witness(
        self, context: QualifiedPublicationContext, intent: WeightIntent, call: Any
    ) -> TimelockedCommitWitness:
        arguments = self._arguments(call)
        ciphertext = _hex_bytes(arguments.get("commit"), "timelocked ciphertext")
        reveal_round = arguments.get("reveal_round")
        version = arguments.get("commit_reveal_version")
        if (
            arguments.get("netuid") != context.netuid
            or arguments.get("mecid") != context.mechanism
            or type(reveal_round) is not int
            or reveal_round < 0
            or type(version) is not int
            or version < 0
            or not ciphertext
        ):
            raise ValueError("prepared CRV3 call differs from qualified context")
        return TimelockedCommitWitness(
            intent.intent_hash(),
            context.context_hash(),
            self._mechanism_index(context),
            _blake2_256(ciphertext),
            reveal_round,
            version,
        )

    def capture_prepared(
        self, context: QualifiedPublicationContext, intent: WeightIntent, call: Any
    ) -> TimelockedCommitWitness:
        """Durably bind the SDK's composed encrypted call before broadcast."""
        prepared_at = self.chain.finalized_height()
        self._context(context, prepared_at)
        provisional = self._commit_witness(context, intent, call)
        witness = TimelockedCommitWitness(
            provisional.intent_hash,
            provisional.context_hash,
            provisional.mechanism_index,
            provisional.ciphertext_hash,
            provisional.reveal_round,
            provisional.commit_reveal_version,
            prepared_at,
        )
        self.witnesses.put(witness)
        return witness

    def _raw_extrinsic(self, receipt: WeightSubmissionReceipt) -> Any:
        raw = self._raw()
        block_hash = self.chain.finalized_block_hash(receipt.submitted_block)
        encoded = self.chain.subtensor._call(raw.rpc_request("chain_getBlock", [block_hash]))[
            "block"
        ]["extrinsics"]
        decoded = self.chain.subtensor.block_info(receipt.submitted_block).extrinsics
        for index, value in enumerate(encoded):
            if _blake2_256(_hex_bytes(value, "raw extrinsic")) == receipt.submitted_extrinsic_hash:
                if index >= len(decoded):
                    raise ValueError("finalized extrinsic lacks decoded call")
                return decoded[index]
        raise ValueError("submission receipt extrinsic is absent from its finalized block")

    def _validate_receipt(
        self, context: QualifiedPublicationContext, receipt: WeightSubmissionReceipt
    ) -> TimelockedCommitWitness:
        if (receipt.network, receipt.netuid) != (context.network, context.netuid):
            raise ValueError("submission receipt belongs to another qualified context")
        witness = self.witnesses.get(receipt.intent_hash)
        if witness.context_hash != context.context_hash():
            raise ValueError("prepared ciphertext belongs to another authority context")
        decoded = self._raw_extrinsic(receipt)
        call = decoded["call"] if isinstance(decoded, Mapping) else getattr(decoded, "call", None)
        if not isinstance(call, Mapping):
            raise ValueError("finalized extrinsic call is invalid")
        arguments = self._arguments(call)
        ciphertext = _hex_bytes(arguments.get("commit"), "finalized timelocked ciphertext")
        if (
            (decoded.get("address") if isinstance(decoded, Mapping) else None)
            != self.validator_hotkey
            or call.get("call_module") != "SubtensorModule"
            or call.get("call_function") != "commit_timelocked_mechanism_weights"
            or arguments.get("netuid") != context.netuid
            or arguments.get("mecid") != context.mechanism
            or _blake2_256(ciphertext) != witness.ciphertext_hash
            or arguments.get("reveal_round") != witness.reveal_round
            or arguments.get("commit_reveal_version") != witness.commit_reveal_version
        ):
            raise ValueError("finalized extrinsic differs from prepared CRV3 call")
        matched = 0
        for event in self._events(receipt.submitted_block):
            module, name, attributes = self._event(event)
            if (
                (module, name) == ("SubtensorModule", "TimelockedWeightsCommitted")
                and len(attributes) == 4
                and str(attributes[0]) == self.validator_hotkey
                and int(attributes[1]) == witness.mechanism_index
                and str(attributes[2]).removeprefix("0x") == witness.ciphertext_hash
                and int(attributes[3]) == witness.reveal_round
            ):
                matched += 1
        if matched != 1:
            raise ValueError("finalized CRV3 commit event does not bind the ciphertext")
        return witness

    def receipt_from_submission(
        self, submission: Any, context: QualifiedPublicationContext, intent: WeightIntent
    ) -> WeightSubmissionReceipt:
        """Turn the SDK block-index inclusion id into a pinned CRV3 receipt."""
        identifier = getattr(submission, "result", None)
        if not isinstance(identifier, str) or "-" not in identifier:
            raise ValueError("SDK set_weights result has no block-index receipt")
        try:
            block, index = (int(part) for part in identifier.split("-", 1))
        except ValueError as error:
            raise ValueError("SDK set_weights receipt is not block-index") from error
        if block < 0 or index < 0 or self.chain.finalized_height() < block:
            raise ValueError("SDK set_weights inclusion is not finalized")
        self._context(context, block)
        entries = self.chain.subtensor._call(
            self._raw().rpc_request("chain_getBlock", [self.chain.finalized_block_hash(block)])
        )["block"]["extrinsics"]
        if index >= len(entries):
            raise ValueError("SDK set_weights index is absent from finalized block")
        receipt = WeightSubmissionReceipt(
            intent.intent_hash(),
            context.network,
            context.netuid,
            block,
            _blake2_256(_hex_bytes(entries[index], "raw extrinsic")),
        )
        self._validate_receipt(context, receipt)
        return receipt

    def reconcile(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt | None,
    ) -> WeightSubmissionReceipt | None:
        self._context(context, self.chain.finalized_height())
        if receipt is None:
            witness = self.witnesses.get(intent.intent_hash())
            if witness.context_hash != context.context_hash():
                raise ValueError("prepared ciphertext belongs to another authority context")
            status, recovered = self.witnesses.recovery(intent.intent_hash())
            if recovered is not None:
                # Recheck durable evidence on every restart; the stored receipt is
                # only an atomic recovery point, never authority by itself.
                self._validate_receipt(context, recovered)
                return recovered
            if status == "AMBIGUOUS":
                # A prior scan found more than one matching finalized commit.
                # Never reinterpret that ambiguity from later chain state.
                return None
            height = self.chain.finalized_height()
            cursor = self.witnesses.scan_cursor(intent.intent_hash())
            matches = []
            for block in range(max(witness.prepared_at_block, cursor) + 1, height + 1):
                for value in self.chain.subtensor._call(
                    self._raw().rpc_request(
                        "chain_getBlock", [self.chain.finalized_block_hash(block)]
                    )
                )["block"]["extrinsics"]:
                    candidate = WeightSubmissionReceipt(
                        intent.intent_hash(),
                        context.network,
                        context.netuid,
                        block,
                        _blake2_256(_hex_bytes(value, "raw extrinsic")),
                    )
                    try:
                        self._validate_receipt(context, candidate)
                    except ValueError:
                        continue
                    matches.append(candidate)
            # The cursor and result must commit together. A process may die as
            # soon as this returns; a later reconcile must still find the exact
            # receipt and must never resend because its block was skipped.
            self.witnesses.record_scan(
                intent.intent_hash(),
                height,
                matches[0] if len(matches) == 1 else None,
                ambiguous=len(matches) > 1,
            )
            if len(matches) == 1:
                return matches[0]
            # An earlier ambiguity remains fail-closed; scanning later blocks may
            # discover no new receipt but can never turn it into a positive result.
            return None
        self._validate_receipt(context, receipt)
        return receipt

    def _queue(self, context: QualifiedPublicationContext, block: int) -> list[tuple[Any, ...]]:
        rows = self.chain.subtensor.at(block).query_map(
            storage.SubtensorModule.TimelockedWeightCommits, [self._mechanism_index(context)]
        )
        return [tuple(item) for _epoch, items in rows for item in (items or [])]

    def _revealed_at(
        self,
        context: QualifiedPublicationContext,
        witness: TimelockedCommitWitness,
        commit_block: int,
        block: int,
    ) -> bool:
        if block <= 0:
            return False
        prior = self._queue(context, block - 1)
        matching = []
        same_hotkey = []
        for entry in prior:
            if len(entry) != 4:
                raise ValueError("timelocked queue entry is invalid")
            hotkey, entry_block, ciphertext, reveal_round = entry
            if str(hotkey) != self.validator_hotkey:
                continue
            same_hotkey.append(entry)
            if (
                int(entry_block) == commit_block
                and _blake2_256(_hex_bytes(ciphertext, "queued ciphertext"))
                == witness.ciphertext_hash
                and int(reveal_round) == witness.reveal_round
            ):
                matching.append(entry)
        if len(matching) != 1 or len(same_hotkey) != 1:
            return False
        revealed = weights_set = 0
        for event in self._events(block):
            module, name, attributes = self._event(event)
            if (module, name) == ("SubtensorModule", "TimelockedWeightsRevealed") and (
                len(attributes) == 2
                and int(attributes[0]) == witness.mechanism_index
                and str(attributes[1]) == self.validator_hotkey
            ):
                revealed += 1
            if (module, name) == ("SubtensorModule", "WeightsSet") and (
                len(attributes) == 2
                and int(attributes[0]) == witness.mechanism_index
                and int(attributes[1]) == self.validator_uid
            ):
                weights_set += 1
        return (revealed, weights_set) == (1, 1)

    def finalized_weights(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt,
    ) -> FinalizedWeights | None:
        """Return a vector only after one archived CRV3 reveal proves attribution."""
        height = self.chain.finalized_height()
        self._context(context, height)
        witness = self._validate_receipt(context, receipt)
        if receipt.intent_hash != intent.intent_hash():
            raise ValueError("receipt is not bound to the requested intent")
        revealed = next(
            (
                block
                for block in range(receipt.submitted_block + 1, height + 1)
                if self._revealed_at(context, witness, receipt.submitted_block, block)
            ),
            None,
        )
        if revealed is None:
            return None
        self._context(context, revealed)
        metagraph = self.chain.subtensor.at(revealed).read("metagraph", netuid=context.netuid)
        if metagraph["hotkeys"][self.validator_uid] != self.validator_hotkey:
            raise ValueError("configured signer UID no longer has its hotkey")
        rows = self.chain.subtensor.query(
            storage.SubtensorModule.Weights,
            [self._mechanism_index(context), self.validator_uid],
            block=revealed,
        )
        return FinalizedWeights(
            finalized_block=revealed,
            extrinsic_hash=receipt.submitted_extrinsic_hash,
            genesis_hash=context.genesis_hash,
            runtime_hash=context.runtime_hash,
            mechanism=context.mechanism,
            # CRV3 records LastUpdate at commit, not the automatic reveal.
            last_update_block=receipt.submitted_block,
            last_step_block=revealed,
            weights_u16={int(uid): int(weight) for uid, weight in rows},
            registered_uids=self.chain.uid_snapshot(revealed),
        )
