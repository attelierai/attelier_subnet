"""Durable, fail-closed publication of a qualified testnet weight intent.

This is deliberately a small boundary around the existing Bittensor SDK adapter.
It never invents a winner, signs an activation plan, or treats an inclusion result
as effective weights.  A caller must supply an activation-qualified context and
an adapter which can both reconcile an uncertain submission and read a finalized
weight vector.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from forge.chain.reward_publication import (
    WeightIntent,
    WeightSubmissionReceipt,
    verify_finalized_replacement,
)
from forge.protocol.hashing import hash_canonical


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class QualifiedPublicationContext:
    """The authority-bearing identities frozen by an external activation plan."""

    mode: str
    network: str
    netuid: int
    genesis_hash: str
    runtime_hash: str
    mechanism: int
    contract_hash: str
    plan_hash: str
    certified_state_hash: str
    certificate_hash: str | None
    activation_plan_hash: str

    def __post_init__(self) -> None:
        if self.mode != "bittensor-sdk/testnet-qualified-v1":
            raise ValueError("publication mode is not qualified")
        if self.network != "test":
            raise ValueError("qualified publisher currently supports testnet only")
        if not isinstance(self.network, str) or not self.network:
            raise ValueError("network is required")
        if type(self.netuid) is not int or self.netuid < 0:
            raise ValueError("netuid must be nonnegative")
        if type(self.mechanism) is not int or self.mechanism < 0:
            raise ValueError("mechanism must be nonnegative")
        if self.mechanism != 0:
            raise ValueError("qualified testnet publisher currently supports mechanism 0 only")
        for name in (
            "genesis_hash",
            "runtime_hash",
            "contract_hash",
            "plan_hash",
            "certified_state_hash",
            "activation_plan_hash",
        ):
            _hash(getattr(self, name), name)
        if self.certificate_hash is not None:
            _hash(self.certificate_hash, "certificate hash")

    def context_hash(self) -> str:
        return hash_canonical("forge/qualified-publication-context/v1", asdict(self))


@dataclass(frozen=True)
class BootstrapAuthority:
    """Signed external authorization for the no-certificate bootstrap fallback."""

    authority_id: str
    signature: str
    genesis_hash: str
    activation_plan_hash: str
    fallback_qualification_hash: str
    target_uid: int
    target_hotkey: str

    def __post_init__(self) -> None:
        if not isinstance(self.authority_id, str) or not self.authority_id:
            raise ValueError("bootstrap authority id is required")
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("bootstrap authority signature is required")
        for name in ("genesis_hash", "activation_plan_hash", "fallback_qualification_hash"):
            _hash(getattr(self, name), name)
        if type(self.target_uid) is not int or self.target_uid < 0 or not self.target_hotkey:
            raise ValueError("bootstrap fallback target is invalid")

    def signed_payload(self) -> bytes:
        return json.dumps(
            {
                "authority_id": self.authority_id,
                "genesis_hash": self.genesis_hash,
                "activation_plan_hash": self.activation_plan_hash,
                "fallback_qualification_hash": self.fallback_qualification_hash,
                "target_uid": self.target_uid,
                "target_hotkey": self.target_hotkey,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


@dataclass(frozen=True)
class FinalizedWeights:
    finalized_block: int
    extrinsic_hash: str
    genesis_hash: str
    runtime_hash: str
    mechanism: int
    last_update_block: int
    last_step_block: int
    weights_u16: dict[int, int]
    registered_uids: dict[int, str]


class PublicationAdapter(Protocol):
    """Network boundary. Implementations own wallets and RPC transport, never callers."""

    def submit(
        self, context: QualifiedPublicationContext, intent: WeightIntent
    ) -> WeightSubmissionReceipt: ...

    def reconcile(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt | None,
    ) -> WeightSubmissionReceipt | None: ...

    def finalized_weights(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt,
    ) -> FinalizedWeights | None: ...


class JsonHttpPublicationAdapter:
    """Small HTTP transport for a separately supervised testnet signer process.

    The signer service is intentionally outside this process.  It must expose
    only the three operations below and retain its own wallet/key handling.
    This client has no fallback to an SDK call when a reconciliation is unknown.
    """

    def __init__(self, endpoint: str, *, timeout_seconds: float = 10.0) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("publication endpoint must be HTTP(S)")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _call(
        self,
        operation: str,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt | None = None,
    ) -> dict | None:
        payload = {
            "context": asdict(context),
            "intent": asdict(intent),
            "receipt": asdict(receipt) if receipt else None,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/{operation}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status == 204:
                    return None
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise RuntimeError(
                f"publication service {operation} failed: HTTP {error.code}"
            ) from error
        except TimeoutError:
            raise
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("publication service timed out") from error
            raise RuntimeError(f"publication service {operation} is unavailable") from error

    def submit(
        self, context: QualifiedPublicationContext, intent: WeightIntent
    ) -> WeightSubmissionReceipt:
        value = self._call("submit", context, intent)
        if value is None:
            raise TimeoutError("signer returned no durable submission receipt")
        return WeightSubmissionReceipt(**value)

    def reconcile(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt | None,
    ) -> WeightSubmissionReceipt | None:
        value = self._call("reconcile", context, intent, receipt)
        return WeightSubmissionReceipt(**value) if value else None

    def finalized_weights(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt,
    ) -> FinalizedWeights | None:
        value = self._call("finalized", context, intent, receipt)
        if value:
            value["weights_u16"] = {
                int(uid): weight for uid, weight in value["weights_u16"].items()
            }
            value["registered_uids"] = {
                int(uid): hotkey for uid, hotkey in value["registered_uids"].items()
            }
        return FinalizedWeights(**value) if value else None


class BittensorSdkTestnetSigner:
    """Persist the SDK's exact CRV3 ciphertext before a single hotkey submission.

    SDK ``plan`` applies its normal registration/rate/weight/policy checks without
    signing. Its already-composed call goes to the same SDK substrate submit
    path used by execute, so encryption is never rebuilt after the durable witness.
    """

    def __init__(self, chain: Any, wallet: Any, reader: Any) -> None:
        from forge.chain.launch_native_publication import TestnetNativePublicationReader

        if not isinstance(reader, TestnetNativePublicationReader) or reader.chain is not chain:
            raise ValueError("qualified signer requires its pinned native testnet reader")
        self.chain, self.wallet, self.reader = chain, wallet, reader

    @classmethod
    def with_native_reader(cls, chain: Any, wallet: Any, reader: Any) -> BittensorSdkTestnetSigner:
        return cls(chain, wallet, reader)

    def _identity(self, context: QualifiedPublicationContext) -> None:
        if (getattr(self.chain, "network", None), getattr(self.chain, "netuid", None)) != (
            context.network,
            context.netuid,
        ):
            raise ValueError("configured BittensorChain belongs to another subnet")

    def submit(
        self, context: QualifiedPublicationContext, intent: WeightIntent
    ) -> WeightSubmissionReceipt:
        from bittensor.intents.weights import SetWeights
        from bittensor.settings import DEFAULT_ERA_PERIOD
        from bittensor.signing import resolve_signer

        from forge.chain import WeightSubmission

        self._identity(context)
        if (intent.network, intent.netuid, intent.runtime_hash) != (
            context.network,
            context.netuid,
            context.runtime_hash,
        ):
            raise ValueError("weight intent belongs to another qualified context")
        uids = sorted(intent.uid_weights)
        if not uids:
            raise ValueError("refusing an empty weight vector")
        weights = [float(intent.uid_weights[uid]) for uid in uids]
        plan = self.chain.subtensor.plan(
            SetWeights(netuid=context.netuid, uids=uids, weights=weights, mechid=context.mechanism),
            self.wallet,
        )
        if not plan.ok:
            raise ValueError("SDK weight plan violates its active policy")
        if plan.signer != "hotkey" or plan.signer_address != self.reader.validator_hotkey:
            raise ValueError("SDK weight signer differs from the native reader hotkey")
        self.reader.capture_prepared(context, intent, plan.call)
        result = self.chain.subtensor._call(
            self.chain.subtensor._client._substrate.submit(
                plan.call,
                resolve_signer(self.wallet, "hotkey"),
                period=DEFAULT_ERA_PERIOD,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
        )
        if not result.success:
            raise RuntimeError(f"set_weights failed: {result.message}")
        submission = WeightSubmission(
            context.netuid,
            uids,
            weights,
            False,
            str(result.extrinsic_id),
            context.mechanism,
        )
        receipt = self.reader.receipt_from_submission(submission, context, intent)
        if receipt.intent_hash != intent.intent_hash() or (receipt.network, receipt.netuid) != (
            context.network,
            context.netuid,
        ):
            raise ValueError("SDK submission receipt is not bound to this intent")
        return receipt

    def reconcile(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt | None,
    ) -> WeightSubmissionReceipt | None:
        self._identity(context)
        return self.reader.reconcile(context, intent, receipt)

    def finalized_weights(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        receipt: WeightSubmissionReceipt,
    ) -> FinalizedWeights | None:
        self._identity(context)
        return self.reader.finalized_weights(context, intent, receipt)


class LaunchPublicationStore:
    """SQLite journal. The intent row is committed before the external call."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS launch_publication_intents (
                intent_hash TEXT PRIMARY KEY, context_hash TEXT NOT NULL, intent_json TEXT NOT NULL,
                context_json TEXT NOT NULL, receipt_json TEXT, status TEXT NOT NULL,
                last_error TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def prepare(self, context: QualifiedPublicationContext, intent: WeightIntent) -> str:
        intent_hash = intent.intent_hash()
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO launch_publication_intents "
                "(intent_hash, context_hash, intent_json, context_json, status) "
                "VALUES (?, ?, ?, ?, 'PREPARED')",
                (
                    intent_hash,
                    context.context_hash(),
                    json.dumps(asdict(intent), sort_keys=True),
                    json.dumps(asdict(context), sort_keys=True),
                ),
            )
            row = db.execute(
                "SELECT context_hash, intent_json FROM launch_publication_intents "
                "WHERE intent_hash=?",
                (intent_hash,),
            ).fetchone()
        if (
            row is None
            or row["context_hash"] != context.context_hash()
            or row["intent_json"] != json.dumps(asdict(intent), sort_keys=True)
        ):
            raise ValueError("persisted publication intent conflicts with this authority context")
        return intent_hash

    def load(self, intent_hash: str) -> tuple[str, WeightSubmissionReceipt | None]:
        with self._connect() as db:
            row = db.execute(
                "SELECT status, receipt_json FROM launch_publication_intents WHERE intent_hash=?",
                (intent_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("publication intent was not persisted")
        receipt = (
            WeightSubmissionReceipt(**json.loads(row["receipt_json"]))
            if row["receipt_json"]
            else None
        )
        return str(row["status"]), receipt

    def claim_submission(self, intent_hash: str) -> bool:
        with self._connect() as db:
            return (
                db.execute(
                    "UPDATE launch_publication_intents SET status='SUBMITTING' "
                    "WHERE intent_hash=? AND status='PREPARED'",
                    (intent_hash,),
                ).rowcount
                == 1
            )

    def update(
        self,
        intent_hash: str,
        status: str,
        receipt: WeightSubmissionReceipt | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE launch_publication_intents SET status=?, "
                "receipt_json=COALESCE(?, receipt_json), last_error=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE intent_hash=?",
                (
                    status,
                    json.dumps(asdict(receipt), sort_keys=True) if receipt else None,
                    error,
                    intent_hash,
                ),
            )


class LaunchPublisher:
    def __init__(
        self,
        store: LaunchPublicationStore,
        adapter: PublicationAdapter,
        verify_publication_authority: Callable[[QualifiedPublicationContext, WeightIntent], bool],
        verify_bootstrap_signature: Callable[[BootstrapAuthority], bool],
        verify_fallback_qualification: Callable[
            [QualifiedPublicationContext, BootstrapAuthority], bool
        ],
        verify_certified_fallback: Callable[[QualifiedPublicationContext, WeightIntent], bool],
    ) -> None:
        self.store, self.adapter = store, adapter
        self.verify_publication_authority = verify_publication_authority
        self.verify_bootstrap_signature = verify_bootstrap_signature
        self.verify_fallback_qualification = verify_fallback_qualification
        self.verify_certified_fallback = verify_certified_fallback

    def _authorize(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        bootstrap: BootstrapAuthority | None,
    ) -> None:
        if not self.verify_publication_authority(context, intent):
            raise ValueError("qualified publication authority is not trusted")
        if (intent.network, intent.netuid, intent.runtime_hash, intent.certified_state_hash) != (
            context.network,
            context.netuid,
            context.runtime_hash,
            context.certified_state_hash,
        ):
            raise ValueError("weight intent is not bound to the qualified context")
        if intent.status == "WINNER":
            if (
                context.certificate_hash is None
                or intent.certificate_hash != context.certificate_hash
            ):
                raise ValueError("winner publication requires the bound certificate")
            return
        if intent.status != "BURN_REQUIRED":
            raise ValueError("weight intent is not publishable")
        if context.certificate_hash is not None:
            if (
                intent.certificate_hash != context.certificate_hash
                or intent.burn_qualification_hash is None
            ):
                raise ValueError("certified fallback is not bound to its qualification")
            if not self.verify_certified_fallback(context, intent):
                raise ValueError("certified fallback is not qualified for this runtime and target")
            return
        if bootstrap is None:
            raise ValueError("no-winner publication requires authenticated bootstrap authority")
        if (bootstrap.genesis_hash, bootstrap.activation_plan_hash) != (
            context.genesis_hash,
            context.activation_plan_hash,
        ):
            raise ValueError("bootstrap authority is not bound to campaign activation")
        if intent.uid_weights != {bootstrap.target_uid: 1.0} or intent.expected_hotkeys != {
            bootstrap.target_uid: bootstrap.target_hotkey
        }:
            raise ValueError("bootstrap fallback differs from the signed target")
        if not self.verify_bootstrap_signature(bootstrap):
            raise ValueError("bootstrap authority signature is not trusted")
        if not self.verify_fallback_qualification(context, bootstrap):
            raise ValueError("fallback is not qualified for this runtime and target")

    def publish_or_reconcile(
        self,
        context: QualifiedPublicationContext,
        intent: WeightIntent,
        *,
        bootstrap: BootstrapAuthority | None = None,
    ) -> str:
        self._authorize(context, intent, bootstrap)
        intent_hash = self.store.prepare(context, intent)
        status, receipt = self.store.load(intent_hash)
        if status == "FINALIZED":
            return status
        # A restart or a timeout always gets a read/reconciliation before another submit.
        if status != "PREPARED":
            receipt = self.adapter.reconcile(context, intent, receipt)
            if receipt is None:
                self.store.update(
                    intent_hash, "SUBMISSION_UNKNOWN", error="submission not yet reconciled"
                )
                return "SUBMISSION_UNKNOWN"
            self.store.update(intent_hash, "SUBMITTED", receipt)
        if receipt is None:
            if not self.store.claim_submission(intent_hash):
                return "SUBMISSION_UNKNOWN"
            try:
                receipt = self.adapter.submit(context, intent)
            except Exception as error:
                self.store.update(intent_hash, "SUBMISSION_UNKNOWN", error=str(error))
                receipt = self.adapter.reconcile(context, intent, None)
                if receipt is None:
                    return "SUBMISSION_UNKNOWN"
            self.store.update(intent_hash, "SUBMITTED", receipt)
        finalized = self.adapter.finalized_weights(context, intent, receipt)
        if finalized is None:
            return "SUBMITTED"
        if (
            finalized.genesis_hash != context.genesis_hash
            or finalized.mechanism != context.mechanism
        ):
            raise ValueError("finalized readback does not prove the new vector became effective")
        if (
            finalized.finalized_block < receipt.submitted_block
            or finalized.last_update_block < receipt.submitted_block
            or finalized.last_step_block <= receipt.submitted_block
        ):
            return "SUBMITTED"
        verify_finalized_replacement(
            intent,
            receipt,
            network=context.network,
            netuid=context.netuid,
            finalized_block=finalized.finalized_block,
            finalized_extrinsic_hash=finalized.extrinsic_hash,
            finalized_runtime_hash=finalized.runtime_hash,
            weights_u16=finalized.weights_u16,
            registered_uids=finalized.registered_uids,
        )
        self.store.update(intent_hash, "FINALIZED", receipt)
        return "FINALIZED"
