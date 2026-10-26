"""Durable signed message exchange; transport does not attest to model execution.

Exported envelopes can be exchanged over an operator-owned authenticated channel.
Import rechecks signatures and round bindings. Missing messages expire the local
round after finalized deadline; they never create a KEEP or PROMOTE certificate.
"""

from __future__ import annotations

import json
from typing import Any

from forge.protocol.consensus_types import RoundPlan
from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.ports import Signer, Verifier
from forge.protocol.store import Store

SCHEMA = "forge/round-envelope/v1"


def envelope_payload(body: dict[str, Any]) -> bytes:
    return canonical_json_bytes({"schema": SCHEMA, **body})


def sign_envelope(
    plan: RoundPlan, signer: Signer, kind: str, payload: dict[str, Any]
) -> dict[str, Any]:
    if kind not in {"commit", "reveal", "certificate"}:
        raise ValueError("unsupported round message kind")
    if signer.validator_hotkey not in plan.committee:
        raise ValueError("sender is not in frozen committee")
    body = {
        "plan_hash": plan.plan_hash(),
        "sender": signer.validator_hotkey,
        "kind": kind,
        "payload": payload,
    }
    return {"schema": SCHEMA, **body, "signature": signer.sign(envelope_payload(body)).hex()}


class RoundMailbox:
    def __init__(self, store: Store, verifier: Verifier):
        self.store, self.verifier = store, verifier
        self.store.connection.executescript("""
            CREATE TABLE IF NOT EXISTS round_messages (
                plan_hash TEXT NOT NULL, sender TEXT NOT NULL, kind TEXT NOT NULL,
                message_hash TEXT NOT NULL, envelope_json TEXT NOT NULL,
                received_finalized_block INTEGER NOT NULL,
                PRIMARY KEY(plan_hash, sender, kind)
            );
            CREATE TABLE IF NOT EXISTS round_closures (
                plan_hash TEXT PRIMARY KEY, receipt_json TEXT NOT NULL
            );
        """)

    def receive(self, plan: RoundPlan, envelope: dict[str, Any], *, finalized_block: int) -> bool:
        # Finality/height comes from the validator chain adapter, never the envelope.
        if type(finalized_block) is not int or finalized_block < plan.snapshot_block:
            raise ValueError("invalid finalized receipt block")
        if set(envelope) != {"schema", "plan_hash", "sender", "kind", "payload", "signature"}:
            raise ValueError("invalid envelope fields")
        sender, kind = envelope["sender"], envelope["kind"]
        if (
            envelope["schema"] != SCHEMA
            or envelope["plan_hash"] != plan.plan_hash()
            or sender not in plan.committee
            or kind not in {"commit", "reveal", "certificate"}
            or not isinstance(envelope["payload"], dict)
        ):
            raise ValueError("envelope is outside the frozen round")
        body = {key: envelope[key] for key in ("plan_hash", "sender", "kind", "payload")}
        try:
            signature = bytes.fromhex(envelope["signature"])
        except (ValueError, TypeError):
            raise ValueError("malformed envelope signature") from None
        if not self.verifier.verify(sender, envelope_payload(body), signature):
            raise ValueError("invalid envelope signature")
        digest = hash_canonical(SCHEMA, body)
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT message_hash FROM round_messages WHERE plan_hash=? AND sender=? AND kind=?",
                (plan.plan_hash(), sender, kind),
            ).fetchone()
            if row:
                if row["message_hash"] != digest:
                    raise ValueError("conflicting signed round message")
                return False
            if db.execute(
                "SELECT 1 FROM round_closures WHERE plan_hash=?", (plan.plan_hash(),)
            ).fetchone():
                raise ValueError("round is locally closed")
            deadline = (
                plan.commit_deadline_block if kind == "commit" else plan.reveal_deadline_block
            )
            if kind != "certificate" and finalized_block > deadline:
                raise ValueError("message arrived after finalized deadline")
            if kind == "reveal" and finalized_block <= plan.commit_deadline_block:
                raise ValueError("reveal arrived before commit window closed")
            db.execute(
                "INSERT INTO round_messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    plan.plan_hash(),
                    sender,
                    kind,
                    digest,
                    json.dumps(envelope, sort_keys=True, allow_nan=False),
                    finalized_block,
                ),
            )
        return True

    def export(self, plan: RoundPlan) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self.store.connection.execute(
                "SELECT envelope_json FROM round_messages WHERE plan_hash=? ORDER BY sender, kind",
                (plan.plan_hash(),),
            )
        ]

    def expire(
        self, plan: RoundPlan, *, finalized_block: int, finalized_hash: str
    ) -> dict[str, Any]:
        if type(finalized_block) is not int or finalized_block <= plan.reveal_deadline_block:
            raise ValueError("reveal deadline has not finalized")
        if len(finalized_hash.removeprefix("0x")) != 64:
            raise ValueError("finalized block hash is required")
        try:
            bytes.fromhex(finalized_hash.removeprefix("0x"))
        except ValueError:
            raise ValueError("invalid finalized block hash") from None
        receipt = {
            "schema": "forge/round-closure/v1",
            "plan_hash": plan.plan_hash(),
            "status": "EXPIRED_NO_CERTIFICATE",
            "finalized_block": finalized_block,
            "finalized_hash": finalized_hash,
            "winner_changed": False,
            "messages_hash": hash_canonical("forge/mailbox/v1", self.export(plan)),
        }
        with self.store.transaction() as db:
            old = db.execute(
                "SELECT receipt_json FROM round_closures WHERE plan_hash=?", (plan.plan_hash(),)
            ).fetchone()
            if old:
                return json.loads(old[0])
            if db.execute(
                "SELECT 1 FROM certificates WHERE round_id=?", (plan.round_id,)
            ).fetchone():
                raise ValueError("certified round cannot expire")
            db.execute(
                "INSERT INTO round_closures VALUES (?, ?)",
                (plan.plan_hash(), json.dumps(receipt, sort_keys=True)),
            )
            db.execute(
                "UPDATE rounds SET status='REJECTED' WHERE round_id=? AND status='PENDING'",
                (plan.round_id,),
            )
        return receipt
