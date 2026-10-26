from dataclasses import replace

import pytest

pytest.importorskip("bittensor")

from bittensor.keyfiles import Keypair

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.protocol.evaluation_record import (
    EvaluationRecord,
    SignedEvaluationRecord,
    create_record,
    sign_record,
    verify_record,
)


def _sha(number: int) -> str:
    return f"{number:064x}"


class _Wallet:
    def __init__(self, keypair):
        self.hotkey = keypair


def _inputs():
    candidate, incumbent = _sha(1), _sha(2)
    plan = {
        "network": "test",
        "netuid": 578,
        "round_id": "r-evidence-1",
        "close_block": 100,
        "admitted": [{"manifest_hash": candidate, "hotkey": "miner-public", "block": 99}],
        "candidate": {
            "manifest_hash": candidate,
            "hotkey": "miner-public",
            "admission_block": 99,
        },
        "incumbent": {
            "manifest_hash": incumbent,
            "certified_state_hash": _sha(3),
            "certified_activation_block": 90,
        },
    }
    policy = {"evaluator_id": "luna-vision/v1"}
    quality = {
        "evaluator_id": "luna-vision/v1",
        "comparisons": [
            {
                "reference_role": "golden",
                "reference_manifest_hash": _sha(4),
                "candidate_manifest_hash": candidate,
                "reference_admission": "public_snapshot",
                "candidate_admission": "public_snapshot",
            },
            {
                "reference_role": "incumbent",
                "reference_manifest_hash": incumbent,
                "candidate_manifest_hash": candidate,
                "reference_admission": "public_snapshot",
                "candidate_admission": "public_snapshot",
            },
        ],
    }
    timing = {
        "reference_manifest_hash": incumbent,
        "candidate_manifest_hash": candidate,
        "reference_admission": "public_snapshot",
        "candidate_admission": "public_snapshot",
    }
    return plan, policy, quality, timing, {"schema": "incomplete-calibration"}


def test_signed_incomplete_evaluation_record_is_replay_bound_and_not_a_crown():
    record = create_record(*_inputs(), evaluator_journal_sha256=_sha(9))
    assert record.gate_status == "INCOMPLETE"
    assert record.public_admission is True
    assert record.candidate_manifest_hash == _sha(1)

    signer = Sr25519Signer(_Wallet(Keypair.create_from_uri("//Alice")))
    signed = sign_record(record, signer)
    verifier = Sr25519Verifier()
    assert verify_record(signed, verifier)
    assert verify_record(SignedEvaluationRecord.from_dict(signed.to_dict()), verifier)
    assert EvaluationRecord.from_dict(record.to_dict()).record_hash() == record.record_hash()

    for tampered in (
        replace(record, network="main"),
        replace(record, netuid=579),
        replace(record, round_id="r-evidence-2"),
        replace(record, candidate_manifest_hash=_sha(8)),
    ):
        assert not verify_record(replace(signed, record=tampered), verifier)
