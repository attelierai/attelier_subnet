"""Real sr25519 committee admission-history verification across SQLite restart."""

from types import SimpleNamespace

import pytest

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.protocol.admission_checkpoint import (
    SignedAdmissionHistory,
    create_signed_history,
    history_verifier,
    verify_signed_history,
)
from forge.validator.admission import (
    AdmissionError,
    AdmissionLedger,
    FinalizedCommitment,
    build_authenticated_history,
)


def digest(value: int) -> str:
    return f"{value:064x}"


def signers():
    from bittensor.keyfiles import Keypair

    return [
        Sr25519Signer(SimpleNamespace(hotkey=Keypair.create_from_uri(uri)))
        for uri in ("//Alice", "//Bob", "//Charlie")
    ]


def checkpoint():
    return build_authenticated_history(
        network="test",
        netuid=9,
        through_block=100,
        certified_state_hash=digest(1),
        consumed_hotkeys=(),
        registered_hotkeys=("new", "old"),
        lineage_edges=(("old", "new"),),
    )


def commitment(hotkey: str, index: int) -> FinalizedCommitment:
    return FinalizedCommitment(hotkey, 99, index, digest(10), digest(11), digest(12), digest(13))


def test_signed_checkpoint_requires_more_than_two_thirds_of_trusted_committee(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    history = checkpoint()
    signed = create_signed_history(history, validators)
    assert verify_signed_history(
        history,
        signed,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )
    only_two = create_signed_history(history, validators[:2])
    assert not verify_signed_history(
        history,
        only_two,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )
    windowed = build_authenticated_history(
        network="test",
        netuid=9,
        through_block=100,
        certified_state_hash=digest(1),
        consumed_hotkeys=(),
        registered_hotkeys=("new", "old"),
        lineage_edges=(("old", "new"),),
        window_id="test/9/100",
        window_open_block=0,
        window_close_block=100,
    )
    signed_windowed = create_signed_history(windowed, validators)
    assert verify_signed_history(
        windowed,
        signed_windowed,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )
    retargeted = build_authenticated_history(
        network="test",
        netuid=9,
        through_block=100,
        certified_state_hash=digest(1),
        consumed_hotkeys=(),
        registered_hotkeys=("new", "old"),
        lineage_edges=(("old", "new"),),
        window_id="test/9/101",
        window_open_block=0,
        window_close_block=100,
    )
    assert not verify_signed_history(
        retargeted,
        signed_windowed,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )
    forged = SignedAdmissionHistory(history, {**signed.signatures, "outside": b"x"})
    assert not verify_signed_history(
        history,
        forged,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )
    wrong_committee = dict(committee)
    wrong_committee.pop(validators[-1].validator_hotkey)
    wrong_committee["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehU6fh5ctu2ACo"] = 1
    assert not verify_signed_history(
        history,
        signed,
        trusted_committee=wrong_committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )


def test_tamper_wrong_state_and_swap_consumption_fail_closed_across_restart(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    history = checkpoint()
    signed = create_signed_history(history, validators)
    verify = history_verifier(
        signed,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        verifier=Sr25519Verifier(),
    )
    assert not verify_signed_history(
        history,
        signed,
        trusted_committee=committee,
        expected_certified_state_hash=digest(99),
        verifier=Sr25519Verifier(),
    )
    altered = build_authenticated_history(
        network="test",
        netuid=9,
        through_block=100,
        certified_state_hash=digest(1),
        consumed_hotkeys=(),
        registered_hotkeys=("new", "old"),
        lineage_edges=(),
    )
    assert not verify(altered)
    path = tmp_path / "validator.sqlite"
    ledger = AdmissionLedger(path, network="test", netuid=9)
    ledger.admit(commitment("old", 1), history=history, verify_history=verify)
    reopened = AdmissionLedger(path, network="test", netuid=9)
    with pytest.raises(AdmissionError, match="lineage was already consumed"):
        reopened.admit(commitment("new", 2), history=history, verify_history=verify)
