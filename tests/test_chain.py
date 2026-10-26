import pytest

pytest.importorskip("bittensor")

from bittensor.keyfiles import Keypair  # noqa: E402

from forge.chain import (  # noqa: E402
    Sr25519Verifier,
    decode_commitment,
    encode_evaluation_record_commitment,
    encode_prompt_pack_commit,
    encode_report_commit,
    encode_submission,
)
from forge.models.artifact_manifest import ArtifactReference
from forge.validator.board import FileBoard  # noqa: E402
from forge.validator.submission import (  # noqa: E402
    V3SubmissionClaim,
    encode_v3_submission_commitment,
)


def test_commitment_round_trip_and_foreign_content_is_ignored():
    fingerprint = "ab" * 32
    decoded = decode_commitment("hk", 7, encode_submission(fingerprint))
    assert (decoded.kind, decoded.fields, decoded.block) == ("sub", (fingerprint,), 7)
    report = decode_commitment("hk", 9, encode_report_commit("r100", "cd" * 32))
    assert report.fields == ("r100", "cd" * 32)
    assert decode_commitment("hk", 1, "some other subnet payload") is None
    assert decode_commitment("hk", 1, "forge:v1:sub:a:b") is None


def test_commitment_limits():
    with pytest.raises(ValueError):
        encode_submission("x" * 200)
    with pytest.raises(ValueError):
        encode_report_commit("r:1", "cd" * 32)


def test_v3_model_and_prompt_pack_commitments_keep_their_distinct_kinds():
    claim = V3SubmissionClaim(
        ArtifactReference("hf", "public/wan-lora", "a" * 40),
        "lora",
        {"steps": 16, "guidance_scale": 5.0},
    )
    model = decode_commitment("hk", 7, encode_v3_submission_commitment(claim))
    assert model is not None
    assert model.kind == "model_v3"
    assert model.fields == ("hf", "public/wan-lora", "a" * 40, "lora", "16", "5")
    pack = decode_commitment("hk", 8, encode_prompt_pack_commit("cd" * 32))
    assert pack is not None
    assert (pack.kind, pack.fields) == ("pack", ("cd" * 32,))
    evaluation = decode_commitment("hk", 9, encode_evaluation_record_commitment("ef" * 32))
    assert evaluation is not None
    assert (evaluation.kind, evaluation.fields) == ("evaluation", ("ef" * 32,))


class _Wallet:
    def __init__(self, keypair):
        self.hotkey = keypair


def test_sr25519_signatures_verify_from_public_address_only():
    from forge.chain import Sr25519Signer

    signer = Sr25519Signer(_Wallet(Keypair.create_from_uri("//Alice")))
    signature = signer.sign(b"certificate")
    verifier = Sr25519Verifier()
    assert verifier.verify(signer.validator_hotkey, b"certificate", signature)
    assert not verifier.verify(signer.validator_hotkey, b"tampered", signature)
    bob = Keypair.create_from_uri("//Bob").ss58_address
    assert not verifier.verify(bob, b"certificate", signature)


def test_board_is_write_once(tmp_path):
    board = FileBoard(tmp_path)
    assert board.put("reveal", "r1", "hk", {"a": 1})
    assert not board.put("reveal", "r1", "hk", {"a": 1})
    with pytest.raises(FileExistsError):
        board.put("reveal", "r1", "hk", {"a": 2})
    assert board.list("reveal", "r1") == {"hk": {"a": 1}}
    with pytest.raises(ValueError):
        board.put("reveal", "../x", "hk", {})
