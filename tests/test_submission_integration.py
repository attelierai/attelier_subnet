import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from forge.chain import decode_commitment
from forge.models.artifact_manifest import (
    ArtifactError,
    ArtifactReference,
    ManifestFile,
    ModelManifest,
)
from forge.protocol.contract import ChallengeContract
from forge.validator.submission import (
    MAX_COMMITMENT_BYTES,
    V3SubmissionClaim,
    claim_from_commitment_fields,
    decode_artifact_commitment,
    decode_submission_commitment,
    decode_v3_submission_commitment,
    encode_artifact_commitment,
    encode_v3_submission_commitment,
    prepare_declaration,
    resolve_submission,
    resolve_v3_submission,
)

REPO = "public/wan-lora"
REVISION = "a" * 40
WEIGHTS = b"public model weights"
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()


def _declaration():
    return {
        "schema": "forge-manifest/v1",
        "role": "lora",
        "files": [
            {"path": "adapter/model.safetensors", "sha256": WEIGHTS_SHA256, "bytes": len(WEIGHTS)}
        ],
        "license": "apache-2.0",
        "generation_config": {"steps": 16, "guidance_scale": 5.0},
        "base_reference": {
            "backend": "hf",
            "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "revision": "b" * 40,
        },
    }


@contextmanager
def submission_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith(f"/api/models/{REPO}/revision/{REVISION}"):
                body = json.dumps(
                    {
                        "private": False,
                        "gated": False,
                        "disabled": False,
                        "sha": REVISION,
                        "cardData": {"license": "apache-2.0"},
                        "siblings": [
                            {"rfilename": "forge-manifest.json"},
                            {
                                "rfilename": "adapter/model.safetensors",
                                "size": len(WEIGHTS),
                                "lfs": {"sha256": WEIGHTS_SHA256, "size": len(WEIGHTS)},
                            },
                        ],
                    }
                ).encode()
            elif self.path == f"/{REPO}/resolve/{REVISION}/forge-manifest.json":
                body = json.dumps(_declaration()).encode()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_commitment_resolves_a_public_immutable_manifest_without_weights():
    reference = ArtifactReference("hf", REPO, REVISION)
    commitment = encode_artifact_commitment(reference)
    assert commitment == f"forge:v2:h:{REPO}:{REVISION}"
    assert decode_artifact_commitment(commitment) == reference
    with submission_server() as server:
        manifest = resolve_submission(reference, hf_base_url=server)
    assert manifest.reference == reference
    assert manifest.files[0].sha256 == WEIGHTS_SHA256


def test_commitment_rejects_ambiguous_repositories_and_decodes_hippius_digest():
    assert decode_artifact_commitment(f"forge:v2:h:public/repo@alias:{REVISION}") is None
    with pytest.raises(ArtifactError, match="repo_id"):
        encode_artifact_commitment(ArtifactReference("hf", "public/repo@alias", REVISION))
    hippius = ArtifactReference("hippius", REPO, f"sha256:{'c' * 64}")
    assert decode_artifact_commitment(encode_artifact_commitment(hippius)) == hippius


def test_prepare_declaration_has_the_exact_upload_schema_without_self_reference():
    manifest = ModelManifest(
        ArtifactReference("hf", REPO, REVISION),
        "lora",
        (ManifestFile("adapter/model.safetensors", WEIGHTS_SHA256, len(WEIGHTS)),),
        "apache-2.0",
        {"steps": 4, "guidance_scale": 5.0},
        ArtifactReference("hf", "Wan-AI/Wan2.1-T2V-1.3B", "b" * 40),
    )
    declaration = prepare_declaration(manifest)
    assert set(declaration) == {
        "schema",
        "role",
        "files",
        "license",
        "generation_config",
        "base_reference",
    }
    assert "reference" not in declaration
    assert declaration["files"][0]["sha256"] == WEIGHTS_SHA256


def test_v3_hf_claim_derives_public_manifest_and_rejects_recipe_aliases():
    claim = V3SubmissionClaim(
        ArtifactReference("hf", REPO, REVISION),
        "lora",
        {"steps": 16, "guidance_scale": 5.0},
    )
    commitment = encode_v3_submission_commitment(claim)
    assert commitment == f"forge:v3:h:{REPO}:{REVISION}:l:16:5"
    assert decode_v3_submission_commitment(commitment) == claim
    assert decode_submission_commitment(commitment) == claim
    assert decode_v3_submission_commitment(commitment.replace(":5", ":5.0")) is None
    assert decode_v3_submission_commitment(commitment.replace(":16:", ":016:")) is None

    with submission_server() as server:
        manifest = resolve_v3_submission(claim, ChallengeContract(max_steps=16), hf_base_url=server)
    assert manifest.role == "lora"
    assert manifest.base_reference == ArtifactReference(
        "hf", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"
    )
    assert manifest.files == (
        ManifestFile("adapter/model.safetensors", WEIGHTS_SHA256, len(WEIGHTS)),
    )
    with pytest.raises(ArtifactError, match="maximum steps"):
        resolve_v3_submission(claim, ChallengeContract(max_steps=15), hf_base_url=server)
    with pytest.raises(ArtifactError, match="Hugging Face"):
        V3SubmissionClaim(
            ArtifactReference("hippius", REPO, f"sha256:{'c' * 64}"),
            "lora",
            {"steps": 16, "guidance_scale": 5.0},
        )


def test_v3_commitment_uses_the_full_raw_field_without_ambiguity():
    maximum = V3SubmissionClaim(
        ArtifactReference("hf", f"a/{'b' * 64}", REVISION),
        "lora",
        {"steps": 100, "guidance_scale": 9.9},
    )
    assert len(encode_v3_submission_commitment(maximum).encode()) == MAX_COMMITMENT_BYTES
    too_long = V3SubmissionClaim(
        ArtifactReference("hf", f"a/{'b' * 65}", REVISION),
        "lora",
        {"steps": 100, "guidance_scale": 9.9},
    )
    with pytest.raises(ArtifactError, match="128"):
        encode_v3_submission_commitment(too_long)


def test_v3_claim_keeps_the_validated_recipe_after_caller_mutation():
    recipe = {"steps": 16, "guidance_scale": 5.0}
    claim = V3SubmissionClaim(ArtifactReference("hf", REPO, REVISION), "lora", recipe)
    recipe["steps"] = 100
    assert encode_v3_submission_commitment(claim).endswith(":l:16:5")


def test_scheduler_commitment_survives_chain_and_public_resolution():
    manifests = []
    for scheduler, token in (("original_unipc", "u"), ("uniform_euler", "e")):
        claim = V3SubmissionClaim(
            ArtifactReference("hf", REPO, REVISION),
            "lora",
            {"steps": 40, "guidance_scale": 5.0, "scheduler": scheduler},
        )
        wire = encode_v3_submission_commitment(claim)
        assert wire == f"forge:v4:h:{REPO}:{REVISION}:l:40:5:{token}"
        assert decode_v3_submission_commitment(wire) is None
        assert decode_submission_commitment(wire) == claim
        commitment = decode_commitment("registered-hotkey", 123, wire)
        assert commitment.kind == "model_v4"
        replayed = claim_from_commitment_fields(commitment.kind, commitment.fields)
        assert replayed == claim
        with submission_server() as server:
            manifest = resolve_v3_submission(
                replayed, ChallengeContract(max_steps=50), hf_base_url=server
            )
        assert manifest.generation_config == claim.generation_config
        assert manifest.files[0].sha256 == WEIGHTS_SHA256
        manifests.append(manifest.to_dict())
        for invalid in (wire[:-1] + "x", wire + ":extra", wire.replace(":40:", ":040:")):
            assert decode_submission_commitment(invalid) is None
            assert decode_commitment("registered-hotkey", 123, invalid) is None
    assert manifests[0] != manifests[1]
