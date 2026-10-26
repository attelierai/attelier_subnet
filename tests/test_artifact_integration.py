import hashlib
import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from forge.models.artifact_manifest import (
    MAX_AUXILIARY_FILE_BYTES,
    MAX_FULL_TRANSFORMER_BYTES,
    MAX_LORA_BYTES,
    MAX_PIPELINE_TOKENIZER_JSON_BYTES,
    ArtifactError,
    ArtifactInspection,
    ArtifactReference,
    ManifestFile,
    ModelManifest,
)
from forge.models.artifacts import (
    build_manifest,
    fetch_verified,
    inspect_public_artifact,
    snapshot_manifest,
)

REPO = "public/video-model"
REVISION = "a" * 40
PAYLOAD = b"public safetensors bytes\n"
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@contextmanager
def artifact_server(
    payload=PAYLOAD,
    siblings=None,
    requests=None,
    repo=REPO,
    revision=REVISION,
    resolve_chunk_delay=0.0,
    malformed_resolve=False,
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if requests is not None:
                requests.append(self.path)
            if self.path.startswith(f"/api/models/{repo}/revision/{revision}"):
                body = json.dumps(
                    {
                        "private": False,
                        "gated": False,
                        "disabled": False,
                        "sha": revision,
                        "cardData": {"license": "apache-2.0"},
                        "siblings": siblings
                        if siblings is not None
                        else [
                            {
                                "rfilename": "transformer/model.safetensors",
                                "size": len(PAYLOAD),
                                "lfs": {"sha256": SHA256, "size": len(PAYLOAD)},
                            }
                        ],
                    }
                ).encode()
            elif self.path == f"/{repo}/resolve/{revision}/transformer/model.safetensors":
                body = payload
            else:
                self.send_error(404)
                return
            if self.path.startswith(f"/{repo}/resolve/") and malformed_resolve:
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"10\r\nshort")
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.path.startswith(f"/{repo}/resolve/") and resolve_chunk_delay:
                for byte in body:
                    self.wfile.write(bytes((byte,)))
                    self.wfile.flush()
                    time.sleep(resolve_chunk_delay)
            else:
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


def manifest(server):
    reference = ArtifactReference("hf", REPO, REVISION)
    return build_manifest(
        reference,
        role="pipeline",
        files=(ManifestFile("transformer/model.safetensors", SHA256, len(PAYLOAD)),),
        license="apache-2.0",
        generation_config={"steps": 4, "guidance_scale": 5.0},
        hf_base_url=server,
    )


def test_public_hf_manifest_and_download_are_verified(tmp_path):
    with artifact_server() as server:
        inspection = inspect_public_artifact(
            ArtifactReference("hf", REPO, REVISION), hf_base_url=server
        )
        assert inspection.license == "apache-2.0"
        assert inspection.files == ("transformer/model.safetensors",)
        candidate = manifest(server)
        result = fetch_verified(candidate, tmp_path / "candidate", hf_base_url=server)
    assert (result / "transformer/model.safetensors").read_bytes() == PAYLOAD
    assert fetch_verified(candidate, result, hf_base_url="http://offline.invalid") == result


def test_corrupt_public_download_is_rejected_without_leaving_a_file(tmp_path):
    with (
        artifact_server(payload=b"x" * len(PAYLOAD)) as server,
        pytest.raises(ArtifactError, match="hash or size mismatch"),
    ):
        fetch_verified(manifest(server), tmp_path / "candidate", hf_base_url=server)
    assert not (tmp_path / "candidate/transformer/model.safetensors").exists()


def test_hf_download_deadline_rejects_a_slow_stream_without_leaving_a_file(tmp_path):
    payload = b"slow-public-model"
    weights = ManifestFile(
        "transformer/model.safetensors", hashlib.sha256(payload).hexdigest(), len(payload)
    )
    candidate = ModelManifest(
        ArtifactReference("hf", REPO, REVISION),
        "pipeline",
        (weights,),
        "apache-2.0",
        {"steps": 4, "guidance_scale": 5.0},
    )
    requests = []
    with (
        artifact_server(payload=payload, requests=requests, resolve_chunk_delay=0.03) as server,
        pytest.raises(ArtifactError, match="deadline"),
    ):
        fetch_verified(
            candidate,
            tmp_path / "candidate",
            hf_base_url=server,
            timeout_seconds=1.0,
            download_deadline_seconds=0.09,
        )
    assert f"/{REPO}/resolve/{REVISION}/transformer/model.safetensors" in requests
    assert not (tmp_path / "candidate/transformer/model.safetensors").exists()


def test_hf_malformed_stream_is_a_candidate_transport_failure_and_leaves_no_file(tmp_path):
    requests = []
    with (
        artifact_server(requests=requests, malformed_resolve=True) as server,
        pytest.raises(ArtifactError, match="transport failed"),
    ):
        fetch_verified(manifest(server), tmp_path / "candidate", hf_base_url=server)
    assert f"/{REPO}/resolve/{REVISION}/transformer/model.safetensors" in requests
    assert not (tmp_path / "candidate/transformer/model.safetensors").exists()


def test_mismatched_existing_cache_file_is_never_overwritten(tmp_path):
    target = tmp_path / "candidate/transformer/model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong cached file")
    with artifact_server() as server, pytest.raises(ArtifactError, match="existing cache file"):
        fetch_verified(manifest(server), tmp_path / "candidate", hf_base_url=server)
    assert target.read_bytes() == b"wrong cached file"


def test_manifest_rejects_unsafe_path_and_honors_both_immutable_reference_forms():
    with pytest.raises(ArtifactError, match="unsafe"):
        ManifestFile("../escape.safetensors", SHA256, len(PAYLOAD))
    with pytest.raises(ArtifactError, match="unsupported"):
        ManifestFile("transformer/model.pkl", SHA256, len(PAYLOAD))
    assert ArtifactReference("hippius", REPO, f"sha256:{'b' * 64}").immutable_id.endswith("b" * 64)
    with pytest.raises(ArtifactError, match="revision"):
        ArtifactReference("hf", REPO, "main")


def test_manifest_cannot_self_declare_a_license_when_public_metadata_lacks_one():
    reference = ArtifactReference("hf", REPO, REVISION)
    with pytest.raises(ArtifactError, match="verifiable license"):
        build_manifest(
            reference,
            role="pipeline",
            files=(ManifestFile("transformer/model.safetensors", SHA256, len(PAYLOAD)),),
            license="apache-2.0",
            generation_config={"steps": 4, "guidance_scale": 5.0},
            inspection=ArtifactInspection(reference, None, ("transformer/model.safetensors",)),
        )


def test_transformer_candidates_require_a_pinned_base_and_pipeline_cannot_name_one():
    reference = ArtifactReference("hf", REPO, REVISION)
    file = ManifestFile("transformer/model.safetensors", SHA256, len(PAYLOAD))
    with pytest.raises(ArtifactError, match="requires an immutable base"):
        ModelManifest(
            reference,
            "full_transformer",
            (file,),
            "apache-2.0",
            {"steps": 4, "guidance_scale": 5.0},
        )
    base = ArtifactReference("hf", "public/base-model", "b" * 40)
    assert (
        ModelManifest(
            reference,
            "full_transformer",
            (file,),
            "apache-2.0",
            {"steps": 4, "guidance_scale": 5.0},
            base,
        ).base_reference
        == base
    )
    with pytest.raises(ArtifactError, match="pipeline manifest"):
        ModelManifest(
            reference, "pipeline", (file,), "apache-2.0", {"steps": 4, "guidance_scale": 5.0}, base
        )


def test_manifest_round_trip_and_fingerprint_include_generation_config():
    with artifact_server() as server:
        original = manifest(server)
    restored = type(original).from_dict(original.to_dict())
    changed = type(original)(
        original.reference,
        original.role,
        original.files,
        original.license,
        {"steps": 5, "guidance_scale": 5.0},
    )
    assert restored == original
    assert restored.to_dict() == original.to_dict()
    assert changed.fingerprint() != original.fingerprint()


def test_snapshot_manifest_uses_lfs_metadata_without_downloading_weights():
    with artifact_server() as server:
        snapshot = snapshot_manifest(
            ArtifactReference("hf", REPO, REVISION),
            role="full_transformer",
            generation_config={"steps": 4, "guidance_scale": 5.0},
            base_reference=ArtifactReference("hf", "public/base-model", "b" * 40),
            hf_base_url=server,
        )
    assert snapshot.files == (ManifestFile("transformer/model.safetensors", SHA256, len(PAYLOAD)),)


def test_hf_lfs_configuration_over_limit_is_rejected_before_download():
    config_size = MAX_AUXILIARY_FILE_BYTES + 1
    siblings = [
        {
            "rfilename": "transformer/model.safetensors",
            "size": len(PAYLOAD),
            "lfs": {"sha256": SHA256, "size": len(PAYLOAD)},
        },
        {
            "rfilename": "transformer/config.json",
            "size": config_size,
            "lfs": {"sha256": "b" * 64, "size": config_size},
        },
    ]
    reference = ArtifactReference("hf", REPO, REVISION)
    base = ArtifactReference("hf", "public/base-model", "b" * 40)
    requests = []
    with (
        artifact_server(siblings=siblings, requests=requests) as server,
        pytest.raises(ArtifactError, match="configuration file exceeds"),
    ):
        snapshot_manifest(
            reference,
            role="full_transformer",
            generation_config={"steps": 4, "guidance_scale": 5.0},
            base_reference=base,
            hf_base_url=server,
        )
    assert requests == [f"/api/models/{REPO}/revision/{REVISION}?blobs=true"]
    assert not any(path.startswith(f"/{REPO}/resolve/") for path in requests)


@pytest.mark.parametrize(
    ("role", "path", "limit"),
    (
        ("full_transformer", "transformer/model.safetensors", MAX_FULL_TRANSFORMER_BYTES),
        ("lora", "adapter.safetensors", MAX_LORA_BYTES),
    ),
)
def test_hf_candidate_over_wan_role_limit_is_rejected_before_download(role, path, limit):
    oversized = limit + 1
    siblings = [
        {
            "rfilename": path,
            "size": oversized,
            "lfs": {"sha256": "b" * 64, "size": oversized},
        }
    ]
    reference = ArtifactReference("hf", REPO, REVISION)
    base = ArtifactReference("hf", "public/base-model", "b" * 40)
    requests = []
    with (
        artifact_server(siblings=siblings, requests=requests) as server,
        pytest.raises(ArtifactError, match=rf"{role} manifest exceeds"),
    ):
        snapshot_manifest(
            reference,
            role=role,
            generation_config={"steps": 4, "guidance_scale": 5.0},
            base_reference=base,
            hf_base_url=server,
        )
    assert requests == [f"/api/models/{REPO}/revision/{REVISION}?blobs=true"]
    assert not any(path.startswith(f"/{REPO}/resolve/") for path in requests)


def test_pinned_pipeline_tokenizer_index_is_within_its_bounded_allowance():
    tokenizer_size = 16_837_459
    siblings = [
        {
            "rfilename": "transformer/model.safetensors",
            "size": len(PAYLOAD),
            "lfs": {"sha256": SHA256, "size": len(PAYLOAD)},
        },
        {
            "rfilename": "tokenizer/tokenizer.json",
            "size": tokenizer_size,
            "lfs": {"sha256": "2" * 64, "size": tokenizer_size},
        },
    ]
    reference = ArtifactReference(
        "hf", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"
    )
    with artifact_server(
        siblings=siblings, repo=reference.repo_id, revision=reference.revision
    ) as server:
        snapshot = snapshot_manifest(
            reference,
            role="pipeline",
            generation_config={"steps": 4, "guidance_scale": 5.0},
            hf_base_url=server,
        )
    assert tokenizer_size <= MAX_PIPELINE_TOKENIZER_JSON_BYTES
    assert snapshot.files[-1].path == "tokenizer/tokenizer.json"
    assert snapshot.files[-1].bytes == tokenizer_size
