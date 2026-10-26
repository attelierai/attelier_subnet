"""CPU boundary tests for the restricted H3 trusted graph adapter.

These tests inspect graph and custody contracts only.  They intentionally do
not claim that a local CPU can execute or qualify MiniMax H3.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forge.inference.trace_runtime import (
    AUDIO_VAE,
    ENCODER,
    RUNTIME_CODE,
    UNET,
    VIDEO_VAE,
    NativeH3Profile,
    RuntimeGraphError,
    TrustedT2VInput,
    build_full_forward_replay_graph,
    build_trusted_baseline_graph,
    build_validated_latent_render_graph,
    renderer_fingerprint,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _profile() -> NativeH3Profile:
    return NativeH3Profile.from_validated(
        {
            "id": "h3-native-blackwell-20-v1",
            "modelManifestHash": "b" * 64,
            "workflowHash": "c" * 64,
            "runtimeHash": "f" * 64,
            "runtime": {
                "comfyRevision": "comfy-b0f4b7",
                "unetName": UNET,
                "encoderName": ENCODER,
                "videoVaeName": VIDEO_VAE,
                "audioVaeName": AUDIO_VAE,
            },
        }
    )


def _request() -> TrustedT2VInput:
    return TrustedT2VInput.from_validated(
        {
            "jobId": "job-001",
            "nonce": "opaque-validator-nonce",
            "parametersHash": "d" * 64,
            "requestHash": "e" * 64,
            "effectiveSeed": 123456789,
            "parameters": {
                "modality": "text-to-video",
                "prompt": "a small kinetic sculpture",
                "duration": 5,
                "resolution": "480p",
                "aspectRatio": "16:9",
                "audio": True,
                "seed": 123456789,
                "profile": "base",
            },
        }
    )


def _candidate_manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    forwards = []
    for index in range(20):
        relative = Path("forward") / f"{index:03d}.pt"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"not-loaded-in-cpu-test-{index}".encode())
        forwards.append(
            {
                "call": index,
                "tensorPath": str(relative),
                "sha256": _sha(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
        )
    manifest = root / "capture.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "forge/inference-trace-capture/v1",
                "forward": forwards,
                "finalLatent": [],
            }
        )
    )
    return manifest


def test_trusted_graph_binds_derived_seed_and_records_all_twenty_forwards(tmp_path):
    graph = build_trusted_baseline_graph(
        _request(), _profile(), trace_root=tmp_path / "trace", filename_prefix="attelier/job-001"
    )
    assert graph["6"]["inputs"]["noise_seed"] == 123456789
    assert graph["9"]["inputs"] == {
        "model": ["1", 0],
        "scheduler": "simple",
        "steps": 20,
        "denoise": 1,
    }
    sampler = graph["10"]["inputs"]
    assert sampler["capture_all_forwards"] is True
    assert sampler["input_hash"] == "e" * 64
    assert sampler["model_manifest_hash"] == "b" * 64
    assert set(graph) == {"1", "2", "3", "5", "6", "7", "8", "9", "10", "17"}
    assert all(node not in graph for node in ("4", "11", "12", "13", "14", "15", "16"))
    assert graph["17"]["class_type"] == "H3TraceSeal"
    assert graph["17"]["inputs"]["samples"] == ["10", 0]


def test_full_replay_is_exactly_twenty_bounded_custodied_bundles(tmp_path):
    manifest = _candidate_manifest(tmp_path / "candidate")
    graph = build_full_forward_replay_graph(
        _profile(),
        candidate_manifest=manifest,
        result_directory=tmp_path / "replay-results",
        candidate_label="approved",
        max_bundle_bytes=10_000,
    )
    requests = json.loads(graph["2"]["inputs"]["requests_json"])
    assert len(requests) == 20
    assert requests[0]["bundle_path"].endswith("forward/000.pt")
    assert requests[-1]["result_path"].endswith("forward-019.json")
    assert graph["2"]["class_type"] == "H3TraceForwardReplayBatch"

    document = json.loads(manifest.read_text())
    document["forward"].pop()
    manifest.write_text(json.dumps(document))
    with pytest.raises(RuntimeGraphError, match="count_not_20"):
        build_full_forward_replay_graph(
            _profile(),
            candidate_manifest=manifest,
            result_directory=tmp_path / "other-results",
            candidate_label="approved",
        )


def test_renderer_accepts_final_latent_manifest_not_miner_media_path(tmp_path):
    manifest = _candidate_manifest(tmp_path / "candidate")
    graph = build_validated_latent_render_graph(
        _profile(), candidate_manifest=manifest, filename_prefix="attelier/verified/job-001"
    )
    assert graph["1"]["class_type"] == "H3TraceFinalLatentLoader"
    assert graph["1"]["inputs"]["manifest_path"] == str(manifest.resolve())
    assert graph["6"]["inputs"]["fps"] == 24
    assert "media_path" not in json.dumps(graph)


def test_real_capture_shape_binds_nested_decoded_torch_archive_to_signed_chunks(tmp_path):
    torch = pytest.importorskip("torch")
    root = tmp_path / "capture"
    root.mkdir()
    tensor_path = root / "tensors" / "decoded-audio.pt"
    tensor_path.parent.mkdir()
    torch.save(torch.tensor([1.0]), tensor_path)
    descriptor = {
        "tensorPath": "tensors/decoded-audio.pt",
        "sha256": _sha(tensor_path.read_bytes()),
        "bytes": tensor_path.stat().st_size,
        "shape": [1],
        "dtype": "float32",
    }
    capture = {
        "schema": "forge/inference-trace-capture/v1",
        "runId": "job-001",
        "inputHash": "e" * 64,
        "capture": {
            "modelManifestHash": "b" * 64,
            "workflowHash": "c" * 64,
            "runtime": "comfy-b0f4b7",
        },
        "decoded": {"audio": descriptor},
    }
    capture_path = root / "capture.json"
    capture_path.write_text(json.dumps(capture))
    spec = importlib.util.spec_from_file_location(
        "h3_trusted_runtime", Path(__file__).parents[2] / "scripts" / "h3_trusted_runtime.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bind_capture_to_chunks(capture, capture_path, [capture_path, tensor_path])
    with pytest.raises(RuntimeError, match="capture_referenced_chunk_unsigned"):
        module.bind_capture_to_chunks(capture, capture_path, [capture_path])


def test_only_comfy_cache_uuid_may_vary_across_otherwise_identical_forwards():
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "h3_trusted_runtime", root / "scripts" / "h3_trusted_runtime.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forward(cache_uuid, *, input_value=1, timestep=2, conditioning_value="same"):
        return {
            "input": {"tensor": input_value},
            "timestep": {"tensor": timestep},
            "prediction": {"tensor": 3},
            "conditioning": {
                "cross_attn": conditioning_value,
                "transformer_options": {
                    "sigmas": {"tensor": 2},
                    "uuids": [{"__forge_uuid_v1__": cache_uuid}],
                },
            },
        }

    candidate = forward("46b80963-5948-4261-b5fd-f84bb7b62e61")
    trusted = forward("07023810-4889-4064-bdc9-61f2d4f40b7f")
    assert module.same_forward(candidate, trusted)
    assert not module.same_forward(
        candidate, forward("07023810-4889-4064-bdc9-61f2d4f40b7f", input_value=9)
    )
    assert not module.same_forward(
        candidate, forward("07023810-4889-4064-bdc9-61f2d4f40b7f", timestep=9)
    )
    assert not module.same_forward(
        candidate,
        forward("07023810-4889-4064-bdc9-61f2d4f40b7f", conditioning_value="changed"),
    )
    assert not module.same_forward(candidate, forward("not-a-uuid"))


def test_unqualified_diagnostic_cannot_emit_a_verified_result():
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "h3_trusted_runtime", root / "scripts" / "h3_trusted_runtime.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.diagnostic_only(
        {
            "schema": "forge-h3-verified-result/v1",
            "verdict": "verified",
            "reason": "capture_profile_binding",
            "result": {"path": "/private/output.mp4", "sha256": "a" * 64, "bytes": 1},
            "finalLatent": {"video": "b" * 64, "audio": "c" * 64},
        },
        True,
    )
    assert report["verdict"] == "inconclusive"
    assert report["diagnosticVerdict"] == "verified"
    assert report["diagnosticReason"] == "capture_profile_binding"
    assert "result" not in report and "finalLatent" not in report


@pytest.fixture
def runtime_environment(tmp_path):
    from forge.inference.trace_protocol import FIXED_PARAMETERS, file_sha256, sha256

    root = Path(__file__).parents[2]
    code = {name: file_sha256(root / name) for name in RUNTIME_CODE}
    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    hashes = {key: "b" * 64 for key in ("unetName", "encoderName", "videoVaeName", "audioVaeName")}
    measurement = {"code": code, "ffmpeg": ffmpeg}
    profile = {
        "id": "h3-native-blackwell-20-v1",
        "modelManifestHash": "b" * 64,
        "workflowHash": code["src/forge/inference/trace_runtime.py"],
        "runtimeHash": sha256(measurement),
        "rendererHash": renderer_fingerprint(code, hashes, ffmpeg),
        "steps": 20,
        "parameters": FIXED_PARAMETERS,
        "numerics": {"maxRelativeRms": "0.000001"},
        "limits": {"maxChunks": 256, "maxChunkBytes": 1048576, "maxBundleBytes": 4194304},
        "runtime": {
            "comfyRevision": "fixture-runtime",
            "unetName": UNET,
            "encoderName": ENCODER,
            "videoVaeName": VIDEO_VAE,
            "audioVaeName": AUDIO_VAE,
            "checkpointSha256": hashes,
            "numerics": {"mode": "bit_exact"},
            "measurement": measurement,
        },
    }
    attestation = {
        "schema": "forge-h3-runtime-attestation/v2",
        "runtimeHash": profile["runtimeHash"],
        "rendererHash": profile["rendererHash"],
        "modelManifestHash": profile["modelManifestHash"],
        "comfyRevision": "fixture-runtime",
        "checkpoints": {key: profile["runtime"][key] for key in hashes},
        "checkpointSha256": hashes,
    }
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    return profile, attestation, profiles


def test_verifier_process_rejects_stale_renderer_and_code_before_gpu_submission(
    tmp_path, runtime_environment
):
    from forge.inference.trace_protocol import FIXED_PARAMETERS, effective_seed, sha256

    root = Path(__file__).parents[2]
    profile, attestation, profiles = runtime_environment
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}")
    public_key_path = tmp_path / "worker-public.txt"
    public_key_path.write_text("unused-before-receipt-validation")
    for variant, expected in (
        ("control", "receipt_fields"),
        ("old-attestation", "runtime_attestation_schema"),
        ("renderer", "runtime_renderer_changed"),
        ("code", "runtime_code_changed"),
    ):
        current = json.loads(json.dumps(profile))
        startup = dict(attestation)
        if variant == "old-attestation":
            startup["schema"] = "forge-h3-runtime-attestation/v1"
            startup.pop("rendererHash")
        elif variant == "renderer":
            current["rendererHash"] = startup["rendererHash"] = "c" * 64
        elif variant == "code":
            current["runtime"]["measurement"]["code"]["scripts/h3_trusted_runtime.py"] = "c" * 64
            current["runtimeHash"] = sha256(current["runtime"]["measurement"])
            startup["runtimeHash"] = current["runtimeHash"]
        identity = {"id": current["id"], "hash": sha256(current)}
        request = {
            "schema": "forge-h3-request/v1",
            "jobId": "job-preflight",
            "workerId": "fixture-worker",
            "nonce": "a" * 64,
            "requestedSeed": 7,
            "profile": identity,
            "parameters": {
                **FIXED_PARAMETERS,
                "prompt": "fixture",
                "seed": effective_seed("job-preflight", "a" * 64, 7, identity["hash"]),
            },
        }
        (profiles / (identity["id"] + ".json")).write_text(
            json.dumps(
                {
                    "schema": "forge/h3-profile/v1",
                    "profile": current,
                    "qualification": {"status": "unqualified"},
                }
            )
        )
        request_path, attestation_path = tmp_path / "request.json", tmp_path / "attestation.json"
        request_path.write_text(json.dumps(request))
        attestation_path.write_text(json.dumps(startup))
        output = tmp_path / variant
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/h3_trusted_runtime.py"),
                "--request",
                str(request_path),
                "--receipt",
                str(receipt_path),
                "--profile-root",
                str(profiles),
                "--evidence-root",
                str(tmp_path / "evidence"),
                "--worker-public-key-file",
                str(public_key_path),
                "--output-dir",
                str(output),
                "--runtime-attestation",
                str(attestation_path),
                "--comfy-origin",
                "http://127.0.0.1:1",
                "--hotkey",
                "preflight-diagnostic-only",
                "--comfy-output-root",
                str(tmp_path),
                "--allow-unqualified-diagnostic",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = json.loads(result.stdout)
        assert report["verdict"] == "inconclusive"
        assert report["diagnosticReason"] == expected
        assert "result" not in report
        assert not (output / "baseline-graph.json").exists()
        assert json.loads((output / "report.json").read_text()) == report
    # Reusing a directory must leave the prior evidence untouched.
    original = (output / "report.json").read_bytes()
    repeated = subprocess.run(result.args, check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(repeated.stdout)["verdict"] == "inconclusive"
    assert (output / "report.json").read_bytes() == original
    invalid_export = [arg for arg in result.args if arg != "--allow-unqualified-diagnostic"]
    invalid_export.append("--export-diagnostic-artifact")
    denied = subprocess.run(invalid_export, capture_output=True, text=True, timeout=30)
    assert denied.returncode == 2 and "requires --allow-unqualified-diagnostic" in denied.stderr
    assert (output / "report.json").read_bytes() == original


def test_signed_full20_runtime_rejects_attacks_and_returns_only_trusted_media(
    tmp_path, runtime_environment
):
    """Real signatures, tensors, subprocess, HTTP and codecs; only Comfy is a fixture.

    Tiny CPU tensors cannot qualify H3. They exercise the actual verifier's
    decision and custody boundaries without mocking its internal comparisons.
    """
    import base64
    import copy
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    torch = pytest.importorskip("torch")
    from forge.inference.trace_protocol import canonical, effective_seed, file_sha256, sha256

    root = Path(__file__).parents[2]
    profile, attestation, profiles = runtime_environment
    identity = {"id": profile["id"], "hash": sha256(profile)}
    (profiles / (identity["id"] + ".json")).write_bytes(
        canonical(
            {
                "schema": "forge/h3-profile/v1",
                "profile": profile,
                "qualification": {
                    "status": "qualified",
                    "evidenceSha256": "e" * 64,
                    "expiresAt": 9_999_999_999,
                    "lanes": [
                        {
                            "hotkey": "test-hotkey",
                            "workerId": "test-worker",
                            "profileId": profile["id"],
                            "profileHash": identity["hash"],
                            "rendererHash": profile["rendererHash"],
                        }
                    ],
                },
            }
        )
    )
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_bytes(canonical(attestation))
    request = {
        "schema": "forge-h3-request/v1",
        "jobId": "e24fffb2-67ad-4c10-aeb9-07e356fdced6",
        "workerId": "test-worker",
        "nonce": "a" * 64,
        "requestedSeed": 7,
        "profile": identity,
        "parameters": {
            **profile["parameters"],
            "prompt": "CPU boundary fixture",
            "seed": effective_seed(
                "e24fffb2-67ad-4c10-aeb9-07e356fdced6", "a" * 64, 7, identity["hash"]
            ),
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical(request))
    key, pub = tmp_path / "signer.pem", tmp_path / "worker.pub"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)],
        capture_output=True,
        check=True,
    )
    public = subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
        capture_output=True,
        check=True,
    ).stdout[-32:]
    pub.write_text(base64.b64encode(public).decode())

    def capture(directory, *, attack=None, trusted=False):
        directory.mkdir(parents=True)
        (directory / "tensors").mkdir()
        (directory / "forward").mkdir()

        def parts(label):
            rows = []
            for component in ("video", "audio"):
                tensor_path = directory / "tensors" / f"{label}-{component}.pt"
                value = 2.0 if attack == "latent" and label == "final" else 1.0
                torch.save(torch.tensor([value], dtype=torch.float32), tensor_path)
                rows.append(
                    {
                        "component": component,
                        "tensorPath": str(tensor_path.relative_to(directory)),
                        "sha256": file_sha256(tensor_path),
                        "bytes": tensor_path.stat().st_size,
                        "shape": [1],
                        "dtype": "float32",
                    }
                )
            return rows

        state = parts("state")
        forwards = []
        for index in range(20):
            path = directory / "forward" / f"{index:03d}.pt"
            torch.save(
                {
                    "input": torch.ones(1),
                    "timestep": torch.tensor([20 - index]),
                    "prediction": torch.tensor(
                        [2.0 if attack == "skipped-call" and index == 7 else 1.0]
                    ),
                    "conditioning": {
                        "cross_attn": torch.ones(1),
                        "transformer_options": {
                            "uuids": [
                                {
                                    "__forge_uuid_v1__": "46b80963-5948-4261-b5fd-f84bb7b62e61"
                                    if trusted
                                    else "07023810-4889-4064-bdc9-61f2d4f40b7f"
                                }
                            ],
                        },
                    },
                },
                path,
            )
            forwards.append(
                {
                    "call": index,
                    "tensorPath": str(path.relative_to(directory)),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        document = {
            "schema": "forge/inference-trace-capture/v1",
            "runId": request["jobId"],
            "inputHash": sha256(request),
            "declaredSteps": 20,
            "callbackSteps": 20,
            "capture": {
                "modelManifestHash": profile["modelManifestHash"],
                "workflowHash": profile["workflowHash"],
                "runtime": "fixture-runtime",
            },
            "inputNoise": parts("noise"),
            "inputLatent": parts("input"),
            "finalLatent": parts("final"),
            "trace": [
                {"step": i, "sigma": float(20 - i), "stateCallback": state, "denoised": state}
                for i in range(20)
            ],
            "forward": forwards,
        }
        if attack == "steps":
            document["callbackSteps"] = 10
        path = directory / ("manifest.json" if trusted else "capture.json")
        path.write_text(json.dumps(document))
        return path

    def signed_receipt(evidence, capture_path):
        paths = [
            capture_path,
            *sorted((evidence / "tensors").glob("*.pt")),
            *sorted((evidence / "forward").glob("*.pt")),
        ]
        chunks = [
            {
                "index": i,
                "name": str(path.relative_to(evidence)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for i, path in enumerate(paths)
        ]
        final = json.loads(capture_path.read_text())["finalLatent"]
        receipt = {
            "schema": "forge-h3-trace/v1",
            "jobId": request["jobId"],
            "workerId": request["workerId"],
            "nonce": request["nonce"],
            "requestHash": sha256(request),
            "parametersHash": sha256(request["parameters"]),
            "profile": identity,
            "modelManifestHash": profile["modelManifestHash"],
            "workflowHash": profile["workflowHash"],
            "steps": 20,
            "trace": {"algorithm": "sha256-manifest-v1", "chunks": chunks, "root": sha256(chunks)},
            "finalLatent": [
                {
                    "component": d["component"],
                    "chunkIndex": next(c["index"] for c in chunks if c["name"] == d["tensorPath"]),
                }
                for d in final
            ],
        }
        message, signature = evidence / "message.json", evidence / "receipt.sig"
        message.write_bytes(canonical(receipt))
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(key),
                "-rawin",
                "-in",
                str(message),
                "-out",
                str(signature),
            ],
            capture_output=True,
            check=True,
        )
        receipt["signature"] = base64.b64encode(signature.read_bytes()).decode()
        path = evidence / "receipt.json"
        path.write_bytes(canonical(receipt))
        return path

    source = tmp_path / "trusted-source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=832x480:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000",
            "-t",
            "5.166667",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    miner_media = tmp_path / "miner.mp4"
    miner_media.write_bytes(b"substituted miner media must never be read")
    histories, graphs = {}, []

    class Comfy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            graph = json.loads(self.rfile.read(int(self.headers["content-length"])))["prompt"]
            graphs.append(graph)
            outputs = {}
            if "10" in graph:
                inputs = graph["10"]["inputs"]
                capture(Path(inputs["trace_root"]) / inputs["run_id"], trusted=True)
            else:
                assert graph["1"]["class_type"] == "H3TraceFinalLatentLoader"
                assert str(miner_media) not in json.dumps(graph)
                outputs = {"7": {"images": [{
                    "filename": "../miner.mp4" if attack == "renderer-path" else source.name,
                    "subfolder": "", "type": "output",
                }], "animated": [True]}}
            prompt_id = str(len(histories))
            histories[prompt_id] = {
                "status": {"status_str": "error" if attack == "platform" else "success"},
                "outputs": outputs,
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"prompt_id": prompt_id}).encode())

        def do_GET(self):
            prompt_id = self.path.rsplit("/", 1)[-1]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({prompt_id: histories[prompt_id]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Comfy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for attack, expected, graph_count in (
            ("honest", "verified", 2),
            ("skipped-call", "invalid", 1),
            ("latent", "invalid", 1),
            ("steps", "invalid", 0),
            ("stale", "invalid", 0),
            ("diagnostic", "inconclusive", 2),
            ("diagnostic-export", "inconclusive", 2),
            ("renderer-path", "inconclusive", 2),
            ("platform", "inconclusive", 1),
            ("unqualified-lane", "inconclusive", 0),
        ):
            evidence = tmp_path / attack / "evidence"
            capture_path = capture(evidence, attack=attack)
            receipt_path = signed_receipt(evidence, capture_path)
            current_request = copy.deepcopy(request)
            if attack == "stale":
                current_request["nonce"] = "b" * 64
                current_request["parameters"]["seed"] = effective_seed(
                    request["jobId"], "b" * 64, 7, identity["hash"]
                )
            request_path.write_bytes(canonical(current_request))
            output = tmp_path / attack / "output"
            before = len(graphs)
            run = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/h3_trusted_runtime.py"),
                    "--request",
                    str(request_path),
                    "--receipt",
                    str(receipt_path),
                    "--profile-root",
                    str(profiles),
                    "--evidence-root",
                    str(evidence),
                    "--worker-public-key-file",
                    str(pub),
                    "--output-dir",
                    str(output),
                    "--runtime-attestation",
                    str(attestation_path),
                    "--comfy-origin",
                    f"http://127.0.0.1:{server.server_port}",
                    "--hotkey",
                    "different-hotkey" if attack == "unqualified-lane" else "test-hotkey",
                    "--comfy-output-root",
                    str(tmp_path),
                    *(
                        ["--allow-unqualified-diagnostic"]
                        if attack.startswith("diagnostic")
                        else []
                    ),
                    *(["--export-diagnostic-artifact"] if attack == "diagnostic-export" else []),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            report = json.loads(run.stdout)
            assert report["verdict"] == expected, report
            assert len(graphs) - before == graph_count
            if expected == "verified":
                media = Path(report["result"]["path"])
                assert media == output / "result.mp4"
                assert report["result"]["sha256"] == file_sha256(media)
                assert media.read_bytes() != miner_media.read_bytes()
                # Optional cross-repository release boundary: execute the real Node
                # adapter and Python CLI against this same external Comfy fixture.
                api_module = os.environ.get("H3_API_RUNTIME_MODULE")
                api_probe = os.environ.get("H3_API_CONTRACT_PROBE")
                if api_module and api_probe:
                    api_root = tmp_path / "api-client"
                    probe = subprocess.run(
                        ["node", api_probe, api_module, sys.executable, str(root),
                         str(api_root), str(profiles), str(attestation_path), str(tmp_path),
                         f"http://127.0.0.1:{server.server_port}", str(request_path),
                         str(receipt_path), str(pub), str(evidence)],
                        check=True, capture_output=True, text=True, timeout=60,
                    )
                    api_report = json.loads(probe.stdout)
                    assert api_report["lane"] == report["lane"]
                    assert api_report["result"]["sha256"] == file_sha256(api_root / "client.mp4")
                    assert api_report["checks"]["predictions"]["matched"] == 20
                assert report["checks"]["predictions"] == {
                    "checked": 20,
                    "matched": 20,
                    "total": 20,
                }
            else:
                assert "result" not in report
            if attack.startswith("diagnostic"):
                assert report["diagnosticVerdict"] == "verified"
                assert "finalLatent" not in report
            if attack == "diagnostic-export":
                artifact_path = output / "diagnostic-artifact.json"
                artifact = json.loads(artifact_path.read_text())
                assert artifact["purpose"] == "qualification_only"
                assert artifact["reportHash"] == sha256(report)
                assert artifact["media"]["sha256"] == file_sha256(output / "result.mp4")
                spec = importlib.util.spec_from_file_location(
                    "h3_remote_runtime", root / "scripts/h3_remote_runtime.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert module.validate_diagnostic_artifact(artifact, report) == artifact["media"]
                with pytest.raises(RuntimeError, match="diagnostic_artifact_binding"):
                    module.validate_diagnostic_artifact(artifact, {**report, "jobId": "stale"})
                with pytest.raises(RuntimeError, match="diagnostic_artifact_media"):
                    module.validate_diagnostic_artifact(
                        {**artifact, "media": {**artifact["media"], "name": "../miner.mp4"}}, report
                    )
            else:
                assert not (output / "diagnostic-artifact.json").exists()
            if attack == "skipped-call":
                assert report["reason"] == "trusted_conditioning_or_forward_input_mismatch"
            if attack == "renderer-path":
                assert report["reason"] == "renderer_output_invalid"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    # Exercise the experiment's actual capture subprocess: derived attacks must
    # preserve their honest source, retain signatures, and keep stale requests stale.
    import shutil

    from forge.inference.trace_protocol import TraceProtocolError, verify_bundle

    campaign = tmp_path / "miner-campaign"
    original = campaign / "cases/holdout/honest-full20/evidence"
    shutil.copytree(tmp_path / "honest/evidence", original)
    shutil.copyfile(original / "message.json", original / "receipt-message.json")
    shutil.copyfile(key, campaign / "test-ed25519.pem")
    shutil.copyfile(pub, campaign / "test-ed25519.pub")
    alias = tmp_path / "mounted-campaign"
    alias.symlink_to(campaign, target_is_directory=True)
    original_hashes = {p: file_sha256(p) for p in original.rglob("*") if p.is_file()}
    for condition in ("altered-final-latent", "substituted-miner-media", "stale-request"):
        current_request = copy.deepcopy(request)
        if condition == "stale-request":
            current_request["nonce"] = "b" * 64
            current_request["parameters"]["seed"] = effective_seed(
                request["jobId"], "b" * 64, 7, identity["hash"]
            )
        payload = {
            "case": {"lineage": "holdout", "condition": condition,
                     "derivedFrom": "holdout/honest-full20"},
            "request": current_request,
        }
        subprocess.run(
            [sys.executable, str(root / "scripts/h3_two_lane_qualification.py"), "capture",
             "--root", str(alias), "--profile-root", str(profiles)],
            input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=30,
        )
        evidence = campaign / "cases/holdout" / condition / "evidence"
        signed = json.loads((evidence / "receipt.json").read_text())
        if condition == "stale-request":
            assert signed == json.loads((original / "receipt.json").read_text())
            with pytest.raises(TraceProtocolError):
                verify_bundle(current_request, signed, evidence, profile, pub.read_text().strip())
        else:
            verify_bundle(current_request, signed, evidence, profile, pub.read_text().strip())
        if condition == "altered-final-latent":
            descriptor = json.loads((evidence / "capture.json").read_text())["finalLatent"][0]
            assert torch.load(evidence / descriptor["tensorPath"], weights_only=True)[0] == 2
        assert {p: file_sha256(p) for p in original_hashes} == original_hashes

    # Recover a completed but unsigned smoke through a symlinked mount, without
    # calling the unavailable Comfy endpoint or modifying the source capture.
    unsigned = campaign / "unsigned-smoke"
    shutil.copytree(original, unsigned)
    for name in ("receipt.json", "receipt-message.json", "receipt.sig"):
        (unsigned / name).unlink()
    recovered = tmp_path / "recovered-campaign"
    payload = {
        "case": {"lineage": "smoke", "condition": "smoke-full20", "derivedFrom": None,
                 "existingCapture": str(unsigned),
                 "existingCaptureSha256": file_sha256(unsigned / "capture.json")},
        "request": request,
    }
    subprocess.run(
        [sys.executable, str(root / "scripts/h3_two_lane_qualification.py"), "capture",
         "--root", str(recovered), "--profile-root", str(profiles)],
        input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=30,
    )
    case = recovered / "cases/smoke/smoke-full20"
    assert json.loads((case / "capture-summary.json").read_text())["reusedUnscoredSmoke"]
    verify_bundle(
        request, json.loads((case / "evidence/receipt.json").read_text()), case / "evidence",
        profile, (case / "worker-public-key.txt").read_text().strip(),
    )
    assert not (unsigned / "receipt.json").exists()
