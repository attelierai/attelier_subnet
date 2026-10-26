#!/usr/bin/env python3
"""Private GPU H3 full-trace verifier and candidate-latent renderer.

The protocol module establishes receipt custody.  This command performs the
expensive local checks: independent 20-step reference, exact claimed forward
input binding, all-20 model forwards, sampler recurrence and controlled VAE
render. It never reads a miner MP4.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from forge.inference.trace_protocol import (
    TraceProtocolError,
    effective_seed,
    file_sha256,
    load_profile_envelope,
    qualified_lane,
    read_json,
    sha256,
    validate_request,
    verify_bundle,
)
from forge.inference.trace_runtime import (
    CAPTURE_SCHEMA,
    RUNTIME_CODE,
    NativeH3Profile,
    RuntimeGraphError,
    TrustedT2VInput,
    build_trusted_baseline_graph,
    build_validated_latent_render_graph,
    renderer_fingerprint,
    write_graph_new,
)
from forge.inference.transition_diag import residual

RESULT_SCHEMA = "forge-h3-verified-result/v1"


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError("output_exists")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def post_json(origin: str, path: str, value: dict[str, Any] | None) -> Any:
    body = None if value is None else json.dumps(value, separators=(",", ":")).encode()
    req = urllib.request.Request(
        origin + path,
        data=body,
        headers={} if body is None else {"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def run_graph(origin: str, graph: dict[str, Any], client_id: str, timeout: int) -> dict[str, Any]:
    result = post_json(origin, "/prompt", {"prompt": graph, "client_id": client_id})
    prompt_id = result.get("prompt_id") if isinstance(result, dict) else None
    if not isinstance(prompt_id, str):
        raise RuntimeError("comfy_submission_invalid")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = post_json(origin, "/history/" + urllib.parse.quote(prompt_id), None)
        run = history.get(prompt_id) if isinstance(history, dict) else None
        if run:
            if run.get("status", {}).get("status_str") != "success":
                raise RuntimeError("comfy_graph_failed")
            return run
        time.sleep(2)
    raise RuntimeError("comfy_timeout")


def torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch_unavailable") from exc
    return torch


def descriptor_path(root: Path, item: dict[str, Any], maximum: int) -> Path:
    required = {"tensorPath", "sha256", "shape", "dtype", "bytes"}
    if not isinstance(item, dict) or not required <= set(item):
        raise RuntimeError("tensor_descriptor_invalid")
    relative = Path(str(item["tensorPath"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("tensor_path_unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("tensor_path_unsafe") from exc
    if not path.is_file() or path.stat().st_size != item["bytes"] or path.stat().st_size > maximum:
        raise RuntimeError("tensor_custody_invalid")
    if file_sha256(path) != item["sha256"]:
        raise RuntimeError("tensor_custody_invalid")
    return path


def tensor(root: Path, item: dict[str, Any], maximum: int):
    torch = torch_module()
    value = torch.load(descriptor_path(root, item, maximum), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
        raise RuntimeError("tensor_invalid")
    if (
        list(value.shape) != item["shape"]
        or str(value.dtype).removeprefix("torch.") != item["dtype"]
    ):
        raise RuntimeError("tensor_metadata_mismatch")
    return value


def same(left: Any, right: Any) -> bool:
    try:
        torch = torch_module()
    except RuntimeError:
        torch = None
    if torch is not None and (isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor)):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _cache_uuid_list(value: Any) -> bool:
    """Accept only Comfy's serialized per-conditioning cache identity list.

    Comfy b0f4b7's ``convert_cond`` assigns ``uuid.uuid4()`` to each
    conditioning branch and sampler code copies those values to this exact
    transformer option. They key branch-local caches but are regenerated for
    independent runs, so their values cannot be a cross-run equality signal.
    """
    # The restricted native T2V profile has one conditioning branch. A second
    # branch would be a different profile rather than an equivalence class.
    if not isinstance(value, list) or len(value) != 1:
        return False
    for item in value:
        if not isinstance(item, dict) or set(item) != {"__forge_uuid_v1__"}:
            return False
        encoded = item["__forge_uuid_v1__"]
        if not isinstance(encoded, str):
            return False
        try:
            if str(uuid.UUID(encoded)) != encoded:
                return False
        except ValueError:
            return False
    return True


def same_forward(left: Any, right: Any) -> bool:
    """Compare a fixed-input forward bundle with one narrow cache-key exception."""
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        return False
    for key in ("input", "timestep", "prediction"):
        if key not in left or not same(left[key], right[key]):
            return False
    candidate, trusted = left.get("conditioning"), right.get("conditioning")
    if (
        not isinstance(candidate, dict)
        or not isinstance(trusted, dict)
        or set(candidate) != set(trusted)
    ):
        return False
    for key in candidate:
        if key != "transformer_options":
            if not same(candidate[key], trusted[key]):
                return False
            continue
        c_options, t_options = candidate[key], trusted[key]
        if (
            not isinstance(c_options, dict)
            or not isinstance(t_options, dict)
            or set(c_options) != set(t_options)
            or "uuids" not in c_options
            or not _cache_uuid_list(c_options["uuids"])
            or not _cache_uuid_list(t_options["uuids"])
            or len(c_options["uuids"]) != len(t_options["uuids"])
        ):
            return False
        for option in c_options:
            if option != "uuids" and not same(c_options[option], t_options[option]):
                return False
    return True


def parts(root: Path, values: Any, maximum: int) -> dict[str, Any]:
    if not isinstance(values, list) or len(values) != 2:
        raise RuntimeError("latent_components_invalid")
    result = {
        item.get("component"): tensor(root, item, maximum)
        for item in values
        if isinstance(item, dict)
    }
    if set(result) != {"video", "audio"}:
        raise RuntimeError("latent_components_invalid")
    return result


def forwards(capture: dict[str, Any], root: Path, maximum: int) -> list[dict[str, Any]]:
    rows = capture.get("forward")
    if not isinstance(rows, list) or len(rows) != 20:
        raise RuntimeError("forward_count_not_20")
    output = []
    for index, item in enumerate(rows):
        if (
            not isinstance(item, dict)
            or set(item) != {"call", "tensorPath", "sha256", "bytes"}
            or item["call"] != index
        ):
            raise RuntimeError("forward_descriptor_invalid")
        path = descriptor_path(root, {**item, "shape": [], "dtype": "unused"}, maximum)
        payload = torch_module().load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {
            "input",
            "timestep",
            "conditioning",
            "prediction",
        }:
            raise RuntimeError("forward_payload_invalid")
        output.append(payload)
    return output


def compare_capture(
    candidate: dict[str, Any],
    candidate_path: Path,
    trusted: dict[str, Any],
    trusted_path: Path,
    request: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    maximum = profile["limits"]["maxChunkBytes"]
    # A malformed trusted baseline is a platform fault. Signed candidate
    # metadata has already passed the separate pre-GPU admission below.
    validate_capture_metadata(trusted, request, profile)
    for key in ("inputNoise", "inputLatent", "finalLatent"):
        left, right = (
            parts(candidate_path.parent, candidate[key], maximum),
            parts(trusted_path.parent, trusted[key], maximum),
        )
        if any(not same(left[k], right[k]) for k in left):
            raise TraceProtocolError("trusted_terminal_or_input_mismatch")
    for index, (left, right) in enumerate(zip(candidate["trace"], trusted["trace"], strict=True)):
        if (
            left.get("step") != index
            or right.get("step") != index
            or left.get("sigma") != right.get("sigma")
        ):
            raise TraceProtocolError("trace_schedule_mismatch")
        for key in ("stateCallback", "denoised"):
            a, b = (
                parts(candidate_path.parent, left.get(key), maximum),
                parts(trusted_path.parent, right.get(key), maximum),
            )
            if any(not same(a[k], b[k]) for k in a):
                raise TraceProtocolError("trusted_trace_mismatch")
    if any(
        not same_forward(a, b)
        for a, b in zip(
            forwards(candidate, candidate_path.parent, maximum),
            forwards(trusted, trusted_path.parent, maximum),
            strict=True,
        )
    ):
        raise TraceProtocolError("trusted_conditioning_or_forward_input_mismatch")


def validate_capture_metadata(
    capture: dict[str, Any], request: dict[str, Any], profile: dict[str, Any]
) -> None:
    if (
        capture.get("schema") != CAPTURE_SCHEMA
        or capture.get("runId") != request["jobId"]
        or capture.get("inputHash") != sha256(request)
    ):
        raise RuntimeError("capture_request_binding")
    if any(
        type(capture.get(key)) is not int or capture[key] != 20
        for key in ("declaredSteps", "callbackSteps")
    ):
        raise RuntimeError("capture_steps_not_20")
    metadata = capture.get("capture")
    if (
        not isinstance(metadata, dict)
        or metadata.get("modelManifestHash") != profile["modelManifestHash"]
        or metadata.get("workflowHash") != profile["workflowHash"]
        or metadata.get("runtime") != profile["runtime"]["comfyRevision"]
    ):
        raise RuntimeError("capture_profile_binding")
    if not isinstance(capture.get("trace"), list) or len(capture["trace"]) != 20:
        raise RuntimeError("trace_count_not_20")
    if not isinstance(capture.get("forward"), list) or len(capture["forward"]) != 20:
        raise RuntimeError("forward_count_not_20")


def bind_receipt_final_latent(
    capture: dict[str, Any], capture_path: Path, bound: dict[str, Path]
) -> None:
    rows = capture.get("finalLatent")
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError("candidate_final_latent_missing")
    root = capture_path.parent.resolve()
    for row in rows:
        if not isinstance(row, dict) or row.get("component") not in bound:
            raise RuntimeError("candidate_final_latent_invalid")
        path = (root / str(row.get("tensorPath", ""))).resolve()
        if path != bound[row["component"]].resolve():
            raise RuntimeError("candidate_final_latent_receipt_mismatch")


def bind_capture_to_chunks(capture: dict[str, Any], capture_path: Path, chunks: list[Path]) -> None:
    """Every tensor/forward named by capture must be in the signed chunk list."""
    root, allowed = capture_path.parent.resolve(), {path.resolve() for path in chunks}
    descriptors: list[dict[str, Any]] = []

    def collect(value: Any, depth: int = 0) -> None:
        if depth > 24:
            raise RuntimeError("capture_too_deep")
        if isinstance(value, dict):
            if "tensorPath" in value:
                descriptors.append(value)
            for child in value.values():
                collect(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 4096:
                raise RuntimeError("capture_array_size")
            for child in value:
                collect(child, depth + 1)

    collect(capture)
    if not descriptors:
        raise RuntimeError("capture_referenced_chunks_missing")
    for item in descriptors:
        name = item.get("tensorPath")
        if not isinstance(name, str):
            raise RuntimeError("capture_referenced_chunk_invalid")
        path = (root / name).resolve()
        if path not in allowed:
            raise RuntimeError("capture_referenced_chunk_unsigned")


def rendered_media(history: dict[str, Any], output_root: Path) -> Path:
    """Resolve only Comfy's relative SaveVideo reference under owned output."""
    node = history.get("outputs", {}).get("7", {})
    # Pinned Comfy SaveVideo exposes its video reference through UI.images.
    records = node.get("images", [])
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise RuntimeError("renderer_output_missing")
    record = records[0]
    name, subfolder = record.get("filename"), record.get("subfolder", "")
    if (
        not isinstance(name, str) or not isinstance(subfolder, str)
        or record.get("type") != "output"
        or Path(name).name != name or Path(subfolder).is_absolute()
        or ".." in Path(subfolder).parts
    ):
        raise RuntimeError("renderer_output_invalid")
    candidate = (output_root.resolve() / subfolder / name).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError("renderer_output_escapes_root") from exc
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise RuntimeError("renderer_output_unavailable")
    return candidate


def trim_and_validate_media(source: Path, destination: Path) -> Path:
    """Create the fixed five-second delivery only under the verifier output.

    H3's 17-frame alignment yields 124 source frames (5.167 s).  The public
    profile promises five seconds, so the validator owns this explicit 120-frame
    re-encode rather than handing the Comfy output path to the gateway.
    """
    if destination.exists() or destination.parent != destination.parent.resolve():
        raise RuntimeError("result_destination_unsafe")
    temporary = destination.with_suffix(".partial.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-t",
        "5",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
        "-c:a",
        "aac",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, capture_output=True, timeout=300, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("trusted_media_trim_failed") from exc
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(temporary),
            ],
            capture_output=True,
            timeout=30,
            check=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("trusted_media_probe_failed") from exc
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    if (
        not isinstance(video, dict)
        or not isinstance(audio, dict)
        or video.get("width") != 832
        or video.get("height") != 480
        or video.get("r_frame_rate") != "24/1"
        or int(video.get("nb_read_frames", "0")) != 120
        or audio.get("sample_rate") != "32000"
        or audio.get("channels") != 2
    ):
        raise RuntimeError("trusted_media_profile_mismatch")
    try:
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(temporary), "-map", "0:a:0", "-f", "f32le", "-"],
            capture_output=True,
            timeout=60,
            check=True,
        ).stdout
        import numpy as np

        waveform = np.frombuffer(decoded, dtype="<f4")
        # AAC priming/end padding is container metadata, not source audio.
        # Permit one codec frame of tail while retaining the fixed 5 s target.
        if abs(waveform.size - 320000) > 4096 or not np.isfinite(waveform).all():
            raise RuntimeError("trusted_media_audio_invalid")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_runtime_attestation(path: Path, profile: dict[str, Any]) -> None:
    """Bind the runtime to a startup measurement, not worker-declared hashes."""
    attestation = read_json(path)
    runtime = profile["runtime"]
    required = {
        "schema",
        "runtimeHash",
        "rendererHash",
        "modelManifestHash",
        "comfyRevision",
        "checkpoints",
        "checkpointSha256",
    }
    if set(attestation) != required or attestation["schema"] != "forge-h3-runtime-attestation/v2":
        raise RuntimeError("runtime_attestation_schema")
    if (
        attestation["runtimeHash"] != profile["runtimeHash"]
        or attestation["modelManifestHash"] != profile["modelManifestHash"]
        or attestation["rendererHash"] != profile["rendererHash"]
    ):
        raise RuntimeError("runtime_attestation_hash_mismatch")
    if attestation["comfyRevision"] != runtime["comfyRevision"]:
        raise RuntimeError("runtime_attestation_revision_mismatch")
    names = {
        "unetName": runtime["unetName"],
        "encoderName": runtime["encoderName"],
        "videoVaeName": runtime["videoVaeName"],
        "audioVaeName": runtime["audioVaeName"],
    }
    if attestation["checkpoints"] != names:
        raise RuntimeError("runtime_attestation_checkpoint_mismatch")
    expected_hashes = runtime.get("checkpointSha256")
    if (
        not isinstance(expected_hashes, dict)
        or set(expected_hashes) != set(names)
        or attestation["checkpointSha256"] != expected_hashes
    ):
        raise RuntimeError("runtime_attestation_checkpoint_sha_mismatch")
    if runtime.get("numerics", {}).get("mode") != "bit_exact":
        raise RuntimeError("runtime_model_equality_not_bit_exact")
    measurement = runtime.get("measurement")
    if not isinstance(measurement, dict) or sha256(measurement) != profile["runtimeHash"]:
        raise RuntimeError("runtime_measurement_binding")
    code_root = Path(__file__).resolve().parents[1]
    code = {relative: file_sha256(code_root / relative) for relative in RUNTIME_CODE}
    if measurement.get("code") != code:
        raise RuntimeError("runtime_code_changed")
    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True, timeout=30
    ).stdout.splitlines()[0]
    if (
        measurement.get("ffmpeg") != ffmpeg
        or profile["workflowHash"] != code["src/forge/inference/trace_runtime.py"]
        or profile["rendererHash"] != renderer_fingerprint(code, expected_hashes, ffmpeg)
    ):
        raise RuntimeError("runtime_renderer_changed")


def diagnostic_only(report: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """An unqualified profile can inspect evidence but cannot publish media."""
    if not enabled:
        return report
    diagnostic = report.get("verdict")
    diagnostic_reason = report.get("reason")
    safe = {key: value for key, value in report.items() if key not in {"result", "finalLatent"}}
    safe["verdict"] = "inconclusive"
    safe["reason"] = "diagnostic_profile_unqualified"
    safe["diagnosticVerdict"] = diagnostic
    safe["diagnosticReason"] = diagnostic_reason
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "request",
        "receipt",
        "profile-root",
        "evidence-root",
        "worker-public-key-file",
        "output-dir",
        "runtime-attestation",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--comfy-origin", required=True)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--comfy-output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--allow-unqualified-diagnostic", action="store_true")
    parser.add_argument("--export-diagnostic-artifact", action="store_true")
    args = parser.parse_args()
    if args.export_diagnostic_artifact and not args.allow_unqualified_diagnostic:
        parser.error("--export-diagnostic-artifact requires --allow-unqualified-diagnostic")
    output = args.output_dir.resolve()
    output_created = False
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "verdict": "inconclusive",
        "reason": "unknown",
    }
    try:
        origin = urllib.parse.urlparse(args.comfy_origin)
        if (
            origin.scheme != "http"
            or origin.hostname not in {"127.0.0.1", "::1"}
            or not origin.port
        ):
            raise RuntimeError("comfy_origin_must_be_loopback")
        if output.exists() or not 60 <= args.timeout_seconds <= 3600:
            raise RuntimeError("output_exists_or_timeout_invalid")
        output.mkdir(parents=True, mode=0o700)
        output_created = True
        request, receipt = read_json(args.request), read_json(args.receipt)
        validate_request(request)
        trace = receipt.get("trace")
        report.update(
            {
                "jobId": request["jobId"],
                "requestHash": sha256(request),
                "receiptHash": sha256(receipt),
                "profile": request["profile"],
                "traceRoot": trace.get("root") if isinstance(trace, dict) else None,
            }
        )
        envelope = load_profile_envelope(
            args.profile_root.resolve(),
            request["profile"],
            allow_unqualified=args.allow_unqualified_diagnostic,
        )
        profile = envelope["profile"]
        report["rendererHash"] = profile["rendererHash"]
        lane = {
            "hotkey": args.hotkey,
            "workerId": request["workerId"],
            "profileId": profile["id"],
            "profileHash": sha256(profile),
            "rendererHash": profile["rendererHash"],
        }
        report["lane"] = lane
        if not args.allow_unqualified_diagnostic:
            qualified_lane(profile, envelope["qualification"], args.hotkey, request["workerId"])
        validate_runtime_attestation(args.runtime_attestation, profile)
        verified = verify_bundle(
            request,
            receipt,
            args.evidence_root.resolve(),
            profile,
            args.worker_public_key_file.read_text(encoding="ascii").strip(),
        )
        graph_profile = NativeH3Profile.from_validated(profile)
        bind_capture_to_chunks(verified["capture"], verified["capturePath"], verified["chunks"])
        bind_receipt_final_latent(
            verified["capture"], verified["capturePath"], verified["finalLatent"]
        )
        try:
            validate_capture_metadata(verified["capture"], request, profile)
        except RuntimeError as exc:
            # The signature and immutable chunk hashes above establish who
            # committed these contradictory metadata, before GPU work begins.
            raise TraceProtocolError(str(exc)) from exc
        seed = effective_seed(
            request["jobId"], request["nonce"], request["requestedSeed"], request["profile"]["hash"]
        )
        trusted = TrustedT2VInput.from_validated(
            {
                "jobId": request["jobId"],
                "nonce": request["nonce"],
                "parametersHash": sha256(request["parameters"]),
                "requestHash": sha256(request),
                "parameters": request["parameters"],
                "effectiveSeed": seed,
            }
        )
        write_new(
            output / "intent.json",
            {
                "requestHash": trusted.request_hash,
                "receiptHash": verified["receiptHash"],
                "profileHash": graph_profile.profile_hash,
            },
        )
        started = time.monotonic()
        baseline = build_trusted_baseline_graph(
            trusted,
            graph_profile,
            trace_root=output / "trusted-trace",
            filename_prefix=f"forge-trusted/{trusted.job_id}",
        )
        write_graph_new(output / "baseline-graph.json", baseline)
        run_graph(
            args.comfy_origin.rstrip("/"),
            baseline,
            "trusted-baseline-" + trusted.job_id,
            args.timeout_seconds,
        )
        preprocess_ms = int((time.monotonic() - started) * 1000)
        reference_path = output / "trusted-trace" / trusted.job_id / "manifest.json"
        reference = read_json(reference_path)
        started = time.monotonic()
        compare_capture(
            verified["capture"],
            verified["capturePath"],
            reference,
            reference_path,
            request,
            profile,
        )
        transition = residual(verified["capturePath"])
        threshold = float(profile["numerics"]["maxRelativeRms"])
        if (
            transition.get("status") != "MEASURED"
            or transition.get("maxRelativeRms", float("inf")) > threshold
        ):
            raise RuntimeError("sampler_recurrence_mismatch")
        # Candidate payloads and all 20 predictions were compared exactly to
        # the locally executed baseline above. A second 20-call replay would
        # repeat that same approved model computation without adding evidence.
        replay_ms = int((time.monotonic() - started) * 1000)
        started = time.monotonic()
        render = build_validated_latent_render_graph(
            graph_profile,
            candidate_manifest=verified["capturePath"],
            filename_prefix=f"forge-verified/{trusted.job_id}",
        )
        write_graph_new(output / "render-graph.json", render)
        history = run_graph(
            args.comfy_origin.rstrip("/"),
            render,
            "trusted-render-" + trusted.job_id,
            args.timeout_seconds,
        )
        source_media = rendered_media(history, args.comfy_output_root)
        media = trim_and_validate_media(source_media, output / "result.mp4")
        render_ms = int((time.monotonic() - started) * 1000)
        report = {
            "schema": RESULT_SCHEMA,
            "jobId": request["jobId"],
            "requestHash": trusted.request_hash,
            "receiptHash": verified["receiptHash"],
            "profile": request["profile"],
            "traceRoot": receipt["trace"]["root"],
            "verdict": "verified",
            "reason": "verified",
            "checks": {
                "conditioning": "pass",
                "predictions": {"checked": 20, "matched": 20, "total": 20},
                "recurrence": "pass",
                "terminalBinding": "pass",
            },
            "finalLatent": {
                "video": file_sha256(verified["finalLatent"]["video"]),
                "audio": file_sha256(verified["finalLatent"]["audio"]),
            },
            "rendererHash": profile["rendererHash"],
            "lane": lane,
            "result": {
                "path": str(media),
                "sha256": file_sha256(media),
                "bytes": media.stat().st_size,
            },
            "timingsMs": {"preprocess": preprocess_ms, "replay": replay_ms, "render": render_ms},
        }
        if args.export_diagnostic_artifact:
            # Separate private research custody from the public verifier report.
            # The report remains inconclusive and contains no deliverable result.
            write_new(
                output / "diagnostic-artifact.json",
                {
                    "schema": "forge/h3-qualification-artifact/v1",
                    "purpose": "qualification_only",
                    "reportHash": sha256(diagnostic_only(report, True)),
                    "media": {
                        "name": "result.mp4",
                        "sha256": report["result"]["sha256"],
                        "bytes": report["result"]["bytes"],
                    },
                },
            )
    except TraceProtocolError as exc:
        report.update({"verdict": exc.category, "reason": exc.reason})
    except (
        RuntimeGraphError,
        RuntimeError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        # Preserve useful fixed reason codes without publishing file paths,
        # raw process errors or user-controlled content through downstream UI.
        reason = str(exc)
        if (
            not reason
            or len(reason) > 96
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in reason)
        ):
            reason = "trusted_runtime_error"
        report.update({"verdict": "inconclusive", "reason": reason})
    finally:
        report = diagnostic_only(report, args.allow_unqualified_diagnostic)
        if output_created:
            write_new(output / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
