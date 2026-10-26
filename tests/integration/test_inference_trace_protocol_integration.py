"""Real signed, stored protocol exchange; these tests do not execute H3."""

import base64
import copy
import json
import subprocess
import zipfile

import pytest

from forge.inference.trace_protocol import (
    FIXED_PARAMETERS,
    PROFILE_SCHEMA,
    RECEIPT_SCHEMA,
    REQUEST_SCHEMA,
    TraceProtocolError,
    canonical,
    effective_seed,
    file_sha256,
    load_profile,
    sha256,
    validate_receipt_metadata,
    validate_request,
    verify_bundle,
)


def test_signed_trace_custody_profile_gate_and_tampering(tmp_path):
    evidence, profiles = tmp_path / "evidence", tmp_path / "profiles"
    (evidence / "tensors").mkdir(parents=True)
    profiles.mkdir()
    key = tmp_path / "signer.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(key)],
        check=True,
        capture_output=True,
    )
    public = subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout[-32:]
    public_key = base64.b64encode(public).decode()
    profile = {
        "id": "h3-native-blackwell-20-v1",
        "modelManifestHash": "1" * 64,
        "workflowHash": "2" * 64,
        "runtimeHash": "3" * 64,
        "rendererHash": "4" * 64,
        "steps": 20,
        "parameters": FIXED_PARAMETERS,
        "numerics": {"maxRelativeRms": "0.000001"},
        "limits": {"maxChunkBytes": 1048576, "maxBundleBytes": 4194304, "maxChunks": 256},
    }
    identity = {"id": profile["id"], "hash": sha256(profile)}
    profile_file = profiles / (profile["id"] + ".json")
    profile_envelope = {
        "schema": PROFILE_SCHEMA,
        "profile": profile,
        "qualification": {"status": "unqualified"},
    }
    profile_file.write_bytes(canonical(profile_envelope))
    with pytest.raises(TraceProtocolError, match="profile_unqualified"):
        load_profile(profiles, identity, now=10)
    profile_envelope["qualification"] = {
        "status": "qualified",
        "evidenceSha256": "e" * 64,
        "expiresAt": 100,
    }
    profile_file.write_bytes(canonical(profile_envelope))
    assert load_profile(profiles, identity, now=10) == profile
    with pytest.raises(TraceProtocolError, match="qualification_expired"):
        load_profile(profiles, identity, now=100)
    profile_envelope["qualification"]["expiresAt"] = 1_792_800_000_000
    profile_file.write_bytes(canonical(profile_envelope))
    with pytest.raises(TraceProtocolError, match="qualification_expired"):
        load_profile(profiles, identity, now=10)
    profile_envelope["qualification"]["expiresAt"] = 100
    profile_file.write_bytes(canonical(profile_envelope))

    # These tiny stored ZIPs exercise transport, not the GPU/tensor loader.
    final = []
    for component in ("video", "audio"):
        relative = f"tensors/{component}.pt"
        path = evidence / relative
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data/0", component.encode())
        final.append(
            {
                "component": component,
                "tensorPath": relative,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    (evidence / "capture.json").write_text(json.dumps({"finalLatent": final}))
    paths = [evidence / "capture.json", *(evidence / d["tensorPath"] for d in final)]
    chunks = [
        {
            "index": i,
            "name": str(path.relative_to(evidence)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for i, path in enumerate(paths)
    ]
    request = {
        "schema": REQUEST_SCHEMA,
        "jobId": "job-1",
        "workerId": "worker-1",
        "nonce": "a" * 64,
        "requestedSeed": 7,
        "profile": identity,
        "parameters": {
            **FIXED_PARAMETERS,
            "prompt": "Luz sobre el mar — 海",
            "seed": effective_seed("job-1", "a" * 64, 7, identity["hash"]),
        },
    }
    validate_request(request)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "jobId": "job-1",
        "workerId": "worker-1",
        "nonce": request["nonce"],
        "requestHash": sha256(request),
        "parametersHash": sha256(request["parameters"]),
        "profile": identity,
        "modelManifestHash": profile["modelManifestHash"],
        "workflowHash": profile["workflowHash"],
        "steps": 20,
        "trace": {"algorithm": "sha256-manifest-v1", "chunks": chunks, "root": sha256(chunks)},
        "finalLatent": [
            {"component": "video", "chunkIndex": 1},
            {"component": "audio", "chunkIndex": 2},
        ],
    }

    def sign(body):
        body = copy.deepcopy(body)
        body.pop("signature", None)
        message, signature = tmp_path / "message", tmp_path / "signature"
        message.write_bytes(canonical(body))
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
            check=True,
            capture_output=True,
        )
        body["signature"] = base64.b64encode(signature.read_bytes()).decode()
        return body

    receipt = sign(receipt)
    oversized = copy.deepcopy(receipt)
    oversized["trace"]["chunks"][0]["bytes"] = profile["limits"]["maxBundleBytes"] + 1
    oversized["trace"]["root"] = sha256(oversized["trace"]["chunks"])
    with pytest.raises(TraceProtocolError, match="chunk_size"):
        validate_receipt_metadata(request, sign(oversized), profile, public_key)

    checked = verify_bundle(request, receipt, evidence, profile, public_key)
    assert checked["finalLatent"] == {"video": paths[1], "audio": paths[2]}
    assert checked["receiptHash"] == sha256(receipt)
    assert "verdict" not in checked  # Signature/custody are not a model proof.

    replay_request = copy.deepcopy(request)
    replay_request["nonce"] = "b" * 64
    with pytest.raises(TraceProtocolError, match="effective_seed_binding"):
        verify_bundle(replay_request, receipt, evidence, profile, public_key)
    replay_request["parameters"]["seed"] = effective_seed("job-1", "b" * 64, 7, identity["hash"])
    with pytest.raises(TraceProtocolError, match="receipt_request_binding"):
        verify_bundle(replay_request, receipt, evidence, profile, public_key)

    changed = copy.deepcopy(receipt)
    changed["finalLatent"][0]["chunkIndex"] = 2
    with pytest.raises(TraceProtocolError, match="receipt_signature"):
        verify_bundle(request, changed, evidence, profile, public_key)
    with pytest.raises(TraceProtocolError, match="final_latent_alias"):
        verify_bundle(request, sign(changed), evidence, profile, public_key)

    changed = copy.deepcopy(receipt)
    changed["trace"]["chunks"][1]["name"] = "../outside.pt"
    changed["trace"]["root"] = sha256(changed["trace"]["chunks"])
    with pytest.raises(TraceProtocolError, match="chunk_name"):
        verify_bundle(request, sign(changed), evidence, profile, public_key)
    # A signed malformed capture must be a bounded rejection, not an uncaught
    # unhashable-type error in the consumer of the signed tensor references.
    original_capture = paths[0].read_bytes()
    malformed = {"finalLatent": copy.deepcopy(final)}
    malformed["finalLatent"][0]["tensorPath"] = ["tensors/video.pt"]
    paths[0].write_text(json.dumps(malformed))
    changed = copy.deepcopy(receipt)
    changed["trace"]["chunks"][0].update(
        sha256=file_sha256(paths[0]), bytes=paths[0].stat().st_size
    )
    changed["trace"]["root"] = sha256(changed["trace"]["chunks"])
    with pytest.raises(TraceProtocolError, match="capture_uncommitted_tensor"):
        verify_bundle(request, sign(changed), evidence, profile, public_key)
    paths[0].write_bytes(original_capture)
    paths[1].write_bytes(b"substituted")
    with pytest.raises(TraceProtocolError, match="chunk_custody"):
        verify_bundle(request, receipt, evidence, profile, public_key)


def test_protocol_json_hash_is_cross_language_and_strict(tmp_path):
    payload = {"z": 2**53 - 1, "a": "Luz — 海", "nested": {"b": None, "a": True}}
    data = tmp_path / "payload.json"
    data.write_bytes(canonical(payload))
    # Actual Node serialization of the already sorted protocol keys must match.
    script = (
        "const f=require('fs'),c=require('crypto');"
        "const v=JSON.parse(f.readFileSync(process.argv[1],'utf8'));"
        "process.stdout.write(c.createHash('sha256').update(JSON.stringify(v)).digest('hex'));"
    )
    node = subprocess.run(
        ["node", "-e", script, str(data)], capture_output=True, text=True, check=True
    )
    assert node.stdout == sha256(payload)
    for value in ({"x": float("nan")}, {"x": 1.0}, {"x": 2**53}, {"é": 1}):
        with pytest.raises(TraceProtocolError):
            canonical(value)
