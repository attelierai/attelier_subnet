import hashlib
import json
import struct
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from forge.models.artifact_manifest import (
    ArtifactError,
    ArtifactReference,
    ManifestFile,
    ModelManifest,
)
from forge.models.artifacts import fetch_verified
from forge.protocol.contract import ChallengeContract
from forge.validator.runtime_preflight import (
    DUPLICATE,
    INCOMPLETE,
    PASS,
    preflight_candidate_batch,
    validate_preflight_receipt,
)


@contextmanager
def _artifact_server(files: dict[str, bytes]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = files.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    offset, header, payload = 0, {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _file(path: Path, relative: str) -> ManifestFile:
    data = path.read_bytes()
    return ManifestFile(relative, hashlib.sha256(data).hexdigest(), len(data))


def _base(root: Path) -> ModelManifest:
    weight = root / "transformer/model.safetensors"
    _write_safetensors(
        weight, {"block.attn.weight": ("F32", (2, 3), struct.pack("<6f", *range(6)))}
    )
    return ModelManifest(
        ArtifactReference("hf", "public/base", "b" * 40),
        "pipeline",
        (_file(weight, "transformer/model.safetensors"),),
        "apache-2.0",
        {"steps": 16, "guidance_scale": 5.0},
    )


def _candidate(root: Path, revision: str, steps: int) -> ModelManifest:
    weight = root / "adapter.safetensors"
    _write_safetensors(
        weight,
        {
            "transformer.block.attn.lora_A.weight": ("F32", (2, 3), struct.pack("<6f", *range(6))),
            "transformer.block.attn.lora_B.weight": ("F32", (2, 2), struct.pack("<4f", *range(4))),
        },
    )
    return ModelManifest(
        ArtifactReference("hf", "public/candidate", revision),
        "lora",
        (_file(weight, "adapter.safetensors"),),
        "apache-2.0",
        {"steps": steps, "guidance_scale": 5.0},
        ArtifactReference("hf", "public/base", "b" * 40),
    )


def test_preflight_deduplicates_repackaged_exact_tensors_but_keeps_a_new_recipe(tmp_path):
    contract = ChallengeContract(
        max_candidates=16, base_model_id="public/base", base_model_revision="b" * 40
    )
    base_root = tmp_path / "base"
    base = _base(base_root)
    first_root, copy_root, altered_recipe_root = (
        tmp_path / name for name in ("first", "copy", "steps12")
    )
    first = _candidate(first_root, "a" * 40, 16)
    copy = _candidate(copy_root, "c" * 40, 16)
    steps12 = _candidate(altered_recipe_root, "d" * 40, 12)
    admissions = [
        {
            "manifest_hash": model.fingerprint(),
            "manifest": model.to_dict(),
            "block": block,
            "hotkey": hotkey,
        }
        for block, hotkey, model in (
            (10, "first", first),
            (11, "copy", copy),
            (12, "steps12", steps12),
        )
    ]
    plan = {"contract_hash": contract.contract_hash(), "admitted": admissions}

    receipt = preflight_candidate_batch(
        plan,
        contract,
        base,
        base_root,
        ((first, first_root), (copy, copy_root), (steps12, altered_recipe_root)),
    )

    statuses = {row["manifest_hash"]: row["status"] for row in receipt["candidates"]}
    assert statuses == {
        first.fingerprint(): PASS,
        copy.fingerprint(): DUPLICATE,
        steps12.fingerprint(): PASS,
    }
    assert (
        validate_preflight_receipt(
            receipt, plan, contract, candidate_manifest_hash=steps12.fingerprint()
        )["status"]
        == PASS
    )
    with pytest.raises(ValueError, match="did not pass unique"):
        validate_preflight_receipt(
            receipt, plan, contract, candidate_manifest_hash=copy.fingerprint()
        )
    tampered = deepcopy(receipt)
    duplicate = next(row for row in tampered["candidates"] if row["status"] == DUPLICATE)
    duplicate["duplicate_of_manifest_hash"] = steps12.fingerprint()
    with pytest.raises(ValueError, match="duplicate row"):
        validate_preflight_receipt(
            tampered, plan, contract, candidate_manifest_hash=first.fingerprint()
        )


def test_full_transformer_can_share_the_verified_base_cache_with_extra_runtime_files(tmp_path):
    contract = ChallengeContract(
        max_candidates=16, base_model_id="public/base", base_model_revision="b" * 40
    )
    base_root = tmp_path / "base"
    base = _base(base_root)
    config = base_root / "transformer/config.json"
    config.write_text('{"hidden_size": 3, "_class_name": "WanTransformer3DModel"}')
    tokenizer = base_root / "tokenizer/tokenizer.json"
    tokenizer.parent.mkdir()
    tokenizer.write_text("{}")
    base = ModelManifest(
        base.reference,
        "pipeline",
        (
            *base.files,
            _file(config, "transformer/config.json"),
            _file(tokenizer, "tokenizer/tokenizer.json"),
        ),
        "apache-2.0",
        {"steps": 16, "guidance_scale": 5.0},
    )
    candidate = ModelManifest(
        base.reference,
        "full_transformer",
        (base.files[0],),
        "apache-2.0",
        {"steps": 16, "guidance_scale": 5},
        base.reference,
    )
    canonical_equivalent = ModelManifest(
        base.reference,
        "full_transformer",
        (base.files[0],),
        "apache-2.0",
        {"steps": 16, "guidance_scale": 5.0},
        base.reference,
    )
    assert candidate.fingerprint() != canonical_equivalent.fingerprint()
    plan = {
        "contract_hash": contract.contract_hash(),
        "admitted": [
            {
                "manifest_hash": model.fingerprint(),
                "manifest": model.to_dict(),
                "block": block,
                "hotkey": hotkey,
            }
            for block, hotkey, model in (
                (10, "full-transformer", candidate),
                (11, "numeric-repack", canonical_equivalent),
            )
        ],
    }
    receipt = preflight_candidate_batch(
        plan,
        contract,
        base,
        base_root,
        ((candidate, base_root), (canonical_equivalent, base_root)),
    )
    assert [row["status"] for row in receipt["candidates"]] == [PASS, DUPLICATE]


def test_preflight_keeps_valid_candidate_when_public_fetch_or_validation_fails(tmp_path):
    contract = ChallengeContract(
        max_candidates=16, base_model_id="public/base", base_model_revision="b" * 40
    )
    base_root = tmp_path / "base"
    base = _base(base_root)
    valid_source, unavailable_source, invalid_root = (
        tmp_path / name for name in ("valid-source", "unavailable-source", "invalid")
    )
    valid = _candidate(valid_source, "a" * 40, 16)
    unavailable = _candidate(unavailable_source, "c" * 40, 16)
    invalid = _candidate(invalid_root, "d" * 40, 12)
    invalid_file = invalid_root / "adapter.safetensors"
    invalid_file.write_bytes(b"not a safetensors file")
    admissions = [
        {
            "manifest_hash": manifest.fingerprint(),
            "manifest": manifest.to_dict(),
            "block": block,
            "hotkey": hotkey,
        }
        for block, hotkey, manifest in (
            (10, "valid", valid),
            (11, "unavailable", unavailable),
            (12, "invalid", invalid),
        )
    ]
    plan = {"contract_hash": contract.contract_hash(), "admitted": admissions}
    valid_bytes = (valid_source / "adapter.safetensors").read_bytes()
    valid_path = "/public/candidate/resolve/" + "a" * 40 + "/adapter.safetensors"
    with _artifact_server({valid_path: valid_bytes}) as server:
        downloaded = fetch_verified(valid, tmp_path / "cache-valid", hf_base_url=server)
        with pytest.raises(ArtifactError):
            fetch_verified(unavailable, tmp_path / "cache-unavailable", hf_base_url=server)
    receipt = preflight_candidate_batch(
        plan,
        contract,
        base,
        base_root,
        ((valid, downloaded), (unavailable, None), (invalid, invalid_root)),
    )
    rows = {row["manifest_hash"]: row for row in receipt["candidates"]}
    assert rows[valid.fingerprint()]["status"] == PASS
    assert rows[unavailable.fingerprint()] == {
        "manifest_hash": unavailable.fingerprint(),
        "status": INCOMPLETE,
        "stage": "fetch",
    }
    assert rows[invalid.fingerprint()] == {
        "manifest_hash": invalid.fingerprint(),
        "status": INCOMPLETE,
        "stage": "validation",
    }
    valid_row = validate_preflight_receipt(
        receipt, plan, contract, candidate_manifest_hash=valid.fingerprint()
    )
    assert valid_row["status"] == PASS
    with pytest.raises(ValueError, match="did not pass unique"):
        validate_preflight_receipt(
            receipt, plan, contract, candidate_manifest_hash=unavailable.fingerprint()
        )
    forged = deepcopy(receipt)
    row = next(item for item in forged["candidates"] if item["status"] == INCOMPLETE)
    row["weight_content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="incomplete row"):
        validate_preflight_receipt(
            forged, plan, contract, candidate_manifest_hash=valid.fingerprint()
        )


def test_preflight_still_aborts_for_a_bad_base_or_plan(tmp_path):
    contract = ChallengeContract(
        max_candidates=16, base_model_id="public/base", base_model_revision="b" * 40
    )
    base_root, candidate_root = tmp_path / "base", tmp_path / "candidate"
    base = _base(base_root)
    candidate = _candidate(candidate_root, "a" * 40, 16)
    plan = {
        "contract_hash": contract.contract_hash(),
        "admitted": [
            {
                "manifest_hash": candidate.fingerprint(),
                "manifest": candidate.to_dict(),
                "block": 10,
                "hotkey": "candidate",
            }
        ],
    }
    bad_plan = dict(plan, contract_hash="0" * 64)
    with pytest.raises(ValueError, match="contract hash"):
        preflight_candidate_batch(
            bad_plan, contract, base, base_root, ((candidate, candidate_root),)
        )
    (base_root / "transformer/model.safetensors").write_bytes(b"corrupt base")
    with pytest.raises(ValueError, match="model file failed"):
        preflight_candidate_batch(plan, contract, base, base_root, ((candidate, candidate_root),))


def test_committed_scheduler_survives_manifest_and_preflight_without_alias_bypass(tmp_path):
    from dataclasses import replace

    from forge.models.generation import GenerationRecipe, paired_reference_scheduler

    contract = ChallengeContract(
        max_candidates=16, base_model_id="public/base", base_model_revision="b" * 40
    )
    base_root = tmp_path / "base"
    base = _base(base_root)
    root = tmp_path / "candidate"
    legacy = _candidate(root, "a" * 40, 16)
    explicit = replace(
        legacy, generation_config={**dict(legacy.generation_config), "scheduler": "uniform_euler"}
    )
    unipc = replace(
        legacy, generation_config={**dict(legacy.generation_config), "scheduler": "original_unipc"}
    )
    models = [ModelManifest.from_dict(m.to_dict()) for m in (legacy, explicit, unipc)]
    assert models[0].fingerprint() == legacy.fingerprint()
    assert "scheduler" not in models[0].generation_config
    plan = {
        "contract_hash": contract.contract_hash(),
        "admitted": [
            {
                "manifest_hash": m.fingerprint(),
                "manifest": m.to_dict(),
                "block": 10 + i,
                "hotkey": f"miner-{i}",
            }
            for i, m in enumerate(models)
        ],
    }
    receipt = preflight_candidate_batch(
        plan, contract, base, base_root, [(m, root) for m in models]
    )
    assert [row["status"] for row in receipt["candidates"]] == [PASS, DUPLICATE, PASS]
    assert (
        paired_reference_scheduler(
            unipc, base, GenerationRecipe.from_dict(dict(unipc.generation_config)), contract
        )
        == "original_unipc"
    )
    for malformed in ("remote_python", {"class": "remote"}, None):
        payload = unipc.to_dict()
        payload["generation_config"]["scheduler"] = malformed
        with pytest.raises(ArtifactError, match="scheduler"):
            ModelManifest.from_dict(payload)


def test_miner_cli_preflight_uses_validator_checks_without_round_or_admission(tmp_path):
    import shutil

    from forge.cli import main
    from forge.validator.validator_cli import _cache_path

    base = _base(tmp_path / "base")
    candidate = _candidate(tmp_path / "candidate", "a" * 40, 16)
    cache = tmp_path / "cache"
    shutil.copytree(tmp_path / "base", _cache_path(cache, base))
    shutil.copytree(tmp_path / "candidate", _cache_path(cache, candidate))
    contract = ChallengeContract(base_model_id="public/base", base_model_revision="b" * 40)
    for name, value in (
        ("base", base.to_dict()),
        ("candidate", candidate.to_dict()),
        ("contract", contract.to_dict()),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    output = tmp_path / "receipt.json"
    arguments = [
        "preflight-submission",
        "--manifest",
        str(tmp_path / "candidate.json"),
        "--base-manifest",
        str(tmp_path / "base.json"),
        "--contract",
        str(tmp_path / "contract.json"),
        "--cache-dir",
        str(cache),
        "--out",
        str(output),
    ]
    assert main(arguments) == 0
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "PUBLIC_ARTIFACT_PREFLIGHT_PASSED"
    assert receipt["admission_granted"] is False
    assert receipt["quality_evaluated"] is False
    assert receipt["duplicate_history_checked"] is False
    plan = {
        "contract_hash": contract.contract_hash(),
        "admitted": [
            {
                "block": 1,
                "hotkey": "registered-miner",
                "manifest": candidate.to_dict(),
                "manifest_hash": candidate.fingerprint(),
            }
        ],
    }
    validator = preflight_candidate_batch(
        plan,
        contract,
        base,
        _cache_path(cache, base),
        [(candidate, _cache_path(cache, candidate))],
    )
    assert (
        receipt["effective_candidate_hash"]
        == validator["candidates"][0]["effective_candidate_hash"]
    )
    with pytest.raises(ValueError, match="output must be new"):
        main(arguments)
    # Limits fail in CPU preflight, using the same cached valid tensors.
    arguments[-1] = str(tmp_path / "limited.json")
    for override, reason in (
        ({"max_steps": 8}, "step limit"),
        ({"max_forward_passes": 16}, "forward-pass limit"),
        ({"base_model_revision": "c" * 40}, "pinned challenge"),
    ):
        (tmp_path / "contract.json").write_text(json.dumps({**contract.to_dict(), **override}))
        with pytest.raises(ArtifactError, match=reason):
            main(arguments)
        assert not (tmp_path / "limited.json").exists()
    (tmp_path / "contract.json").write_text(json.dumps(contract.to_dict()))
    # A validly hashed snapshot with an undeclared loadable file is still refused.
    (_cache_path(cache, candidate) / "untrusted.py").write_text("raise RuntimeError()")
    arguments[-1] = str(tmp_path / "rejected.json")
    with pytest.raises(ValueError, match="undeclared loadable"):
        main(arguments)
    assert not (tmp_path / "rejected.json").exists()
