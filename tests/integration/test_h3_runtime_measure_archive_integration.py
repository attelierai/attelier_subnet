import hashlib
import importlib.util
import tarfile
from io import BytesIO
from pathlib import Path

import pytest


def runtime_measure_module():
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "h3_runtime_measure", root / "scripts" / "h3_runtime_measure.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add(archive: tarfile.TarFile, name: str, body: bytes) -> None:
    entry = tarfile.TarInfo(name)
    entry.size = len(body)
    archive.addfile(entry, BytesIO(body))


def test_runtime_measure_accepts_only_pinned_archive_tree_with_the_locked_quant_patch(tmp_path):
    module = runtime_measure_module()
    revision = "b0f4b7b294ce482a2e071d9d762c133d38c7aa07"
    prefix = f"ComfyUI-{revision}"
    archive = tmp_path / "ComfyUI.tar.gz"
    original = b"original quant ops\n"
    requirements = b"torch==2\n"
    with tarfile.open(archive, "w:gz") as bundle:
        directory = tarfile.TarInfo(prefix)
        directory.type = tarfile.DIRTYPE
        bundle.addfile(directory)
        add(bundle, f"{prefix}/comfy/ops.py", original)
        add(bundle, f"{prefix}/main.py", b"print('pinned')\n")
        add(bundle, f"{prefix}/requirements.txt", requirements)
    comfy = tmp_path / "installed"
    (comfy / "comfy").mkdir(parents=True)
    patched = b"locked NVFP4 quant ops\n"
    (comfy / "comfy" / "ops.py").write_bytes(patched)
    (comfy / "main.py").write_text("print('pinned')\n")
    (comfy / "requirements.txt").write_bytes(requirements)
    trace_node = comfy / "custom_nodes" / "attelier_h3_trace" / "__init__.py"
    trace_node.parent.mkdir(parents=True)
    trace_node.write_text("print('trusted trace node')\n")
    (tmp_path / "runtime-models").mkdir()
    (comfy / "models").symlink_to(tmp_path / "runtime-models", target_is_directory=True)
    lock = {
        "comfy": {
            "revision": revision,
            "nvfp4ScalePatch": {
                "target": "comfy/ops.py",
                "afterSha256": hashlib.sha256(patched).hexdigest(),
            },
        }
    }
    observed, archive_hash = module.validate_archive_source_tree(
        comfy, archive, lock, trace_node, hashlib.sha256(trace_node.read_bytes()).hexdigest()
    )
    assert observed["comfy/ops.py"] == hashlib.sha256(patched).hexdigest()
    assert observed["custom_nodes/attelier_h3_trace/__init__.py"] == hashlib.sha256(
        trace_node.read_bytes()
    ).hexdigest()
    assert archive_hash == hashlib.sha256(archive.read_bytes()).hexdigest()

    (comfy / "custom_nodes" / "evil.py").write_text("print('drift')\n")
    with pytest.raises(RuntimeError, match="comfy_source_tree_changed"):
        module.validate_archive_source_tree(
            comfy, archive, lock, trace_node, hashlib.sha256(trace_node.read_bytes()).hexdigest()
        )
