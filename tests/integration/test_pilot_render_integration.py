"""Real CPU codec/derivation boundaries; synthetic renders never qualify a model."""

import importlib.util
import json
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import pytest

from forge.evaluation.pilot_worklist import pilot_worklist
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.hashing import sha256_file


def test_frozen_pilot_derivation_codec_receipts_and_restart(tmp_path):
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "pilot_render", root / "scripts/evaluator_pilot_render.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    pilot = json.loads((root / "configs/evaluator-pilot-manifest-v1.json").read_text())
    manifest = ModelManifest.from_dict(
        json.loads((root / "configs/wan50-reference.json").read_text())
    )
    bundle = runner.prepare(
        pilot=pilot, manifest=manifest, base_dir=tmp_path, deadline_unix=time.time() + 600
    )
    assert len(bundle.jobs) == 34
    assert len({job.job_id for job in bundle.jobs}) == 34
    assert bundle.contract.max_steps == 50
    assert sum(job.kind == "warmup" for job in bundle.jobs) == 2
    work = pilot_worklist(pilot)
    # Small CPU codec fixtures only: no assertion of real Wan generation/quality.
    work["recipe"].update(width=32, height=32)
    geometry = {key: work["recipe"][key] for key in ("width", "height", "frames", "fps")}
    source = tmp_path / "synthetic.mp4"
    frames = np.random.default_rng(8).integers(0, 256, (81, 32, 32, 3), dtype=np.uint8)
    with imageio.get_writer(str(source), fps=16, codec="libx264", macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    renders = tmp_path / "renders"
    renders.mkdir()
    for item in [*work["warmups"], *work["renders"]]:
        video = renders / f"{item['job_id']}.mp4"
        shutil.copyfile(source, video)
        receipt = {
            "status": "OK",
            "job_hash": runner.diagnostic_job(item, manifest, tmp_path).job_hash(),
            "video_sha256": sha256_file(video),
            "round_id": work["pilot_id"],
            "diagnostic_only": True,
            "geometry": geometry,
        }
        (renders / f"{item['job_id']}.json").write_text(json.dumps(receipt))
    index = runner.derive_and_index(
        work=work, manifest=manifest, root=tmp_path, deadline_unix=time.time() + 600
    )
    assert len(index["videos"]) == 88 and len(index["generation_receipts"]) == 34
    assert sum(row["transformation"] is not None for row in index["videos"]) == 56
    files = list((tmp_path / "derived").iterdir())
    before = {path: path.stat().st_mtime_ns for path in files}
    assert (
        runner.derive_and_index(
            work=work, manifest=manifest, root=tmp_path, deadline_unix=time.time() + 600
        )
        == index
    )
    assert {path: path.stat().st_mtime_ns for path in files} == before
    transform = work["transforms"][0]
    receipt_path = tmp_path / "derived" / f"{transform['job_id']}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "RUNNING"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="partial or changed transformation"):
        runner.derive_and_index(
            work=work, manifest=manifest, root=tmp_path, deadline_unix=time.time() + 600
        )
