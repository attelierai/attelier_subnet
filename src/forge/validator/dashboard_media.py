"""Public diagnostic snapshot data; never authorizes rewards."""

import hashlib
import io
import json
import shutil
from pathlib import Path

from forge.protocol.hashing import sha256_file


def public_examples(plan, quality, selection_path, panel_paths, media_out):
    """Export the preselected scenes, matching the exact videos judged in this run."""
    selection = json.loads(selection_path.read_text())
    identities = selection["prompt_ids"]
    assert identities and identities == [p["prompt_id"] for p in plan["prompts"][: len(identities)]]
    assert selection["examples_roles"] == ["golden", "incumbent", "candidate"]
    comparisons = {item["reference_role"]: item for item in quality["comparisons"]}
    reports = {role: json.loads(path.read_text()) for role, path in panel_paths.items()}
    media_out.mkdir(parents=True, exist_ok=True)
    examples = []
    for prompt in plan["prompts"][: len(identities)]:
        seed = prompt["video_seeds"][0]
        slot = (prompt["prompt_id"], seed)
        judged = {}
        for reference in ("golden", "incumbent"):
            rows = [
                r
                for r in comparisons[reference]["observations"]
                if (r["prompt_id"], r["seed"]) == slot
            ]
            assert len(rows) == 1
            judged[reference] = rows[0]["reference_video_sha256"]
            assert (
                judged.setdefault("candidate", rows[0]["candidate_video_sha256"])
                == rows[0]["candidate_video_sha256"]
            )
        clips = []
        for role, report in reports.items():
            assert report["contract_hash"] == plan["contract_hash"]
            assert report["prompts"] == plan["prompts"]
            rows = [r for r in report["observations"] if (r["prompt_id"], r["seed"]) == slot]
            assert len(rows) == 1 and rows[0]["status"] == "OK"
            row = rows[0]
            filename = row["video"]
            assert Path(filename).name == filename
            source = panel_paths[role].parent / filename
            assert not source.is_symlink() and source.is_file()
            digest = sha256_file(source)
            assert digest == row["video_sha256"] == judged[role]
            target = media_out / (digest + ".mp4")
            assert not target.is_symlink()
            if target.exists():
                assert sha256_file(target) == digest
            else:
                with source.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                assert sha256_file(target) == digest
            # A first-frame poster gives browsers an actual preview while they
            # load only MP4 metadata. It is derived from the verified clip.
            import imageio_ffmpeg
            from PIL import Image

            reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
            try:
                metadata = next(reader)
                assert metadata["size"] == (plan["geometry"]["width"], plan["geometry"]["height"])
                first_frame = next(reader)
            finally:
                reader.close()
            encoded = io.BytesIO()
            Image.frombytes("RGB", metadata["size"], first_frame).save(
                encoded, format="JPEG", quality=90
            )
            poster_bytes = encoded.getvalue()
            poster_digest = hashlib.sha256(poster_bytes).hexdigest()
            poster = media_out / (poster_digest + ".jpg")
            assert not poster.is_symlink()
            if poster.exists():
                assert sha256_file(poster) == poster_digest
            else:
                with poster.open("xb") as handle:
                    handle.write(poster_bytes)
            steps = row["steps"]
            clips.append(
                {
                    "label": {
                        "golden": "Original Wan reference",
                        "incumbent": "Wan base",
                        "candidate": "Public Wan candidate",
                    }[role],
                    "kind": role,
                    "steps": steps,
                    "sha256": digest,
                    "src": f"/media/{digest}.mp4",
                    "poster": f"/media/{poster_digest}.jpg",
                    "poster_sha256": poster_digest,
                    "note": "Original UniPC scheduler"
                    if role == "golden"
                    else "Uniform Euler scheduler; unchanged public base weights",
                }
            )
        examples.append(
            {
                "prompt_id": prompt["prompt_id"],
                "prompt": prompt["text"],
                "selection": (
                    "First two prompts from the frozen panel, selected before viewing results."
                ),
                "clips": clips,
                "note": (
                    "One fixed seed per scene. Examples illustrate this experiment; "
                    "they do not certify quality."
                ),
            }
        )
    return examples
