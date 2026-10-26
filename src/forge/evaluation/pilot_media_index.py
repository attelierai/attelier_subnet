"""Bind the complete pilot media index to planned renders and executed transforms."""

from forge.protocol.hashing import hash_canonical


def validate_pilot_media_index(work: dict, index: dict) -> dict:
    if (
        index.get("schema") != "forge/evaluator-pilot-media-index/v1"
        or index.get("status") != "COMPLETE"
        or index.get("worklist_id") != work["worklist_id"]
        or index.get("recipe") != work["recipe"]
    ):
        raise ValueError("media index must bind the complete frozen pilot")
    rows = index.get("videos")
    if (
        not isinstance(rows, list)
        or len(rows) != 88
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("exactly 88 media records are required")
    videos = {row.get("job_id"): row for row in rows}
    expected = {row["job_id"]: row for row in work["renders"]}
    for row in work["transforms"]:
        expected[row["job_id"]] = expected[row["reference_job_id"]]
    if len(videos) != 88 or set(videos) != set(expected):
        raise ValueError("media job IDs must be unique and exactly match the worklist")
    for identity, row in videos.items():
        if any(row.get(key) != expected[identity][key] for key in ("prompt_text", "seed")):
            raise ValueError("media prompt or seed differs from the frozen job")
        digest = row.get("video_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("media digest must be a lowercase SHA-256")
    for job in work["transforms"]:
        row = videos[job["job_id"]]
        expected_transformation = {
            "reference_job_id": job["reference_job_id"],
            "reference_video_sha256": videos[job["reference_job_id"]]["video_sha256"],
            "kind": job["kind"],
            "options": job["options"],
            "candidate_video_sha256": row["video_sha256"],
        }
        if row.get("transformation") != expected_transformation:
            raise ValueError("transformation receipt differs from frozen treatment or media bytes")
    # A checksum is useful for downstream receipts, not a claim of remote attestation.
    return {
        "videos": videos,
        "index_id": hash_canonical("forge/evaluator-pilot-media-index/v1", index),
    }
