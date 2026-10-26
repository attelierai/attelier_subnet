"""Persistent provider-journal replay boundary; no external provider call."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from forge.evaluation.vision_judge import ApiBudget, judge_pair
from forge.protocol.contract import DEFAULT_AXES


def _png(pixel: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), pixel)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_luna_pair_reuses_only_journalled_settled_orders_and_never_redispatches(
    tmp_path, monkeypatch
):
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        response = {
            "id": f"response-{len(calls)}",
            "model": "gpt-6-luna",
            "status": "completed",
            "usage": {
                "input_tokens": 300_000,
                "output_tokens": 8192,
                "input_tokens_details": {"cache_write_tokens": 300_000},
            },
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "axes": {
                                        axis: {"a": 3, "b": 3, "reason": "same"}
                                        for axis in DEFAULT_AXES
                                    },
                                    "uncertain_axes": [],
                                }
                            ),
                        }
                    ],
                }
            ],
        }
        return _Response(json.dumps(response).encode())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("forge.evaluation.vision_judge.urllib.request.urlopen", urlopen)
    budget = ApiBudget(tmp_path / "luna.json", 1.0, reservation_usd=0.30)
    first = judge_pair(
        "a cube moves",
        [_png((1, 2, 3))],
        [_png((4, 5, 6))],
        budget=budget,
        env_file=None,
        pair_id="frozen-pair",
    )
    resumed = judge_pair(
        "a cube moves",
        [_png((1, 2, 3))],
        [_png((4, 5, 6))],
        budget=budget,
        env_file=None,
        pair_id="frozen-pair",
    )
    assert len(calls) == 2
    assert resumed["calls"] == first["calls"]

    with pytest.raises(ValueError, match="cannot be reconstructed"):
        judge_pair(
            "a cube moves",
            [_png((9, 9, 9))],
            [_png((4, 5, 6))],
            budget=budget,
            env_file=None,
            pair_id="frozen-pair",
        )
    assert len(calls) == 2

    ledger = json.loads(budget.path.read_text())
    assert ledger["reservation_usd"] == 0.30
    assert all(row["cost_usd"] == pytest.approx(0.081144) for row in ledger["calls"])
    # Restart binds the reserve policy and cannot silently restore legacy limits.
    with pytest.raises(ValueError, match="reservation differs"):
        ApiBudget(budget.path, 1.0).reserve()
