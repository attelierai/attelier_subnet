"""Filesystem/process boundary for the single-writer inference publication intent."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from forge.inference.publication import _claim_publication_intent, _write_publication_intent


def _hold(
    directory: str, receipt: str, ready: multiprocessing.Event, release: multiprocessing.Event
) -> None:
    lock, intent = _claim_publication_intent(
        Path(directory), "test", 578, "validator", Path(receipt)
    )
    _write_publication_intent(intent, "test", 578, "validator", Path(receipt))
    ready.set()
    release.wait(10)
    lock.close()


def test_publication_intent_is_single_writer_and_survives_output_directory_changes(tmp_path):
    receipt = tmp_path / "first" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps({"schema": "forge/inference-publication/v1", "status": "SUBMITTED"})
    )
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    process = multiprocessing.Process(
        target=_hold, args=(str(tmp_path), str(receipt), ready, release)
    )
    process.start()
    try:
        assert ready.wait(5)
        with pytest.raises(ValueError, match="Another inference publication"):
            _claim_publication_intent(
                tmp_path, "test", 578, "validator", tmp_path / "other" / "receipt.json"
            )
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0
    with pytest.raises(ValueError, match="Existing pending"):
        _claim_publication_intent(
            tmp_path, "test", 578, "validator", tmp_path / "other" / "receipt.json"
        )
