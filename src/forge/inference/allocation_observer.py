"""Read-only API allocation checks alongside the burn publisher."""

from __future__ import annotations

import contextlib
import threading
import time
from functools import partial
from pathlib import Path

from forge.inference.final_allocation import FinalAllocationRunner, onchain_weights
from forge.inference.light import _atomic_write, fetch_feed, final_allocation_public_key
from forge.inference.light_registry import verify_finalized_registry
from forge.inference.prelaunch import digest


def observe_allocations(
    state_dir: Path,
    url: str,
    public_key_file: Path,
    *,
    network: str,
    interval: int,
    stop: threading.Event,
    once: bool = False,
    allow_http_fixture: bool = False,
) -> None:
    """Never receive a wallet or publisher; failures stay in observer state.

    Run in the publisher's daemon thread so slow gateway or registry reads do
    not delay burn publication. Keep the existing durable allocation journal.
    """
    runner = None
    try:
        while not stop.is_set():
            try:
                if runner is None:
                    runner = FinalAllocationRunner(
                        state_dir,
                        final_allocation_public_key(public_key_file),
                        registry_verifier=partial(verify_finalized_registry, network=network),
                    )
                body = runner.process(
                    fetch_feed(
                        url,
                        allow_http_fixture=allow_http_fixture,
                        production_path="/v1/subnet/126/final-allocation/latest",
                    )
                )
                relative = onchain_weights(body)
                _atomic_write(
                    state_dir / "live-dry-run.json",
                    {
                        "bodyDigest": digest(body),
                        "allocationShares": [
                            [row["uid"], row["weight"]] for row in body["weights"]
                        ],
                        "relativeWeights": [
                            [uid, weight] for uid, weight in sorted(relative.items())
                        ],
                    },
                )
                status = "ready"
            except Exception:
                # Do not log external exception text: it may include request
                # details. A failed observer must never stop the publisher.
                status = "unavailable"
                print("allocation_observer_unavailable", flush=True)
                if runner is not None:
                    with contextlib.suppress(Exception):
                        runner.unavailable("allocation_observer_unavailable")
            try:
                _atomic_write(
                    state_dir / "observer.json",
                    {
                        "status": status,
                        "checkedAtMs": int(time.time() * 1000),
                        "simulationOnly": True,
                        "publishable": False,
                    },
                )
            except Exception:
                print("allocation_observer_status_unavailable", flush=True)
            if once or stop.wait(interval):
                return
    finally:
        if runner is not None:
            runner._lock.close()


def start_allocation_observer(*args, **kwargs) -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()
    worker = threading.Thread(
        target=observe_allocations,
        args=args,
        kwargs={**kwargs, "stop": stop},
        name="allocation-observer",
        daemon=True,
    )
    worker.start()
    return worker, stop
