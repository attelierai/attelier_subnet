"""CLI for the dashboard report ingestion endpoint and durable validator outbox."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from forge.chain import Sr25519Verifier
from forge.evaluation.distributed import verify_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.protocol.distributed_transport import report_digest
from forge.validator.dashboard_ingestion import (
    MAX_REQUEST_BYTES,
    _json_loads,
    load_registered_plans,
    serve,
)


def add_commands(commands: argparse._SubParsersAction) -> None:
    api = commands.add_parser("dashboard-api", help="serve signed validator report ingestion")
    api.add_argument("--database", type=Path, required=True)
    api.add_argument("--plan", type=Path, action="append", default=[])
    api.add_argument("--plan-directory", type=Path)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, required=True)
    api.add_argument("--snapshot", type=Path)
    push = commands.add_parser("dashboard-push", help="push one already-signed report idempotently")
    push.add_argument("--plan", type=Path, required=True)
    push.add_argument("--report", type=Path, required=True)
    push.add_argument("--endpoint", required=True)
    push.add_argument("--outbox", type=Path, required=True)
    flush = commands.add_parser(
        "dashboard-flush", help="retry a directory of signed outbox reports"
    )
    flush.add_argument("--plan-directory", type=Path, required=True)
    flush.add_argument("--outbox-directory", type=Path, required=True)
    flush.add_argument("--endpoint", required=True)


def _write_once(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError("outbox already contains a different payload")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != value:
            raise ValueError("outbox already contains a different payload") from None
    finally:
        temporary.unlink(missing_ok=True)


def _endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or not parsed.hostname
    ):
        raise ValueError("dashboard endpoint must be a bare origin")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("dashboard endpoint requires HTTPS outside loopback")
    return parsed.geturl().rstrip("/")


def push(plan_path: Path, report_path: Path, endpoint: str, outbox: Path) -> dict:
    plan = DistributedEvaluationPlan.from_dict(_json_loads(plan_path.read_bytes()))
    raw = report_path.read_bytes()
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("report exceeds dashboard request limit")
    report = _json_loads(raw)
    if report.get("plan_hash") != plan.plan_hash():
        raise ValueError("report plan hash differs from --plan")
    verify_partial_report(plan, report, Sr25519Verifier())
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _write_once(outbox, payload)
    request = Request(
        _endpoint(endpoint) + "/api/v1/validator-reports",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "User-Agent": "forge-validator/0.1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            raw_reply = response.read(MAX_REQUEST_BYTES + 1)
            if len(raw_reply) > MAX_REQUEST_BYTES:
                raise RuntimeError("dashboard response exceeds size limit")
            reply = _json_loads(raw_reply)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("dashboard delivery pending; exact payload remains in outbox") from exc
    row = reply.get("report") if isinstance(reply, dict) else None
    if (
        reply.get("status") not in {"stored", "duplicate"}
        or not isinstance(row, dict)
        or (
            row.get("report_hash") != report_digest(report)
            or row.get("plan_hash") != plan.plan_hash()
            or row.get("validator_hotkey") != report.get("validator_hotkey")
        )
    ):
        raise RuntimeError("dashboard acknowledgement does not bind the submitted report")
    return {"status": reply.get("status"), "report_hash": row["report_hash"], "outbox": str(outbox)}


def flush_outbox(plans: Path, outbox: Path, endpoint: str) -> dict:
    """One bounded retry pass, suitable for a timer; never re-sign or re-render.

    The dedicated outbox contains only immutable signed reports. Acknowledgements
    are deliberately not authority: loss of one merely causes a harmless duplicate.
    Stop on a delivery failure so an API outage cannot cause an unbounded wait.
    """
    _endpoint(endpoint)
    if not plans.is_dir() or not outbox.is_dir():
        raise ValueError("plan and outbox directories must exist")
    files = sorted(plans.glob("*.json"))
    registered = load_registered_plans(files)
    plan_paths = {
        DistributedEvaluationPlan.from_dict(_json_loads(path.read_bytes())).plan_hash(): path
        for path in files
    }
    delivered, pending = [], []
    for path in sorted(outbox.glob("*.json")):
        report = _json_loads(path.read_bytes())
        digest = report.get("plan_hash")
        if digest not in registered:
            pending.append({"file": path.name, "reason": "PLAN_UNAVAILABLE"})
            continue
        try:
            result = push(plan_paths[digest], path, endpoint, path)
        except RuntimeError:
            pending.append({"file": path.name, "reason": "DELIVERY_PENDING"})
            break
        delivered.append({"file": path.name, "status": result["status"]})
    return {
        "status": "PENDING" if pending else "COMPLETE",
        "delivered": delivered,
        "pending": pending,
    }


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command == "dashboard-api":
        serve(args.database, args.plan, args.host, args.port, args.snapshot, args.plan_directory)
        return 0
    if args.command == "dashboard-push":
        print(json.dumps(push(args.plan, args.report, args.endpoint, args.outbox), allow_nan=False))
        return 0
    if args.command == "dashboard-flush":
        result = flush_outbox(args.plan_directory, args.outbox_directory, args.endpoint)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["status"] == "COMPLETE" else 1
    return None
