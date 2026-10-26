"""Actual supervisor/process/HTTP boundaries; no paid resource or provider key."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from forge.evaluation.pilot_guard import supervise


def test_lium_owned_closure_and_restart_preserve_other_pods(tmp_path):
    owned = {
        "id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "pod_name": "attelier-evaluator-pilot-lium-integration",
        "executor_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    }
    other = {**owned, "id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc", "pod_name": "unrelated"}
    pods, deletes, deleting_reads = [owned, other], [], []
    inventory = []
    owned["executor"] = {"id": owned["executor_id"], "specs": {"docker": {"containers": inventory}}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            assert self.headers["X-API-Key"] == "integration-only"
            if self.path == "/pods":
                value = [row for row in pods if row.get("status") != "DELETING"]
            else:
                value = next(
                    (row for row in pods if row["id"] == self.path.rsplit("/", 1)[-1]), None
                )
                if value and value.get("status") == "DELETING":
                    deleting_reads.append(time.time())
                    if len(deleting_reads) >= 4:
                        pods.remove(value)
                        value = None
                if value is None:
                    self.send_response(404)
                    self.end_headers()
                    return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_DELETE(self):
            pod_id = self.path.rsplit("/", 1)[-1]
            deletes.append(pod_id)
            next(row for row in pods if row["id"] == pod_id)["status"] = "DELETING"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    env_file = tmp_path / ".env"
    env_file.write_text("LIUM_API_KEY=integration-only\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = json.loads(
        (Path(__file__).parents[2] / "configs/evaluator-pilot-guard-envelope.json").read_text()
    )
    config["poll_seconds"] = 0.1
    config["backup_seconds"] = 1
    archive = tmp_path / "archive.py"
    archive.write_text("raise SystemExit(1)\n")
    ownership = {
        "pod_id": owned["id"],
        "name": owned["pod_name"],
        "executor_id": owned["executor_id"],
        "worklist_id": "frozen-lium-work",
        "envelope": config,
        "request_started_unix": time.time() - 20_000,
        "terminate_on_backup_failure_authorized": True,
        "archive_argv": [sys.executable, str(archive)],
        "repair_notify_argv": [sys.executable, "-c", "raise SystemExit(1)"],
    }
    contract = tmp_path / "ownership.json"
    contract.write_text(json.dumps(ownership))
    provider = tmp_path / "provider"
    provider.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        "from forge.evaluation.pilot_lium import LiumClient\n"
        f"client=LiumClient(Path({str(env_file)!r}),base_url='http://127.0.0.1:{server.server_port}')\n"
        f"ownership=json.loads(Path({str(contract)!r}).read_text())\n"
        "target=sys.argv[3] if sys.argv[2]!='list' else None\n"
        "print(json.dumps(client.provider_command(sys.argv[2],target,ownership)))\n"
    )
    provider.chmod(0o700)
    try:
        import pytest

        from forge.evaluation.pilot_lium import LiumClient

        client = LiumClient(env_file, base_url=f"http://127.0.0.1:{server.server_port}")
        digest = "sha256:" + "1" * 64
        assert client.running_container(ownership, digest) is None
        inventory.append(
            {"name": "pod_" + owned["id"], "digest": digest, "container_id": "actual-id"}
        )
        assert client.running_container(ownership, digest)["container_id"] == "actual-id"
        with pytest.raises(ValueError, match="differs"):
            client.running_container(ownership, "sha256:" + "2" * 64)
        state = tmp_path / "guard.json"
        state.with_suffix(".close").touch()
        assert supervise(ownership, state, provider=str(provider)) == 0
        assert json.loads(state.read_text())["status"] == "CLOSED"
        assert pods == [other] and deletes == [owned["id"]]
        assert len(deleting_reads) == 4
        assert supervise(ownership, state, provider=str(provider)) == 0
        assert deletes == [owned["id"]]
        # A direct adapter invocation also refuses a different target or renamed owner.
        import subprocess

        denied = subprocess.run([str(provider), "pod", "delete", other["id"]], capture_output=True)
        assert denied.returncode != 0 and pods == [other]
        pods.insert(0, {**owned, "pod_name": "another-campaign", "status": "RUNNING"})
        denied = subprocess.run([str(provider), "pod", "delete", owned["id"]], capture_output=True)
        assert denied.returncode != 0 and len(pods) == 2 and len(deletes) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
