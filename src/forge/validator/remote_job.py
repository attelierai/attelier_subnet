"""Detached, idempotent SSH jobs for validator-owned remote workers.

The local SSH transport is deliberately not the job supervisor.  Once its
immutable intent is persisted, a short SSH launcher asks the remote host to
start one ``nohup`` wrapper.  Later callers only inspect that same remote job;
they never turn a lost transport into a second GPU command.
"""

from __future__ import annotations

import base64
import fcntl
import json
import math
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from forge.protocol.hashing import hash_canonical
from forge.validator.launch_service import atomic_json, write_immutable_json


class RemoteJobError(RuntimeError):
    """The persisted remote job completed unsuccessfully."""


def _encoded(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _ssh_argv(ssh_argv: list[str], remote: list[str]) -> list[str]:
    return [*ssh_argv, " ".join(shlex.quote(item) for item in remote)]


def _runner_script() -> str:
    return """set -u
d=$1; command=$2; deadline=$3
now=$(date +%s); seconds=$((deadline-now))
atomic() { tmp="$1.tmp.$$"; cat >"$tmp"; mv "$tmp" "$1"; }
if test "$seconds" -lt 1; then
  printf '{"status":"FAILED","exit_code":124}\\n' | atomic "$d/result.json"
  exit 0
fi
printf '{"status":"RUNNING"}\\n' | atomic "$d/status.json"
decoded=$(printf %s "$command" | base64 -d)
if timeout --signal=TERM --kill-after=10s "${seconds}s" sh -lc "$decoded" >"$d/job.log" 2>&1; then
  printf '{"status":"SUCCEEDED","exit_code":0}\\n' | atomic "$d/result.json"
else
  code=$?
  printf '{"status":"FAILED","exit_code":%s}\\n' "$code" | atomic "$d/result.json"
fi
"""


def _launcher_script() -> str:
    return """set -eu
d=$1; spec=$2; command=$3; deadline=$4
mkdir -p "$d"
if test -e "$d/result.json" || test -e "$d/job.json"; then
  exit 0
fi
if ! mkdir "$d/start-claim" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$d/start-claim" 2>/dev/null || true' EXIT
if test -e "$d/result.json" || test -e "$d/job.json"; then
  exit 0
fi
umask 077
printf %s "$spec" | base64 -d > "$d/job.json"
printf %s '""" + _encoded(_runner_script()) + """' | base64 -d > "$d/runner.sh"
nohup sh "$d/runner.sh" "$d" "$command" "$deadline" </dev/null >"$d/launcher.log" 2>&1 &
printf '%s\\n' "$!" > "$d/pid"
"""


class DetachedRemoteJob:
    """One immutable sequential argv job with transport-independent recovery."""

    def __init__(
        self,
        *,
        ssh_argv: list[str],
        remote_dir: str,
        local_state: Path,
        job_id: str,
        deadline_unix: float,
        stages: list[list[str]],
        cwd: str,
        env: dict[str, str] | None = None,
    ):
        if not ssh_argv or any(not isinstance(item, str) or not item for item in ssh_argv):
            raise ValueError("remote job SSH argv is invalid")
        if (
            not remote_dir.startswith("/")
            or not cwd.startswith("/")
            or not job_id
            or not stages
            or any(
                not stage or any(not isinstance(item, str) or not item for item in stage)
                for stage in stages
            )
            or any(
                not isinstance(key, str)
                or not key
                or "=" in key
                or not isinstance(value, str)
                for key, value in (env or {}).items()
            )
        ):
            raise ValueError("remote job identity, environment or argv stages are invalid")
        if (
            isinstance(deadline_unix, bool)
            or not isinstance(deadline_unix, (int, float))
            or not math.isfinite(deadline_unix)
            or deadline_unix <= 0
        ):
            raise ValueError("remote job deadline is invalid")
        self.ssh_argv = ssh_argv
        self.remote_dir = remote_dir.rstrip("/")
        self.local_state = Path(local_state)
        self.job_id = job_id
        self.deadline_unix = deadline_unix
        self.stages, self.cwd, self.env = stages, cwd, dict(env or {})

    @property
    def command(self) -> str:
        assignments = [f"{key}={value}" for key, value in sorted(self.env.items())]
        environment = shlex.join(["env", *assignments])
        stages = " && ".join(f"{environment} {shlex.join(stage)}" for stage in self.stages)
        return f"cd {shlex.quote(self.cwd)} && {stages}"

    @property
    def intent_path(self) -> Path:
        return self.local_state / "remote-job-intent.json"

    @property
    def status_path(self) -> Path:
        return self.local_state / "remote-job-status.json"

    def _intent(self) -> dict[str, Any]:
        body = {
            "schema": "forge/detached-remote-job/v1",
            "job_id": self.job_id,
            "remote_dir": self.remote_dir,
            "deadline_unix": self.deadline_unix,
            "cwd": self.cwd,
            "env": self.env,
            "stages": self.stages,
            "command_hash": hash_canonical(
                "forge/detached-remote-command/v1", self.cwd, self.env, self.stages
            ),
        }
        return {**body, "intent_hash": hash_canonical("forge/detached-remote-job/v1", body)}

    def start_once(self) -> dict[str, Any]:
        self.local_state.mkdir(parents=True, exist_ok=True)
        intent = self._intent()
        lock_path = self.local_state / ".remote-job.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.intent_path.exists():
                existing = json.loads(self.intent_path.read_text())
                if existing != intent:
                    raise ValueError("local remote job intent differs from this request")
                return {"status": "ALREADY_INTENDED", "intent_hash": intent["intent_hash"]}
            if self.deadline_unix <= time.time():
                raise ValueError("remote job deadline has elapsed before launch")
            write_immutable_json(self.intent_path, intent)
            log = (self.local_state / "remote-job-launch.log").open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    _ssh_argv(
                        self.ssh_argv,
                        [
                            "sh",
                            "-lc",
                            _launcher_script(),
                            "sh",
                            self.remote_dir,
                            _encoded(json.dumps(intent, sort_keys=True, separators=(",", ":"))),
                            _encoded(self.command),
                            str(int(self.deadline_unix)),
                        ],
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            finally:
                log.close()
            status = {
                "status": "LAUNCHING",
                "local_pid": process.pid,
                "intent_hash": intent["intent_hash"],
            }
            atomic_json(self.status_path, status)
            return status

    def poll(self, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
        intent = self._intent()
        if not self.intent_path.exists():
            return self.start_once()
        if json.loads(self.intent_path.read_text()) != intent:
            raise ValueError("local remote job intent differs from this request")
        script = """test -f "$1/job.json" || exit 3
printf '{"job_b64":"'; base64 <"$1/job.json" | tr -d '\\n'; printf '"'
if test -f "$1/result.json"; then
  printf ',"result_b64":"'; base64 <"$1/result.json" | tr -d '\\n'; printf '"'
fi
printf '}\\n'
"""
        try:
            completed = subprocess.run(
                _ssh_argv(self.ssh_argv, ["sh", "-lc", script, "sh", self.remote_dir]),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            status = {"status": "UNKNOWN_TRANSPORT", "detail": type(error).__name__}
            atomic_json(self.status_path, status)
            return status
        if completed.returncode:
            # An intent without observable remote state is still unknown: retrying
            # could duplicate an already accepted command after a transport loss.
            status = {"status": "UNKNOWN_REMOTE", "ssh_exit": completed.returncode}
            atomic_json(self.status_path, status)
            return status
        try:
            payload = json.loads(completed.stdout)
            job = json.loads(base64.b64decode(payload["job_b64"], validate=True))
        except json.JSONDecodeError as error:
            raise RemoteJobError("remote job status is not JSON") from error
        except (KeyError, TypeError, ValueError) as error:
            raise RemoteJobError("remote job identity is invalid") from error
        if job != intent:
            raise RemoteJobError("remote job intent differs from this request")
        if "result_b64" not in payload:
            status = {"status": "RUNNING", "intent_hash": intent["intent_hash"]}
            atomic_json(self.status_path, status)
            return status
        try:
            result = json.loads(base64.b64decode(payload["result_b64"], validate=True))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RemoteJobError("remote job result is invalid") from error
        exit_code = result.get("exit_code") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("status") not in {"SUCCEEDED", "FAILED"}
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or (result["status"] == "SUCCEEDED" and exit_code != 0)
            or (result["status"] == "FAILED" and exit_code == 0)
        ):
            raise RemoteJobError("remote job result is invalid")
        atomic_json(self.status_path, result)
        if result["status"] == "FAILED":
            raise RemoteJobError(f"remote job failed with exit {result.get('exit_code')}")
        return result
