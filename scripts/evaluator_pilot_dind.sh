#!/usr/bin/env bash
# Run only on the owned disposable Sysbox worker under an absolute outer timeout.
set -euo pipefail
: "${PILOT_INNER_NAME:?owned inner container name required}"
: "${PILOT_IMAGE:?digest-pinned image required}"
test "$(id -u)" = 0
test -d /workspace/forge
# Never use or replace another daemon or its socket/data.
test ! -e /run/forge-pilot-docker.sock
# Sysbox supports inner Docker only at its managed /var/lib/docker mount.
# Refuse existing daemon state rather than repurposing it.
test ! -S /var/run/docker.sock
if test -d /var/lib/docker; then
    test -z "$(find /var/lib/docker -mindepth 1 -maxdepth 1 -print -quit)"
fi
command -v dockerd >/dev/null || {
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends docker.io
}
mkdir -p /var/lib/docker
nohup dockerd --host unix:///run/forge-pilot-docker.sock \
    --data-root /var/lib/docker --exec-root /run/forge-pilot-docker \
    --pidfile /run/forge-pilot-docker.pid --bridge none --iptables=false \
    > /workspace/forge/dockerd.log 2>&1 < /dev/null &
daemon_pid=$!
for attempt in $(seq 1 30); do
    if docker --host unix:///run/forge-pilot-docker.sock info >/dev/null 2>&1; then break; fi
    kill -0 "$daemon_pid"
    sleep 1
done
docker --host unix:///run/forge-pilot-docker.sock info >/dev/null
docker --host unix:///run/forge-pilot-docker.sock pull "$PILOT_IMAGE"
python3 - <<'PY'
import glob
import os
import pathlib
import re
import shutil
import stat
import subprocess

name = os.environ['PILOT_INNER_NAME']
image = os.environ['PILOT_IMAGE']
assert re.fullmatch(r'forge-pilot-[a-z0-9-]+', name)
assert re.fullmatch(r'runpod/base@sha256:[0-9a-f]{64}', image)
command = ['docker', '--host', 'unix:///run/forge-pilot-docker.sock', 'run',
           '--pull=never', '-d', '--name', name, '--network', 'host', '--shm-size', '8g',
           '--mount', 'type=bind,source=/workspace/forge,target=/workspace/forge',
           '--entrypoint', 'sleep']
# Expose only the assigned GPU devices and driver libraries already visible inside
# this rented pod. No host Docker socket or broad host library-directory mount.
devices = sorted(p for p in glob.glob('/dev/nvidia*') if stat.S_ISCHR(os.stat(p).st_mode))
assert '/dev/nvidia0' in devices and '/dev/nvidiactl' in devices
for path in devices:
    command += ['--device', path]
mounts = {}
for line in subprocess.check_output(['ldconfig', '-p'], text=True).splitlines():
    fields = line.split()
    if len(fields) < 4 or not fields[0].startswith(('libcuda.', 'libnvidia-')):
        continue
    source = pathlib.Path(fields[-1]).resolve(strict=True)
    mounts[str(source)] = str(source)
    mounts[str(source.parent / fields[0])] = str(source)
smi = str(pathlib.Path(shutil.which('nvidia-smi')).resolve(strict=True))
mounts['/usr/bin/nvidia-smi'] = smi
assert any(path.endswith('/libcuda.so.1') for path in mounts)
for target, source in sorted(mounts.items()):
    command += ['--mount', f'type=bind,source={source},target={target},readonly']
command += [image, 'infinity']
subprocess.run(command, check=True)
PY
