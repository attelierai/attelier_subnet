#!/usr/bin/env bash
# Remote worker only. The caller enforces the original create-time +900s deadline
# and runs the independent billing/closure supervisor before invoking this file.
set -euo pipefail
test "$(pwd -P)" = /workspace/forge
test -f subnet/configs/wan50-reference.json
for destination in .venv .metrics VBench models metric-cache bootstrap-events.jsonl bootstrap-assets.log cuda-host-probe.json; do
    test ! -e "$destination" && test ! -L "$destination"
done
timeout --signal=TERM --kill-after=2s 25s python3 subnet/scripts/evaluator_pilot_cuda_preflight.py --out cuda-host-probe.json \
  --expected-gpu-name "${PILOT_GPU_NAME:-NVIDIA GeForce RTX 5090}"
runtime_root=/opt/forge-pilot-runtime
test ! -e "$runtime_root" && test ! -L "$runtime_root"
# /workspace may be a shared filesystem. Package installation belongs on the
# disposable container disk; model weights and all evidence stay in /workspace.
case "$(findmnt -n -o FSTYPE --target /opt)" in
    overlay|ext4|xfs|btrfs) ;;
    *) echo 'Container runtime storage must be local; refusing package installation.' >&2; exit 2 ;;
esac
python3 - <<'PY'
import json
import shutil
from pathlib import Path
runtime = json.loads(Path('pilot-runtime.json').read_text())
if runtime.get('provider') == 'lium':
    if shutil.disk_usage('/workspace').free < 120_000_000_000:
        raise SystemExit('Require at least 120GB free local Lium workspace storage')
elif runtime.get('containerDiskInGb', 0) < 40 or runtime.get('volumeInGb', 0) < 80:
    raise SystemExit('Require at least 40GB container disk and 80GB pod volume')
if shutil.disk_usage('/opt').free < 32_000_000_000:
    raise SystemExit('Require at least 32GB free container disk for two environments and cache')
PY
mkdir -m 700 "$runtime_root"
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_NO_CACHE_DIR
export PIP_CONFIG_FILE=/dev/null PIP_CACHE_DIR="$runtime_root/pip-cache" HF_HUB_DISABLE_TELEMETRY=1
stage() {
    printf '{"stage":"%s","unix":%s,"elapsed_seconds":%s}\n' "$1" "$(date +%s)" "$SECONDS" >> bootstrap-events.jsonl
}
cleanup() {
    code=$?
    trap - EXIT
    # Only this shell's still-running background jobs, never a stale PID.
    for child in $(jobs -pr); do kill "$child" 2>/dev/null || true; done
    wait || true
    if (( code != 0 )); then stage INCOMPLETE; fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Invoke the physical environment path: imports through the /workspace symlink
# repeatedly traverse the provider shared filesystem. Keep the aliases for consumers.
stage GENERATION_ENV_START
python3 -m venv "$runtime_root/generation"
ln -s "$runtime_root/generation" .venv
"$runtime_root/generation/bin/python" -m pip install --no-compile --disable-pip-version-check torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
"$runtime_root/generation/bin/python" -m pip install --no-compile -e './subnet[gpu,vision]'
"$runtime_root/generation/bin/python" -m pip check
"$runtime_root/generation/bin/python" -m pip freeze > generation-freeze.txt
stage GENERATION_ENV_READY
# Overlap the independent verified downloads with metrics installation. Neither
# generation nor DINO may start until this child exits successfully.
"$runtime_root/generation/bin/python" - <<'PY' > bootstrap-assets.log 2>&1 &
import json
import time
from pathlib import Path
from forge.models.artifact_manifest import ModelManifest
from forge.models.artifacts import fetch_verified
from forge.models.metric_assets import provision
print(json.dumps({'stage': 'ASSETS_START', 'unix': time.time()}), flush=True)
base = ModelManifest.from_dict(json.loads(Path('subnet/configs/wan50-reference.json').read_text()))
fetch_verified(base, Path('models/wan50'))
print(json.dumps({'stage': 'MODEL_VERIFIED', 'unix': time.time()}), flush=True)
provision(json.loads(Path('subnet/configs/vbench-weights.json').read_text()), Path('metric-cache'))
print(json.dumps({'stage': 'ASSETS_VERIFIED', 'unix': time.time()}), flush=True)
PY
assets_pid=$!
stage METRICS_ENV_START
python3 -m venv "$runtime_root/metrics"
ln -s "$runtime_root/metrics" .metrics
"$runtime_root/metrics/bin/python" -m pip install --no-compile --disable-pip-version-check torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
"$runtime_root/metrics/bin/python" -m pip install --no-compile -r subnet/configs/vbench-requirements.txt
"$runtime_root/metrics/bin/python" -m pip check
"$runtime_root/metrics/bin/python" -m pip freeze > metrics-freeze.txt
"$runtime_root/metrics/bin/python" - <<'PY'
from pathlib import Path
import site
with (Path(site.getsitepackages()[0])/'forge-validator-source.pth').open('x') as stream:
    stream.write('/workspace/forge/subnet/src\n')
PY
stage METRICS_ENV_READY
git clone --no-checkout https://github.com/Vchitect/VBench.git VBench
git -C VBench checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
mkdir -p metric-cache/dino_model
git clone --no-checkout https://github.com/facebookresearch/dino.git metric-cache/dino_model/facebookresearch_dino_main
git -C metric-cache/dino_model/facebookresearch_dino_main checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a
wait "$assets_pid"
stage ASSETS_JOINED
"$runtime_root/metrics/bin/python" subnet/scripts/evaluator_pilot_dino_probe.py --out dino-cache-probe.json
stage COMPLETE
