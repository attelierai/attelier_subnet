#!/usr/bin/env bash
# Recovery of metrics only; never installs or runs the generation model.
set -euo pipefail
test "$(pwd -P)" = /workspace/forge
test ! -e /opt/forge-pilot-runtime
test ! -e .metrics
test ! -e .venv
export PYTHONDONTWRITEBYTECODE=1
stage() { printf '{"stage":"%s","elapsed_seconds":%s}\n' "$1" "$SECONDS" >> bootstrap-events.jsonl; }
stage RECOVERY_METRICS_START
python3 subnet/scripts/evaluator_pilot_cuda_preflight.py --out cuda-host-probe.json \
  --expected-gpu-name "${PILOT_GPU_NAME:-NVIDIA GeForce RTX 5090}"
mkdir -m 700 /opt/forge-pilot-runtime
export PIP_CONFIG_FILE=/dev/null PIP_CACHE_DIR=/opt/forge-pilot-runtime/pip-cache
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_NO_CACHE_DIR
python3 -m venv /opt/forge-pilot-runtime/orchestration
ln -s /opt/forge-pilot-runtime/orchestration .venv
/opt/forge-pilot-runtime/orchestration/bin/python -m pip install --no-compile -e './subnet[vision]'
python3 -m venv /opt/forge-pilot-runtime/metrics
ln -s /opt/forge-pilot-runtime/metrics .metrics
/opt/forge-pilot-runtime/metrics/bin/python -m pip install --no-compile torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
/opt/forge-pilot-runtime/metrics/bin/python -m pip install --no-compile -r subnet/configs/vbench-requirements.txt
/opt/forge-pilot-runtime/metrics/bin/python -m pip check
/opt/forge-pilot-runtime/metrics/bin/python -m pip freeze > metrics-freeze.txt
/opt/forge-pilot-runtime/metrics/bin/python - <<'PY'
import site
from pathlib import Path
with (Path(site.getsitepackages()[0])/'forge-validator-source.pth').open('x') as stream:
    stream.write('/workspace/forge/subnet/src\n')
PY
git clone --no-checkout https://github.com/Vchitect/VBench.git VBench
git -C VBench checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
mkdir -p metric-cache/dino_model
git clone --no-checkout https://github.com/facebookresearch/dino.git metric-cache/dino_model/facebookresearch_dino_main
git -C metric-cache/dino_model/facebookresearch_dino_main checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a
/opt/forge-pilot-runtime/metrics/bin/python - <<'PY' > bootstrap-assets.log
import json
from pathlib import Path
from forge.models.metric_assets import provision
result=provision(json.loads(Path('subnet/configs/vbench-weights.json').read_text()),Path('metric-cache'))
print(json.dumps({'status':'VERIFIED','asset_count':len(result)}))
PY
/opt/forge-pilot-runtime/metrics/bin/python subnet/scripts/evaluator_pilot_dino_probe.py --out dino-cache-probe.json

stage COMPLETE
