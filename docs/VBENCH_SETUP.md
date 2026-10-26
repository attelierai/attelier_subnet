# Entorno de métricas del validator

Estas versiones son las provisionadas en la RTX 5090 del ensayo de desarrollo.
El entorno pasó `pip check`; esto no cualifica científicamente el evaluador.
Instálalo en el worker autorizado, antes de cronometrar inferencia. No instales
Forge con sus dependencias en este entorno: Wan necesita otra versión de
Transformers y el paquete principal declara NumPy 2; VBench aquí usa NumPy 1.26.

Desde el clon público en `/workspace/forge/subnet`, con Python 3.12:

```bash
test ! -e /workspace/forge/metrics-env
python3 -m venv /workspace/forge/metrics-env
/workspace/forge/metrics-env/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
/workspace/forge/metrics-env/bin/python -m pip install -r configs/vbench-requirements.txt
/workspace/forge/metrics-env/bin/python -m pip check
/workspace/forge/metrics-env/bin/python - <<'PY_SETUP'
from pathlib import Path
import site
source = Path("/workspace/forge/subnet/src")
assert (source / "forge/evaluation/vbench_worker.py").is_file()
path = Path(site.getsitepackages()[0]) / "forge-validator-source.pth"
with path.open("x") as handle:
    handle.write(str(source) + "\n")
PY_SETUP
/workspace/forge/metrics-env/bin/python -m forge.evaluation.vbench_worker --help
/workspace/forge/metrics-env/bin/python -m pip freeze > /workspace/forge/metrics-freeze.txt

git clone --no-checkout https://github.com/Vchitect/VBench.git /workspace/forge/VBench
git -C /workspace/forge/VBench checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
mkdir -p /workspace/forge/metric-cache/dino_model
git clone --no-checkout https://github.com/facebookresearch/dino.git \
  /workspace/forge/metric-cache/dino_model/facebookresearch_dino_main
git -C /workspace/forge/metric-cache/dino_model/facebookresearch_dino_main \
  checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a

.venv/bin/python -m forge.models.metric_assets \
  --manifest configs/vbench-weights.json --cache /workspace/forge/metric-cache
```

Ejecuta cada paso sólo si el anterior termina correctamente. Las rutas deben
ser nuevas, salvo el cache de pesos: los archivos completos se verifican de
nuevo; un `.partial` queda conservado y requiere revisar el fallo antes de
reintentar. El descargador no alquila GPU, ejecuta modelos ni carga los `.pth`.
El ZIP RAFT se verifica antes de extraer únicamente el miembro declarado.
Los hashes fijan los bytes aunque una URL upstream use `main`.

En `assigned-worker.json`, configura `vbench.checkout`, `vbench.cache` y
`vbench.interpreter` con esas rutas. `weight_sha256` tiene estas claves:

| Clave | Archivo dentro del cache |
|---|---|
| `subject_consistency` | `dino_model/dino_vitbase16_pretrain.pth` |
| `background_consistency` | `clip_model/ViT-B-32.pt` |
| `imaging_quality` | `pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` |
| `motion_smoothness` | `amt_model/amt-s.pth` |
| `dynamic_degree` | `raft_model/models/raft-things.pth` |

Copia cada SHA-256 desde `configs/vbench-weights.json`. `dino_revision` es el
commit anterior. `runner_image_digest` debe identificar la imagen real del
worker; no uses un digest de ejemplo. Conserva el freeze de dependencias con
los recibos. El archivo `.pth` anterior permite importar el código de Forge
sin instalar sus dependencias en el entorno de métricas. El intérprete debe
ser el ejecutable del entorno, sin resolver su symlink al
Python del sistema. Ejecuta un ensayo real de métricas antes de admitir trabajo
competitivo: instalar correctamente no prueba que todos los kernels funcionen
en otro hardware o driver.

El runner vincula también el caché Torch Hub de DINO al checkpoint verificado.
El modo local de VBench por sí solo no evita que el `hubconf` de DINO consulte
el caché global o descargue otra copia; un checkpoint distinto se rechaza.

La normalización y el coseno de las características CLIP se calculan en FP32,
conservando las características producidas por el encoder. Un control congelado
produjo 1.000128173828125 en FP16 y 0.9999122977256775 con esas mismas
características en FP32. No se amplía la tolerancia de frontera de 1e-6.
Cada recibo conserva el SHA-256 del runner, la política de precisión y el hash
del JSON nativo; ambos archivos deben archivarse. Este cambio invalida una
calibración ligada a otra implementación, aunque los pesos sigan siendo iguales.
