#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from forge.inference.transition_diag import residual

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
r = residual(a.manifest)
a.out.write_text(json.dumps(r, indent=2) + "\n")
print(r["status"])
