"""CPU-only Res Multistep callback-transition diagnostic (eta=0, no CFG++)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from forge.inference.trace_eval import _tensor


def _load(root, d):
    return _tensor(root, d)


def _parts(root, values):
    if len({x["component"] for x in values}) != len(values):
        raise ValueError("duplicate_trace_component")
    return {x["component"]: _load(root, x) for x in values}


def residual(manifest: Path):
    """Return max relative RMS of nonterminal advertised Res Multistep transitions."""
    m = json.loads(manifest.read_text())
    root = manifest.parent
    if (
        m.get("sampler") not in (None, "res_multistep")
        or m.get("eta", 0) != 0
        or m.get("cfg_pp", False)
    ):
        raise ValueError("profile_not_res_multistep_eta0")
    rows = m["trace"]
    sig = [np.float32(x["sigma"]) for x in rows]
    if len(rows) < 2 or any(not np.isfinite(s) or s <= 0 for s in sig):
        raise ValueError("insufficient_or_invalid_sigmas")
    if any(a <= b for a, b in zip(sig, sig[1:], strict=False)):
        raise ValueError("sigmas_must_decrease")
    out = []
    for i in range(len(rows) - 1):
        xi = _parts(root, rows[i]["stateCallback"])
        di = _parts(root, rows[i]["denoised"])
        nexts = _parts(root, rows[i + 1]["stateCallback"])
        if set(xi) != set(di) or set(xi) != set(nexts):
            raise ValueError("trace_components_changed")
        if sig[i + 1] == 0:
            continue
        pred = {}
        for key in xi:
            if i == 0:
                pred[key] = xi[key] + (xi[key] - di[key]) / sig[i] * (sig[i + 1] - sig[i])
            else:
                old = _parts(root, rows[i - 1]["denoised"])[key]
                t = -math.log(float(sig[i]))
                h = -math.log(float(sig[i + 1])) - t
                c2 = (-math.log(float(sig[i - 1])) - t) / h
                phi1 = math.expm1(-h) / (-h)
                phi2 = (phi1 - 1) / (-h)
                b1 = phi1 - phi2 / c2
                b2 = phi2 / c2
                pred[key] = math.exp(-h) * xi[key] + h * (b1 * di[key] + b2 * old)
            e = pred[key] - nexts[key]
            out.append(
                float(
                    np.sqrt(np.mean(e * e)) / max(np.sqrt(np.mean(nexts[key] * nexts[key])), 1e-12)
                )
            )
    return {
        "status": "MEASURED",
        "maxRelativeRms": max(out) if out else None,
        "transitions": len(out),
        "claim": "callback_transition_diagnostic_only",
        "limits": ["no_raw_forward_binding", "no_initial_or_final_output_binding"],
    }
