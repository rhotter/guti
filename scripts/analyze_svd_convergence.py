#!/usr/bin/env python3
"""Locate converged ("optimal") geometry parameters from saved SVD sweeps.

For each modality and each swept parameter we auto-discover the sweep (the
largest set of variants that differ *only* in that parameter), then report how
the singular-value spectrum changes as the parameter increases. The spectrum
fully determines the achievable channel capacity, so the parameter is
"converged" once further refinement stops changing the spectrum.

Metrics per variant (all scale-invariant, so they are directly comparable
across a grid-resolution sweep where the absolute sigma scale changes with the
voxel volume):

  rank@1e-3 / 1e-4 : effective rank = #{ sigma_i / sigma_0 >= tau }
  Crel             : capacity proxy  sum_i log2(1 + (sigma_i / (sigma_0 * eta))^2)
                     with eta = 1e-4 (noise pinned to 1e-4 of the top mode)

Convergence is flagged at the smallest parameter value whose step *to the next*
value changes Crel by < REL_TOL and rank@1e-4 by < RANK_TOL (and stays small).

Usage:  python scripts/analyze_svd_convergence.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from guti.parameters import Parameters  # noqa: E402

VARIANTS = REPO_ROOT / "results" / "variants"

ETA = 1e-4          # capacity-proxy noise, relative to sigma_0
REL_TOL = 0.05      # <5% change in Crel == converged
RANK_TOL = 0.03     # <3% change in effective rank == converged

# Which params define the sweep axes for each modality's variant directory.
MODALITIES = {
    "cw_fnirs": ("num_sensors", "grid_resolution_mm", "max_dist"),
    "td_fnirs": ("num_sensors", "grid_resolution_mm", "max_dist", "n_time_gates"),
    "eeg_openmeeg": (
        "num_sensors",
        "grid_resolution_mm",
        "n_radial_lines",
        "source_spacing_mm",
    ),
}


def load_variants(dirname):
    out = []
    d = VARIANTS / dirname
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.npz")):
        try:
            data = np.load(f, allow_pickle=True)
            if "singular_values" not in data.files:
                continue
            s = np.asarray(data["singular_values"], dtype=np.float64)
            s = s[np.isfinite(s) & (s > 0)]
            if s.size == 0:
                continue
            s = np.sort(s)[::-1]
            pd = data["parameters"].item() if "parameters" in data.files else {}
            out.append((f.name, s, Parameters.from_dict(pd)))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f.name}: {e}")
    return out


def metrics(s):
    s0 = s[0]
    r = s / s0
    rank3 = int(np.count_nonzero(r >= 1e-3))
    rank4 = int(np.count_nonzero(r >= 1e-4))
    crel = float(np.sum(np.log2(1.0 + (r / ETA) ** 2)))
    return s0, rank3, rank4, crel


def analyze(dirname, axes):
    variants = load_variants(dirname)
    print(f"\n{'='*92}\n{dirname}: {len(variants)} variants\n{'='*92}")
    if not variants:
        print("  (no data)")
        return

    for axis in axes:
        # Group by the tuple of all OTHER axis params; within a group the only
        # thing that varies is `axis`. Print every group that forms a real sweep
        # (>= 4 distinct values), largest first, so overlapping coarse/fine
        # sweeps are both visible.
        groups = defaultdict(list)
        for name, s, p in variants:
            if getattr(p, axis, None) is None:
                continue
            key = tuple((a, getattr(p, a, None)) for a in axes if a != axis)
            groups[key].append((name, s, p))

        sweeps = []
        for key, rows in groups.items():
            if len({getattr(p, axis) for _, _, p in rows}) >= 4:
                sweeps.append((key, rows))
        sweeps.sort(key=lambda kr: -len({getattr(p, axis) for _, _, p in kr[1]}))

        for best_key, best_rows in sweeps:
            analyze_sweep(axis, best_key, best_rows)


def analyze_sweep(axis, best_key, best_rows):
        # dedupe by axis value (prefer the spectrum with more singular values)
        by_val = {}
        for name, s, p in best_rows:
            v = getattr(p, axis)
            if v not in by_val or s.size > by_val[v][1].size:
                by_val[v] = (name, s, p)
        items = sorted(by_val.items())

        pins = ", ".join(f"{a}={v}" for a, v in best_key if v is not None)
        print(f"\n  -- sweep over {axis}   (pinned: {pins}) --")
        print(f"    {'value':>10} {'matrix':>16} {'sigma0':>11} "
              f"{'rank@1e-3':>9} {'rank@1e-4':>9} {'Crel':>10} {'dCrel':>7} {'drank4':>7}")
        prev = None
        conv_val = None
        rows_print = []
        for v, (name, s, p) in items:
            s0, r3, r4, crel = metrics(s)
            ms = p.matrix_size
            mss = f"{ms[0]}x{ms[1]}" if ms else f"?x{s.size}"
            dC = dR = None
            if prev is not None:
                pc, pr4 = prev
                dC = abs(crel - pc) / max(pc, 1e-9)
                dR = abs(r4 - pr4) / max(pr4, 1)
            rows_print.append((v, mss, s0, r3, r4, crel, dC, dR))
            prev = (crel, r4)
        # find first value whose forward step is below both tolerances
        for i in range(len(rows_print) - 1):
            nxtC, nxtR = rows_print[i + 1][6], rows_print[i + 1][7]
            if nxtC is not None and nxtC < REL_TOL and nxtR < RANK_TOL:
                conv_val = rows_print[i][0]
                break
        for (v, mss, s0, r3, r4, crel, dC, dR) in rows_print:
            mark = "  <-- converged" if v == conv_val else ""
            dCs = f"{dC*100:5.1f}%" if dC is not None else "   -- "
            dRs = f"{dR*100:5.1f}%" if dR is not None else "   -- "
            print(f"    {v:>10} {mss:>16} {s0:>11.3e} {r3:>9} {r4:>9} "
                  f"{crel:>10.1f} {dCs:>7} {dRs:>7}{mark}")
        if conv_val is not None:
            print(f"    => CONVERGED at {axis} = {conv_val} "
                  f"(further increase changes Crel <{REL_TOL*100:.0f}% & rank <{RANK_TOL*100:.0f}%)")
        else:
            print(f"    => NOT converged within swept range (still changing at the top)")


if __name__ == "__main__":
    for dirname, axes in MODALITIES.items():
        analyze(dirname, axes)
