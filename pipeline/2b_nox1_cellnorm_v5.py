"""2b — cellular-background NOX1 normalization (replaces the ECM-based offset).

The v4 offset aligned the per-sample NOX1 background using the median NOX1 in ECM
(extracellular) regions. The ECM background does not track the cellular background,
so this step aligns on the CELLULAR background instead, with the same additive
per-sample structure: the per-sample background is the median NOX1 of the
NOX1-negative reference cells (double-negative 'other'), aligned to the cohort median.

    bg_s = median(NOX1_mean | non-hepatocyte)_s
    offset_s = bg_s - median_over_samples(bg_s)
    NOX1_mean_norm      = clip(NOX1_mean      - offset_s, 0)
    NOX1_cyto_mean_norm = clip(NOX1_cyto_mean - offset_s, 0)

Overwrites NOX1_*_norm in each <sample>/segmentation_v5/cell_stats_v5_gated.csv,
rewrites the cohort parquet, and writes nox1_cell_offsets_v5.csv (audit trail;
the ECM refuse_norm_offsets.csv is left untouched for comparison).

Run:
  HD_DATA_ROOT=/path/to/HD \
    micromamba run -n cynif python 2b_nox1_cellnorm_v5.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("HD_DATA_ROOT", "."))
OUT_ROOT = DATA_ROOT / "_nox1_normalization"
SEG_DIR = "segmentation_v5"
GATED = "cell_stats_v5_gated.csv"
NOX1_COLS = ["NOX1_mean", "NOX1_cyto_mean"]


def cellular_background(df):
    """Per-sample NOX1 background = median NOX1_mean of the DOUBLE-NEGATIVE cells
    (celltype 'other' = ASGR1- AND CD45-).

    These are used as the NOX1-negative reference: not hepatocytes (which express
    NOX1) and not leukocytes (Kupffer/macrophages carry oxidative-burst NADPH-oxidase
    activity). Falls back to all non-hepatocytes if 'other' is too small (<200 cells)."""
    other = df["celltype"].to_numpy() == "other"
    vals = df.loc[other, "NOX1_mean"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if vals.size >= 200:
        return float(np.median(vals))
    nonhep = df.loc[df["celltype"] != "hepatocyte", "NOX1_mean"].to_numpy(float)
    nonhep = nonhep[np.isfinite(nonhep)]
    return float(np.median(nonhep)) if nonhep.size > 50 else float(np.nanmedian(df["NOX1_mean"]))


def main():
    samples = sorted(p.name for p in DATA_ROOT.glob("sample_*")
                     if (p / SEG_DIR / GATED).exists())
    bgs = {}
    frames = {}
    for s in samples:
        df = pd.read_csv(DATA_ROOT / s / SEG_DIR / GATED)
        frames[s] = df
        bgs[s] = cellular_background(df)
    ref = float(np.median(list(bgs.values())))
    print(f"cohort cellular-background reference (median of non-hepatocyte NOX1) = {ref:.1f}", flush=True)

    rows, parts = [], []
    for s in samples:
        df = frames[s]
        off = bgs[s] - ref
        for col in NOX1_COLS:
            if col in df:
                df[col + "_norm"] = (df[col] - off).clip(lower=0)
        df.to_csv(DATA_ROOT / s / SEG_DIR / GATED, index=False)
        parts.append(df)
        rows.append(dict(sample=s, patient=s.split("_")[1], background_cell=round(bgs[s], 1),
                         reference=round(ref, 1), offset=round(off, 1)))
        print(f"  {s}: cell-bg={bgs[s]:.0f}  offset={off:+.0f}  "
              f"hepMed_norm={df.loc[df.celltype=='hepatocyte','NOX1_mean_norm'].median():.0f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_ROOT / "nox1_cell_offsets_v5.csv", index=False)
    pd.concat(parts, ignore_index=True).to_parquet(OUT_ROOT / "cell_stats_v5_gated.parquet", index=False)
    print(f"\nwrote nox1_cell_offsets_v5.csv + refreshed cohort parquet ({len(samples)} samples)", flush=True)


if __name__ == "__main__":
    main()
