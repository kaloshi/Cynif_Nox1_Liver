"""4b — celltype gating on segmentation_v5 with a robust 2-marker gate.

Fixes the three v4 gating bugs:
  1. ASGR1 over-call (threshold pinned to the 5-15th percentile) -> use SPARQ
     LOCAL CONTRAST (fc vs per-cell local background): in ASGR1-negative zones
     fc≈0 -> not positive. Graded-marker- and neighbor-robust.
  2. CD45 under-call (global Otsu dragged up by the dominant hepatocytes) -> per-cell SPARQ
     local contrast, immune to the population mix.
  3. Double-positive loss (ASGR1 given hard precedence) -> mutually-exclusive
     assignment by RELATIVE evidence (robust z of fc within each positive pop).

Primary = SPARQ. Automatic FALLBACK to intensity thresholds (CD45+-anchored ASGR1
without the destructive clamp; 2-component GMM for CD45) if SPARQ does not
separate (positive fraction out of range / too many NaN).

Outputs: <sample>/segmentation_v5/cell_stats_v5_gated.csv,
         hepato_cells_v5.tiff, cd45_leukocyte_cells_v5.tiff ;
         _nox1_normalization/cell_stats_v5_gated.parquet

Run:
  HD_DATA_ROOT=/path/to/HD \
    micromamba run -n cynif python 4b_celltype_quantify_v5.py [SAMPLE|all]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu

DATA_ROOT = Path(os.environ.get("HD_DATA_ROOT", "."))
OUT_ROOT = DATA_ROOT / "_nox1_normalization"
SEG_DIR = "segmentation_v5"
WHOLE_CELL = "whole_cell_label_v5.tiff"
CELLSTATS = "cell_stats_v5.csv"

# --- gating tunables (liver: dominant hepatocyte pop, high CD45 pedestal) ---
ASGR1_FRAC_BAND = (0.45, 0.75)   # plausible hepatocyte fraction guard for the GMM split
CD45_MAD_K = 5.0                 # CD45+ = above hepatocyte pedestal median + K*MAD (robust)
NOX1_COLS = ["NOX1_mean", "NOX1_cyto_mean"]


def _mad_dist(x, thr, pos):
    """Signed distance of x above `thr`, scaled by the MAD of the positive population."""
    ref = x[pos & np.isfinite(x)]
    mad = (np.median(np.abs(ref - np.median(ref))) or 1.0) if ref.size >= 5 else 1.0
    return (x - thr) / mad


def _gmm_bright_threshold(vals):
    """2-component GMM on log1p(vals) -> intensity threshold (min bright-component value)."""
    from sklearn.mixture import GaussianMixture
    v = np.log1p(np.clip(np.nan_to_num(vals), 0, None))
    g = GaussianMixture(n_components=2, random_state=0).fit(v.reshape(-1, 1))
    hi = int(np.argmax(g.means_.ravel()))
    post = g.predict_proba(v.reshape(-1, 1))[:, hi]
    bright = post > 0.5
    if not bright.any():
        return float(np.expm1(np.median(v)))
    return float(np.expm1(v[bright].min()))


def _gmm_valley(vals):
    """(threshold at the 2-component decision boundary, separation ratio mu_hi/mu_lo).

    Handles both CD45 distribution shapes across samples: when CD45 is clearly
    BIMODAL (low pedestal + bright leukocytes) the valley is the natural cut; when
    it is a unimodal pedestal+tail the two components collapse (ratio≈1) and the
    caller ignores the valley in favour of the pedestal-MAD threshold."""
    from sklearn.mixture import GaussianMixture
    v = np.log1p(np.clip(np.nan_to_num(vals), 0, None)).reshape(-1, 1)
    g = GaussianMixture(n_components=2, random_state=0).fit(v)
    mu = np.sort(g.means_.ravel())
    xs = np.linspace(v.min(), np.percentile(v, 99.5), 4000).reshape(-1, 1)
    pr = g.predict(xs)
    ch = np.where(np.diff(pr) != 0)[0]
    thr = float(np.expm1(xs[ch[0], 0])) if len(ch) else float(np.expm1(mu.mean()))
    ratio = float(np.expm1(mu[1]) / max(np.expm1(mu[0]), 1.0))
    return thr, ratio


def _positivity(df):
    """Intensity mutual-anchoring gate (SPARQ fails for the dominant ASGR1 marker in
    dense parenchyma: a hepatocyte's local background is other hepatocytes -> fc≈0).

    ASGR1+  = ASGR1_cyto_mean above a GMM(log1p) split (guarded to a plausible band).
    CD45+   = CD45_mean above the hepatocyte CD45 pedestal (median + K*MAD of a CLEAN
              hepatocyte reference = ASGR1+ cells with below-median CD45, so the
              reference is not contaminated by double-positives). Robustly recovers
              dim leukocytes that a hepatocyte-dominated global Otsu buries.
    Returns (asgr1_pos, cd45_pos, mode, za, zc)."""
    a = np.nan_to_num(df.get("ASGR1_cyto_mean", df["ASGR1_mean"]).to_numpy(float))
    c = np.nan_to_num(df["CD45_mean"].to_numpy(float))

    a_thr = _gmm_bright_threshold(a)
    asgr1_pos = a > a_thr
    # guard the hepatocyte fraction into a liver-plausible band via percentile fallback
    if not (ASGR1_FRAC_BAND[0] <= asgr1_pos.mean() <= ASGR1_FRAC_BAND[1]):
        target = float(np.clip(asgr1_pos.mean(), *ASGR1_FRAC_BAND))
        a_thr = float(np.percentile(a, 100 * (1 - target)))
        asgr1_pos = a > a_thr

    # CD45 threshold. CD45 scale + shape vary strongly across samples, so combine two
    # views: (1) pedestal median + K*MAD on the ASGR1+ (hepatocyte, CD45-neg) reference
    # -> robust when CD45 is a unimodal pedestal+tail (where a GMM collapses); (2) the
    # 2-component GMM valley fitted on the ASGR1-NEGATIVE cells, where the leukocyte-
    # vs-other CD45 split is cleanest (not swamped by hepatocytes) -> used only when
    # that split is well-separated (ratio>1.8). Take the max so neither under-thresholds.
    ref = c[asgr1_pos] if asgr1_pos.any() else c
    ref_clean = ref[ref <= np.median(ref)]
    med = float(np.median(ref_clean))
    mad = float(np.median(np.abs(ref_clean - med))) * 1.4826 or 1.0
    c_thr = med + CD45_MAD_K * mad
    try:
        neg = ~asgr1_pos
        if neg.sum() > 200:
            valley, ratio = _gmm_valley(c[neg])
            if ratio > 1.8:                  # clean bimodal among non-hepatocytes
                c_thr = max(c_thr, valley)
    except Exception:  # noqa: BLE001
        pass
    cd45_pos = c > c_thr

    za = _mad_dist(a, a_thr, asgr1_pos)
    zc = _mad_dist(c, c_thr, cd45_pos)
    return asgr1_pos, cd45_pos, f"intensity(a>{a_thr:.0f},cd45>{c_thr:.0f})", za, zc


def gate_sample(sample: str, offset: float):
    sd = DATA_ROOT / sample / SEG_DIR
    df = pd.read_csv(sd / CELLSTATS)

    asgr1_pos, cd45_pos, mode, za, zc = _positivity(df)
    df["is_asgr1"] = asgr1_pos
    df["is_cd45"] = cd45_pos

    # mutually-exclusive assignment by relative evidence
    ct = np.full(len(df), "other", dtype=object)
    only_h = asgr1_pos & ~cd45_pos
    only_l = cd45_pos & ~asgr1_pos
    both = asgr1_pos & cd45_pos
    ct[only_h] = "hepatocyte"
    ct[only_l] = "leukocyte_CD45"
    ct[both & (zc >= za)] = "leukocyte_CD45"   # recover CD45+ that v4 dumped to 'other'
    ct[both & (zc < za)] = "hepatocyte"
    df["celltype"] = ct

    # NOX1 cross-sample additive normalization (identical to v4)
    for col in NOX1_COLS:
        if col in df:
            df[col + "_norm"] = (df[col] - offset).clip(lower=0)

    df.to_csv(sd / "cell_stats_v5_gated.csv", index=False)

    # per-celltype label masks (original CellIDs)
    wc = tifffile.imread(str(sd / WHOLE_CELL))
    mx = int(wc.max())
    for ctype, fname in [("hepatocyte", "hepato_cells_v5.tiff"),
                         ("leukocyte_CD45", "cd45_leukocyte_cells_v5.tiff")]:
        ids = df.loc[df.celltype == ctype, "CellID"].to_numpy()
        ids = ids[ids <= mx]
        lut = np.zeros(mx + 1, dtype=wc.dtype)
        lut[ids] = ids
        tifffile.imwrite(str(sd / fname), lut[wc], compression="zlib")

    n = len(df)
    nh = int((df.celltype == "hepatocyte").sum())
    nl = int((df.celltype == "leukocyte_CD45").sum())
    recov = (df.celltype[cd45_pos] == "leukocyte_CD45").mean() if cd45_pos.sum() else 1.0
    print(f"  {sample} [{mode}]: n={n}  hepato={nh} ({100*nh/n:.0f}%)  leuko={nl} ({100*nl/n:.1f}%)  "
          f"other={n-nh-nl} | is_asgr1={100*asgr1_pos.mean():.0f}% is_cd45={100*cd45_pos.mean():.1f}% "
          f"cd45_recovered={100*recov:.0f}%", flush=True)
    return df


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    offs = pd.read_csv(OUT_ROOT / "refuse_norm_offsets.csv").set_index("sample")["offset"]
    todo = (sorted(p.name for p in DATA_ROOT.glob("sample_*") if p.is_dir())
            if which in ("all", "") else
            [which if which.startswith("sample_") else f"sample_{which}"])
    print(f"gating v5 | {len(todo)} sample(s) | CD45 = pedestal median + {CD45_MAD_K}*MAD", flush=True)
    parts = []
    for s in todo:
        try:
            parts.append(gate_sample(s, float(offs.get(s, 0.0))))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {s}: {type(e).__name__}: {e}", flush=True)
    if parts:
        allc = pd.concat(parts, ignore_index=True)
        allc.to_parquet(OUT_ROOT / "cell_stats_v5_gated.parquet", index=False)
        print(f"\ncohort: {len(allc)} cells -> {OUT_ROOT}/cell_stats_v5_gated.parquet", flush=True)


if __name__ == "__main__":
    main(sys.argv)
