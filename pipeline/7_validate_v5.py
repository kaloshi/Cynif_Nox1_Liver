"""7 — self-validation of segmentation_v5 + gating: visual QC crops + metrics.

napari offscreen renders blank in this env, so QC images are composed with
numpy + skimage.find_boundaries + PIL.

Per sample writes to <sample>/segmentation_v5/qc/:
  * PNG crops: ASGR1 grayscale + hepatocyte boundaries (yellow);
               CD45  grayscale + leukocyte  boundaries (cyan).
  * metrics.csv row (also printed) with the pass-criteria numbers.

Run:
  HD_DATA_ROOT=/path/to/HD \
    micromamba run -n cynif python 7_validate_v5.py [SAMPLE|all]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

DATA_ROOT = Path(os.environ.get("HD_DATA_ROOT", "."))
PIXEL_SIZE_UM = 0.26
SEG_DIR = "segmentation_v5"
CH = {"DAPI": 0, "AF1": 1, "AF2": 2, "ASGR1": 3, "CD45": 4, "NOX1": 5}
IMG_CANDIDATES = [
    "AF_removal/fused_decon_refused.ome.tif",
    "AF_removal/fused_decon_AF_cleaned_nnls_v3_NOX1norm_tileflat.ome.tif",
    "AF_removal/fused_decon_AF_cleaned_nnls_v3.ome.tif",
]
CROP = 1400
LOG2_12 = float(np.log2(1.2))


def image_path(sample):
    d = DATA_ROOT / sample
    for c in IMG_CANDIDATES:
        if (d / c).exists():
            return d / c
    raise FileNotFoundError(d)


def _gray(a, p=99.5):
    a = a.astype("float32")
    nz = a[a > 0]
    hi = np.percentile(nz, p) if nz.size > 50 else (a.max() or 1)
    return np.clip(a / max(hi, 1e-6), 0, 1)


def _panel(chan, mask, color):
    """grayscale channel (RGB) with instance-mask boundaries overlaid in `color`."""
    g = _gray(chan)
    rgb = np.stack([g, g, g], -1)
    b = find_boundaries(mask, mode="inner")
    for k in range(3):
        rgb[..., k][b] = color[k]
    return (np.clip(rgb, 0, 1) * 255).astype("uint8")


def _pick_crops(hep, cd, wc):
    """3 crops: hepatocyte-dense, leukocyte-containing, ASGR1-neg (cells but few hepato)."""
    H, W = wc.shape
    r = CROP // 2
    step = 350
    best_h = best_l = best_neg = None
    for cy in range(r, H - r, step):
        for cx in range(r, W - r, step):
            sl = (slice(cy - r, cy + r), slice(cx - r, cx + r))
            nh = np.count_nonzero(np.unique(hep[sl]))
            nl = np.count_nonzero(np.unique(cd[sl]))
            ncell = np.count_nonzero(np.unique(wc[sl]))
            if ncell < 30:
                continue
            hepfrac = nh / max(ncell, 1)
            if best_h is None or nh > best_h[0]:
                best_h = (nh, cy, cx)
            if best_l is None or nl > best_l[0]:
                best_l = (nl, cy, cx)
            # ASGR1-neg region: many cells, few hepatocytes
            score = ncell * (1 - hepfrac)
            if (best_neg is None or score > best_neg[0]) and hepfrac < 0.4:
                best_neg = (score, cy, cx)
    picks = {"hepatodense": best_h, "immune": best_l, "asgr1_neg": best_neg}
    return {k: (v[1], v[2]) for k, v in picks.items() if v is not None}


def validate(sample: str):
    sd = DATA_ROOT / sample / SEG_DIR
    qc = sd / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sd / "cell_stats_v5_gated.csv")
    img = image_path(sample)
    asgr1 = tifffile.imread(str(img), key=CH["ASGR1"])
    cd45 = tifffile.imread(str(img), key=CH["CD45"])
    hep = tifffile.imread(str(sd / "hepato_cells_v5.tiff"))
    cd = tifffile.imread(str(sd / "cd45_leukocyte_cells_v5.tiff"))
    wc = tifffile.imread(str(sd / "whole_cell_label_v5.tiff"))
    nuc = tifffile.imread(str(sd / "nuclei_v5.tiff"))

    crops = _pick_crops(hep, cd, wc)
    for name, (cy, cx) in crops.items():
        r = CROP // 2
        sl = (slice(cy - r, cy + r), slice(cx - r, cx + r))
        left = _panel(asgr1[sl], hep[sl], (1.0, 0.95, 0.0))   # ASGR1 + hepato (yellow)
        right = _panel(cd45[sl], cd[sl], (0.0, 0.9, 1.0))     # CD45 + leuko (cyan)
        combo = np.concatenate([left, np.full((CROP, 6, 3), 255, "uint8"), right], axis=1)
        Image.fromarray(combo).save(str(qc / f"{sample}_{name}.png"))

    # ---- metrics ----
    n = len(df)
    is_hep = (df.celltype == "hepatocyte").to_numpy()
    is_leu = (df.celltype == "leukocyte_CD45").to_numpy()
    diam = 2 * np.sqrt(df.Area.to_numpy(float) / np.pi) * PIXEL_SIZE_UM
    nuc_areas = np.bincount(nuc.ravel())[1:]
    nuc_diam_px = 2 * np.sqrt(np.median(nuc_areas) / np.pi) if len(nuc_areas) else 0
    # ASGR1-low hepatocytes measured ABSOLUTELY: fraction of hepatocytes whose ASGR1
    # cytoplasm intensity falls within the ASGR1-negative (leukocyte) population
    # (<= p90 of leukocyte ASGR1). fc is useless here (hepatocytes surround hepatocytes).
    acol = "ASGR1_cyto_mean" if "ASGR1_cyto_mean" in df else "ASGR1_mean"
    a_all = df[acol].to_numpy(float)
    neg_ref = a_all[is_leu]
    if is_hep.sum() and neg_ref.size > 20:
        floor = np.nanpercentile(neg_ref, 90)
        frac_hep_low = float(np.mean(a_all[is_hep] <= floor))
    else:
        frac_hep_low = np.nan
    cd45pos = df.is_cd45.to_numpy(bool) if "is_cd45" in df else is_leu
    recov = float((df.celltype.to_numpy()[cd45pos] == "leukocyte_CD45").mean()) if cd45pos.sum() else 1.0
    m = dict(
        sample=sample, n_cells=n, n_nuclei=int(nuc.max()),
        hepato_pct=round(100 * is_hep.mean(), 1),
        leuko_pct=round(100 * is_leu.mean(), 2),
        other_pct=round(100 * (1 - is_hep.mean() - is_leu.mean()), 1),
        cd45pos_pct=round(100 * cd45pos.mean(), 2),
        frac_hepato_in_ASGR1_low=round(frac_hep_low, 3) if np.isfinite(frac_hep_low) else np.nan,
        frac_cd45pos_recovered=round(recov, 3),
        med_diam_hepato_um=round(float(np.median(diam[is_hep])), 1) if is_hep.sum() else np.nan,
        med_diam_leuko_um=round(float(np.median(diam[is_leu])), 1) if is_leu.sum() else np.nan,
        med_nuc_diam_um=round(nuc_diam_px * PIXEL_SIZE_UM, 2),
    )
    pd.DataFrame([m]).to_csv(qc / "metrics.csv", index=False)
    print(f"  {sample}: hepato={m['hepato_pct']}% leuko={m['leuko_pct']}% other={m['other_pct']}% | "
          f"cd45+={m['cd45pos_pct']}% recov={m['frac_cd45pos_recovered']} | "
          f"hepInASGR1low={m['frac_hepato_in_ASGR1_low']} | "
          f"Ø hep={m['med_diam_hepato_um']}µm leu={m['med_diam_leuko_um']}µm nuc={m['med_nuc_diam_um']}µm | "
          f"crops -> {qc}", flush=True)
    return m


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    todo = (sorted(p.name for p in DATA_ROOT.glob("sample_*") if p.is_dir())
            if which in ("all", "") else
            [which if which.startswith("sample_") else f"sample_{which}"])
    rows = []
    for s in todo:
        try:
            rows.append(validate(s))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {s}: {type(e).__name__}: {e}", flush=True)
    if len(rows) > 1:
        out = DATA_ROOT / "_nox1_normalization" / "qc_metrics_v5.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\ncohort QC -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
