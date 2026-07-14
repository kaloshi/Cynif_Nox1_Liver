"""Pipeline 1 - illumination re-correction and de-seaming of all 6 channels -> clean mosaic.

The upstream per-tile illumination correction used get_darkfield=False, leaving an uncorrected
additive per-tile background (tile seams). Here, for every sample and channel a BaSiC flatfield +
DARKFIELD is fitted (get_darkfield=True) and each tile is corrected as (raw - darkfield)/flatfield.
Residual per-tile additive offsets are then removed by overlap-based least-squares balancing (one
offset per tile, so neighbouring tiles agree in their overlap regions), and tiles are feather-blended
at the detected origins (DAPI template matching, NCC ~0.96). All six channels are de-seamed so that
segmentation (DAPI+ASGR1) and gating (ASGR1/CD45) run on clean data; NNLS-v3 AF removal
(hdcycif.af_removal) is then applied to the markers using the de-seamed AF1/AF2 as donors.

Output: <sample>/AF_removal/fused_decon_refused.ome.tif  (6-channel uint16, clean)
        <sample>/AF_removal/NOX1_refused.tif              (NOX1 channel only)
        <OUT_ROOT>/refuse_seamscore.csv, illum_global/, deseam_illum_qc.png

Env knobs: HD_DATA_ROOT, HD_SAMPLES (subset), HD_FIT_DS (default 4), HD_DEVICE (default cuda).
"""
import os
import time
import json
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import block_reduce
from skimage.transform import resize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basicpy import BaSiC
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from hdcycif import config as C
from hdcycif import tiles as S9          # detect_origins, feather_weight, seam_and_cv
from hdcycif import af_removal as AF

T = 2048
GRID = 5
DEVICE = os.environ.get("HD_DEVICE", "cuda")
FIT_DS = int(os.environ.get("HD_FIT_DS", "4"))
SMOOTHNESS = 1.0
CHAN_ORDER = ["DAPI", "AF1", "AF2", "ASGR1", "CD45", "NOX1"]   # stack/output order (== C.CH index)
MARKERS = ["ASGR1", "CD45", "NOX1"]                            # AF-removed; donors AF1/AF2
RAW = "cyc001/Z-Stacks/fileseries_export/z01/tiles/tile_C{ch:02d}S{k:05d}.tif"
ILLUM_OUT = C.OUT_ROOT / "illum_global"


def sample_list():
    env = os.environ.get("HD_SAMPLES", "").strip()
    if env:
        want = set(env.split(","))
        return [s for s in C.list_samples() if s in want]
    return C.list_samples()


def make_basic(**kw):
    fields = set(getattr(BaSiC, "model_fields", None) or getattr(BaSiC, "__fields__", {}))
    return BaSiC(**{k: v for k, v in kw.items() if k in fields})


def load_raw_tile(sample, ch_idx, k):
    return tiff.imread(str(C.DATA_ROOT / sample / RAW.format(ch=ch_idx, k=k))).astype(np.float32)


def balance_offsets(tiles, origins):
    """Per-tile additive offsets minimizing overlap disagreement (least squares, sum=0).

    tiles: list of 25 illum-corrected 2D tiles ; origins: their (y,x) in the mosaic.
    Returns o (len 25): subtract o[k] from tile k to make neighbouring tiles agree.
    """
    n = len(tiles)
    rows, rhs, wts = [], [], []
    for i in range(n):
        yi, xi = origins[i]
        for j in range(i + 1, n):
            yj, xj = origins[j]
            y0, x0 = max(yi, yj), max(xi, xj)
            y1, x1 = min(yi + T, yj + T), min(xi + T, xj + T)
            if (y1 - y0) < 32 or (x1 - x0) < 32:
                continue
            a = tiles[i][y0 - yi:y1 - yi, x0 - xi:x1 - xi]
            b = tiles[j][y0 - yj:y1 - yj, x0 - xj:x1 - xj]
            d = float(np.median(a - b))                 # robust per-pixel offset (registered overlap)
            w = np.sqrt((y1 - y0) * (x1 - x0))
            r = np.zeros(n); r[i] = w; r[j] = -w
            rows.append(r); rhs.append(w * d); wts.append(w)
    if not rows:
        return np.zeros(n)
    M = np.vstack(rows + [np.ones(n)])                  # anchor: sum(o)=0
    y = np.asarray(rhs + [0.0])
    o, *_ = np.linalg.lstsq(M, y, rcond=None)
    return o


def fit_sample_fields(sample):
    """PER-SAMPLE BaSiC flatfield + DARKFIELD per channel (this sample's 25 z01 tiles).

    Per-sample (not pooled): a single pooled flatfield over 16 samples does not match each
    sample's own illumination and leaves a large-scale background gradient that the AF
    subtraction then imprints onto NOX1. This is the original pipeline's per-sample approach,
    only now with get_darkfield=True (the upstream bug was get_darkfield=False).
    """
    fields = {}
    for c in CHAN_ORDER:
        ci = C.CH[c]
        small = [block_reduce(load_raw_tile(sample, ci, k), (FIT_DS, FIT_DS), np.mean).astype(np.float32)
                 for k in range(GRID * GRID)]
        stack = np.ascontiguousarray(np.stack(small)); del small
        basic = make_basic(get_darkfield=True, device=DEVICE, smoothness_flatfield=SMOOTHNESS,
                           fitting_mode="approximate")
        basic.fit(stack)
        ff = np.asarray(basic.flatfield, np.float32); ff = ff / ff.mean()
        dk = np.asarray(basic.darkfield, np.float32)
        ff2 = resize(ff, (T, T), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        dk2 = resize(dk, (T, T), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        fields[c] = dict(flatfield=ff2, darkfield=dk2)
        del stack, basic
    return fields


def fuse_channel(sample, c, origins, H, W, fields, wtile):
    """Correct + overlap-balance + feather-fuse one channel -> mosaic (float32)."""
    ci = C.CH[c]
    ff, dk = fields[c]["flatfield"], fields[c]["darkfield"]
    ffsafe = np.where(ff <= 0, 1.0, ff)
    tiles = [(load_raw_tile(sample, ci, k) - dk) / ffsafe for k in range(GRID * GRID)]
    offs = balance_offsets(tiles, origins)
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    for k, (y, x) in enumerate(origins):
        acc[y:y + T, x:x + T] += wtile * (tiles[k] - offs[k])
        wsum[y:y + T, x:x + T] += wtile
    out = np.zeros((H, W), np.float32)
    np.divide(acc, wsum, out=out, where=wsum > 0)
    return np.clip(out, 0, 65535)


def main():
    samples = sample_list()
    print(f"Samples ({len(samples)}): {samples}", flush=True)
    ILLUM_OUT.mkdir(parents=True, exist_ok=True)

    print("Pass 1: detect tile positions ...", flush=True)
    geom = {}
    det_rows = []
    for s in samples:
        dapi = C.load_channel(s, "DAPI")
        origins, med_ncc, min_ncc, n_weak = S9.detect_origins(s, dapi)
        geom[s] = (origins, dapi.shape)
        det_rows.append(dict(sample=s, med_ncc=med_ncc, min_ncc=min_ncc, n_weak=n_weak))
        print(f"  {s}: NCC med={med_ncc:.2f} min={min_ncc:.2f} weak={int(n_weak)}/25", flush=True)
        del dapi
    pd.DataFrame(det_rows).to_csv(C.OUT_ROOT / "refuse_positions_qc.csv", index=False)

    print("\nPass 2: per-sample fit + correct + overlap-balance + fuse (6ch) + AF-removal ...", flush=True)
    seam_rows = []
    names = CHAN_ORDER
    fields = None
    for s in samples:
        origins, (H, W) = geom[s]
        overlap = int(np.clip(T - (W - T) / (GRID - 1), 60, 400))
        wtile = S9.feather_weight(overlap)
        fields = fit_sample_fields(s)   # per-sample flatfield + darkfield (all channels)
        fused = {c: fuse_channel(s, c, origins, H, W, fields, wtile) for c in CHAN_ORDER}

        # AF removal (NNLS V3) on markers, using de-seamed AF1/AF2 as donors
        clean = {}
        af_actions = {}
        for m in MARKERS:
            cm, rep = AF.remove_af(fused[m], fused["AF1"], fused["AF2"])
            clean[m] = cm; af_actions[m] = rep["action"]
        # assemble 6ch (DAPI, AF1, AF2 kept; markers AF-removed)
        stack = np.zeros((6, H, W), np.uint16)
        for ci, c in enumerate(CHAN_ORDER):
            src = clean[c] if c in MARKERS else fused[c]
            stack[ci] = np.clip(src, 0, 65535).astype(np.uint16)
        out6 = C.DATA_ROOT / s / "AF_removal" / "fused_decon_refused.ome.tif"
        tiff.imwrite(str(out6), stack, ome=True, bigtiff=True, compression="zlib",
                     photometric="minisblack", metadata={"axes": "CYX", "Channel": {"Name": names}})
        tiff.imwrite(str(C.DATA_ROOT / s / "AF_removal" / "NOX1_refused.tif"),
                     stack[C.CH["NOX1"]], compression="zlib")

        # per-channel seam QC measured on the de-seamed FUSED channels (pre-AF). The seam metric
        # is a background-percentile ratio, so measuring it on the AF-removed/clipped markers
        # (background ~0) would blow up the denominator -> use the pre-AF fused mosaics instead.
        row = dict(sample=s)
        for c in ["DAPI", "ASGR1", "CD45", "NOX1"]:
            sa, ca = S9.seam_and_cv(fused[c], origins)
            row[f"seam_{c}"] = sa; row[f"tilecv_{c}"] = ca
        row["clip0_NOX1_pct"] = 100 * float(np.mean(stack[C.CH["NOX1"]] == 0))
        row["af_nox1"] = af_actions["NOX1"]
        seam_rows.append(row)
        print(f"  {s}: seam DAPI={row['seam_DAPI']:.1f} ASGR1={row['seam_ASGR1']:.1f} "
              f"CD45={row['seam_CD45']:.1f} NOX1={row['seam_NOX1']:.1f}  "
              f"clip0_NOX1={row['clip0_NOX1_pct']:.1f}%  AF={af_actions}", flush=True)
        del fused, clean, stack

    sc = pd.DataFrame(seam_rows); sc.to_csv(C.OUT_ROOT / "refuse_seamscore.csv", index=False)
    print("\n" + sc.to_string(index=False))
    for c in ["DAPI", "ASGR1", "CD45", "NOX1"]:
        print(f"  mean seam {c}: {sc[f'seam_{c}'].mean():.2f}%", flush=True)

    # illum QC figure
    fig, ax = plt.subplots(2, 6, figsize=(20, 7))
    for j, c in enumerate(CHAN_ORDER):
        im0 = ax[0, j].imshow(fields[c]["flatfield"], cmap="magma"); ax[0, j].set_title(f"{c} flat"); ax[0, j].axis("off")
        plt.colorbar(im0, ax=ax[0, j], fraction=0.046)
        im1 = ax[1, j].imshow(fields[c]["darkfield"], cmap="viridis"); ax[1, j].set_title(f"{c} dark"); ax[1, j].axis("off")
        plt.colorbar(im1, ax=ax[1, j], fraction=0.046)
    fig.suptitle("Per-sample illumination model (flatfield + darkfield) — last sample shown", fontsize=13)
    fig.tight_layout(); fig.savefig(C.OUT_ROOT / "deseam_illum_qc.png", dpi=120)
    print(f"Wrote {C.OUT_ROOT/'deseam_illum_qc.png'}")


if __name__ == "__main__":
    main()
