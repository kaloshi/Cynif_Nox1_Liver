"""Tile-origin detection and blending helpers for the illumination re-fusion (pipeline/1).

The exact position of every raw tile in the fused mosaic is recovered by normalized
cross-correlation template matching of the still-existing single DAPI tiles against the mosaic
(coarse 8x + local fine 2x). This is robust to intensity scaling, so the same origins apply to
every channel. ``feather_weight`` gives a linear-ramp blending weight; ``seam_and_cv`` is the
background-discontinuity QC metric used to validate de-seaming.
"""
import numpy as np
import tifffile as tiff
from skimage.feature import match_template

from hdcycif import config as C

T = 2048          # tile size (px)
GRID = 5          # tiles per row/col (5x5 = 25)
DS_COARSE = 8     # downsample for the full-mosaic coarse match
DS_FINE = 2       # downsample for the local fine refinement


def nominal_origins(dim):
    step = (dim - T) / (GRID - 1)
    return [int(round(i * step)) for i in range(GRID)]


def detect_origins(sample, dapi_mos):
    """Return 25 (y, x) tile origins in the mosaic frame by matching the real single DAPI tiles.

    Coarse full-mosaic NCC (8x) then a local fine refinement (2x). Returns
    (origins, median_ncc, min_ncc, n_weak<0.4).
    """
    H, W = dapi_mos.shape
    mosC = dapi_mos[::DS_COARSE, ::DS_COARSE].astype(np.float32)
    tdir = C.DATA_ROOT / sample / "cyc001/Z-Stacks/tiles_precorrected/z01/tiles"
    origins = []
    nccs = []
    for k in range(GRID * GRID):
        tile = tiff.imread(str(tdir / f"tile_S{k:05d}.tif"), key=C.CH["DAPI"]).astype(np.float32)
        tC = tile[::DS_COARSE, ::DS_COARSE]
        r = match_template(mosC, tC)
        py, px = np.unravel_index(np.argmax(r), r.shape)
        cy, cx = py * DS_COARSE, px * DS_COARSE
        pad = DS_COARSE + 4
        y0 = max(0, cy - pad); x0 = max(0, cx - pad)
        y1 = min(H, cy + T + pad); x1 = min(W, cx + T + pad)
        regF = dapi_mos[y0:y1, x0:x1][::DS_FINE, ::DS_FINE].astype(np.float32)
        tF = tile[::DS_FINE, ::DS_FINE]
        ncc = float(r.max())
        if regF.shape[0] > tF.shape[0] and regF.shape[1] > tF.shape[1]:
            rf = match_template(regF, tF)
            fy, fx = np.unravel_index(np.argmax(rf), rf.shape)
            cy, cx = y0 + fy * DS_FINE, x0 + fx * DS_FINE
            ncc = float(rf.max())
        origins.append((int(np.clip(cy, 0, H - T)), int(np.clip(cx, 0, W - T))))
        nccs.append(ncc)
    nccs = np.array(nccs)
    return origins, float(np.median(nccs)), float(np.min(nccs)), float((nccs < 0.4).sum())


def feather_weight(overlap):
    """Linear-ramp (feather) blending weight of one tile, ramping over `overlap` px at the edges."""
    d = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(np.float32)
    w1 = np.clip(d, 1, max(overlap, 2))
    return np.outer(w1, w1)


def seam_and_cv(img, origins):
    """QC metric: mean background discontinuity (%) across detected tile seams + per-tile bg CV."""
    band = 40

    def cluster(vals):
        vals = sorted(vals); groups = [[vals[0]]]
        for v in vals[1:]:
            (groups[-1] if v - groups[-1][-1] < T // 2 else groups.append([v]) or groups[-1]).append(v)
        return [int(np.mean(g)) for g in groups]

    rows = cluster([o[0] for o in origins]); cols = cluster([o[1] for o in origins])
    vals = []
    for c in cols[1:]:
        l = np.percentile(img[:, max(0, c - band):c], 15); r = np.percentile(img[:, c:c + band], 15)
        vals.append(abs(l - r) / max((l + r) / 2, 1e-6) * 100)
    for rr in rows[1:]:
        t = np.percentile(img[max(0, rr - band):rr, :], 15); b = np.percentile(img[rr:rr + band, :], 15)
        vals.append(abs(t - b) / max((t + b) / 2, 1e-6) * 100)
    bgs = np.array([np.percentile(img[y + T // 2 - 300:y + T // 2 + 300, x + T // 2 - 300:x + T // 2 + 300], 15)
                    for y, x in origins])
    return float(np.mean(vals)), float(bgs.std() / max(bgs.mean(), 1e-6))
