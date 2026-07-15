"""3b — segmentation_v5: Cellpose-SAM nuclei (GPU) + tissue-bounded expansion +
own-territory / SPARQ features. Reuses the Lynch `cynif` package.

Fixes the v4 problems at their structural root:
  * v4 nuclei = DAPI 0.6xOtsu watershed under-detects dim/dense immune nuclei
    -> Cellpose-SAM (tile-local norm, flow 0.6) recovers them (the missing CD45+).
  * v4 measured markers on a 4px morphological ring over 32px Voronoi bodies that
    samples NEIGHBOR signal -> here we also compute SPARQ local-contrast (fc vs a
    per-cell local background), the neighbor- and graded-marker-robust positivity
    the gating (4b) is built on.

Outputs (per sample, alongside the untouched segmentation_v4/):
  <sample>/segmentation_v5/{nuclei_v5,whole_cell_label_v5,matched_cytoplasm_v5,
                            membrane_ring_v5}.tiff  and  cell_stats_v5.csv

Run:
  HD_DATA_ROOT=/path/to/HD \
    micromamba run -n cynif python 3b_segment_cellpose_v5.py [SAMPLE|all]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import expand_labels
from scipy.ndimage import binary_closing, binary_fill_holes

# --- make the Lynch cynif package importable (it holds the reusable seg/feature fns) ---
CYNIF_SRC = os.environ.get("CYNIF_SRC")   # companion `cynif` package src dir; else have it on PYTHONPATH
if CYNIF_SRC and CYNIF_SRC not in sys.path:
    sys.path.insert(0, CYNIF_SRC)

from cynif.segmentation.cellpose_nuclei import segment_nuclei
from cynif.segmentation.reconcile import demerge_nuclei
from cynif.segmentation.compartments_masks import make_compartments
from cynif.features import morphology, intensity as fi, sparq_contrast as sq
from cynif.utils import xp as _xp

# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("HD_DATA_ROOT", "."))
PIXEL_SIZE_UM = 0.26
CH = {"DAPI": 0, "AF1": 1, "AF2": 2, "ASGR1": 3, "CD45": 4, "NOX1": 5}
CHANNELS = ["DAPI", "AF1", "AF2", "ASGR1", "CD45", "NOX1"]
MARKERS_SPARQ = ["ASGR1", "CD45"]              # local-contrast positivity markers
SEG_DIR = "segmentation_v5"

# input mosaic candidates (prefer the de-seamed clean 6-channel mosaic)
IMG_CANDIDATES = [
    "AF_removal/fused_decon_refused.ome.tif",
    "AF_removal/fused_decon_AF_cleaned_nnls_v3_NOX1norm_tileflat.ome.tif",
    "AF_removal/fused_decon_AF_cleaned_nnls_v3.ome.tif",
]

# segmentation tunables
EXPAND_PX = round(8.0 / PIXEL_SIZE_UM)          # ~31 px ~ 8 µm cell-body grow cap
ASGR1_TISSUE_FACTOR = 0.4                        # tissue = nuclei | ASGR1 > factor*Otsu
DNA_BG_PCT = 20.0                                # DNA gate: nucleus DAPI > pct*factor
DNA_MIN_FACTOR = 1.1
MIN_CELL_AREA = 120                              # px; low so small CD45+ leukocytes survive
RING_WIDTH_PX = 4


def samples():
    return sorted(p.name for p in DATA_ROOT.glob("sample_*") if p.is_dir())


def image_path(sample: str) -> Path:
    d = DATA_ROOT / sample
    for c in IMG_CANDIDATES:
        if (d / c).exists():
            return d / c
    raise FileNotFoundError(f"no input mosaic in {d/'AF_removal'}")


# --------------------------------------------------------------------------- #
def _dna_gate_and_minarea(nuclei, cells, dapi):
    """Drop DAPI-empty false nuclei and sub-floor cells; keep nuclei<->cells consistent.

    Cells are derived 1:1 from nuclei (expand_labels preserves the seed id), so a
    cell and its nucleus share the same label -> dropping a label removes both.
    """
    nmax = int(max(nuclei.max(), cells.max()))
    # per-nucleus mean DAPI (label-indexed)
    nuc_g = _xp.asarray(nuclei.astype("int32"))
    dapi_g = _xp.asarray(dapi.astype("float32"))
    nuc_dapi = _xp.asnumpy(fi.label_mean(dapi_g, nuc_g, nmax + 1))
    del nuc_g, dapi_g
    _xp.free_pool()
    thr = float(np.percentile(dapi, DNA_BG_PCT)) * DNA_MIN_FACTOR
    cell_area = np.bincount(cells.ravel(), minlength=nmax + 1)

    keep = np.zeros(nmax + 1, dtype=bool)
    labels = np.arange(nmax + 1)
    keep[labels] = (nuc_dapi > thr) & (cell_area >= MIN_CELL_AREA)
    keep[0] = False

    lut = np.zeros(nmax + 1, dtype="int32")
    lut[keep] = labels[keep]
    cells2 = lut[cells]
    nuclei2 = lut[nuclei]
    n_dropped = int((~keep[1:]).sum() - (labels[1:] > cells.max()).sum())
    return nuclei2.astype("int32"), cells2.astype("int32"), int(keep[1:].sum())


def segment(sample: str, *, verbose=True):
    """Cellpose nuclei -> tissue-bounded cells -> DNA-gate/min-area -> compartments.
    Returns dict of label images (all int32)."""
    img = image_path(sample)
    t0 = time.time()
    dapi = tifffile.imread(str(img), key=CH["DAPI"]).astype("float32")
    asgr1 = tifffile.imread(str(img), key=CH["ASGR1"]).astype("float32")

    # 1) nuclei
    nuclei = segment_nuclei(dapi, tile_norm_blocksize=128, flow_threshold=0.6,
                            cellprob_threshold=0.0, diameter=None, gpu=True).astype("int32")
    n_raw = int(nuclei.max())
    # 2) de-merge fused nuclei (guarded: only big blobs)
    nuclei, n_split = demerge_nuclei(nuclei, min_distance=9, min_merged_factor=1.5, min_sub_area=30)
    nuclei = nuclei.astype("int32")

    # 3) tissue mask
    a = gaussian(asgr1, 1.0, preserve_range=True)
    pos = a[a > 0]
    a_thr = (threshold_otsu(pos) if pos.size else 0.0) * ASGR1_TISSUE_FACTOR
    tissue = (nuclei > 0) | (a > a_thr)
    tissue = binary_fill_holes(binary_closing(tissue, structure=np.ones((9, 9), bool)))

    # 4) whole-cell = tissue-bounded nucleus expansion (Voronoi-limited, no membrane marker)
    cells = expand_labels(nuclei, distance=EXPAND_PX)
    cells = np.where(tissue, cells, 0).astype("int32")

    # 5) DNA gate + min-area (keeps nuclei<->cells consistent)
    nuclei, cells, n_keep = _dna_gate_and_minarea(nuclei, cells, dapi)

    # 6) compartments (GPU)
    comp = make_compartments(nuclei, cells, ring_width_px=RING_WIDTH_PX)
    if verbose:
        print(f"  [seg] {sample}: raw={n_raw} nuclei, +{n_split} split, kept={n_keep} cells "
              f"({time.time()-t0:.0f}s)", flush=True)
    return comp


# --------------------------------------------------------------------------- #
def features(sample: str, comp: dict, *, verbose=True):
    """Own-territory means (whole-cell + cytoplasm + ring) + SPARQ local contrast."""
    img = image_path(sample)
    wc = comp["whole_cell"].astype("int32")
    cyto = comp["matched_cytoplasm"].astype("int32")
    ring = comp["membrane_ring"].astype("int32")
    n_labels = int(wc.max()) + 1

    regions, rows = morphology.regions_and_morphology(wc)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{sample}: no cells after segmentation")
    cell_ids = df["CellID"].to_numpy()

    wc_g = _xp.asarray(wc)
    cyto_g = _xp.asarray(cyto)
    ring_g = _xp.asarray(ring)

    # SPARQ per-cell fg/bg sample coords
    fg, bg, bg_valid = sq.build_sample_coords(regions, wc.shape, n_samples=200, kf=200,
                                              mu_factor=1.0, seed=0)
    bg_frac = bg_valid.mean(axis=1)

    fc_mat, p_mat, marker_names = [], [], []
    for nm in CHANNELS:
        plane = _xp.asarray(tifffile.imread(str(img), key=CH[nm]).astype("float32"))
        df[f"{nm}_mean"] = _xp.asnumpy(fi.label_mean(plane, wc_g, n_labels))[cell_ids]
        # cytoplasm mean (own territory, excludes nucleus) for the lineage/readout markers
        if nm in ("ASGR1", "CD45", "NOX1"):
            df[f"{nm}_cyto_mean"] = _xp.asnumpy(fi.label_mean(plane, cyto_g, n_labels))[cell_ids]
        # membrane-ring mean+median for QC comparison to v4 (NOT used for gating)
        if nm in ("ASGR1", "CD45"):
            rmean, rmed, _rp = fi.label_mean_quant(plane, ring_g, n_labels, qs=(0.5, 0.75))
            df[f"{nm}_mem_mean"] = _xp.asnumpy(rmean)[cell_ids]
            df[f"{nm}_mem_median"] = _xp.asnumpy(rmed)[cell_ids]
        # SPARQ local contrast for the positivity markers
        if nm in MARKERS_SPARQ:
            fc, p = sq.contrast_for_channel(plane, fg, bg)
            low = bg_frac < 0.5
            fc[low] = np.nan
            p[low] = np.nan
            df[f"{nm}_fc"] = fc
            df[f"{nm}_pval"] = p
            fc_mat.append(fc)
            p_mat.append(p)
            marker_names.append(nm)
        del plane
        _xp.free_pool()

    if fc_mat:
        padj = sq.bonferroni(np.stack(p_mat, axis=1).astype("float32"))
        for k, nm in enumerate(marker_names):
            df[f"{nm}_padj"] = padj[:, k]

    df["bg_frac"] = bg_frac
    df["Sample"] = sample
    df["patient"] = sample.split("_")[1]
    del wc_g, cyto_g, ring_g
    _xp.free_pool()
    if verbose:
        print(f"  [feat] {sample}: {len(df)} cells, {len(marker_names)} SPARQ markers", flush=True)
    return df


def run_sample(sample: str):
    t0 = time.time()
    comp = segment(sample)
    df = features(sample, comp)
    outdir = DATA_ROOT / sample / SEG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(outdir / "nuclei_v5.tiff"), comp["nuclei"].astype("uint32"), compression="zlib")
    tifffile.imwrite(str(outdir / "whole_cell_label_v5.tiff"), comp["whole_cell"].astype("uint32"), compression="zlib")
    tifffile.imwrite(str(outdir / "matched_cytoplasm_v5.tiff"), comp["matched_cytoplasm"].astype("uint32"), compression="zlib")
    tifffile.imwrite(str(outdir / "membrane_ring_v5.tiff"), comp["membrane_ring"].astype("uint32"), compression="zlib")
    df.to_csv(outdir / "cell_stats_v5.csv", index=False)
    print(f"[done] {sample}: {len(df)} cells -> {outdir}/cell_stats_v5.csv  ({time.time()-t0:.0f}s)", flush=True)
    return df


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    todo = samples() if which in ("all", "") else [which if which.startswith("sample_") else f"sample_{which}"]
    print(f"segmentation_v5 | DATA_ROOT={DATA_ROOT} | {len(todo)} sample(s) | expand={EXPAND_PX}px @ {PIXEL_SIZE_UM}µm", flush=True)
    for s in todo:
        try:
            run_sample(s)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {s}: {type(e).__name__}: {e}", flush=True)
        _xp.free_pool()


if __name__ == "__main__":
    main(sys.argv)
