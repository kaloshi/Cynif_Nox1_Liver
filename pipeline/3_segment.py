"""Pipeline 3 - fast, GPU-free nuclei + cell-body segmentation on the de-seamed clean mosaic,
tuned for HEPATOCYTE detection. CellposeSAM v4 (the only installed cellpose) needs ~105 min/sample
(whole-cell) on CPU -> impractical for 16. This watershed pipeline runs in minutes/sample.

Nucleus-only segmentation gives cell bodies too small to capture the ASGR1 cytoplasm/membrane, so
hepatocytes fail the ASGR1+ gate. Here nuclei are grown into realistic hepatocyte bodies (Voronoi,
radius-capped) so the membrane ring sits on the ASGR1+ membrane and the gate (pipeline/4) recovers
far more hepatocytes.

Per sample (fused_decon_refused.ome.tif):
  nuclei: clean DAPI -> gaussian -> Otsu -> distance-watershed (splits touching nuclei)
  cells : tissue-filling Voronoi watershed from the nuclei, radius-capped at CELL_MAX_EXPAND px
  ring  : MEM_RING_PX-px membrane ring (hdcycif.segmentation.make_membrane_ring); all 6 channels quantified
Output: <sample>/segmentation_v4/{whole_cell_label_v4,nuclei_v4,membrane_ring_v4}.tiff,
        cell_stats_v4.csv, seg_v4_overview_<tag>.png

Env: HD_DATA_ROOT, HD_SAMPLES (subset). Tunables at top.
"""
import os
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, expand_labels, find_boundaries
from skimage.morphology import remove_small_objects
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from hdcycif import config as C
from hdcycif import segmentation as seg   # make_membrane_ring + quantify

PX_UM = 0.325
NUC_SIGMA = 1.0
NUC_THR_FACTOR = 0.6      # fraction of Otsu -> catch dim hepatocyte nuclei (more cells)
NUC_MIN_DIST = 7          # px between nucleus centers (~2.3 um)
NUC_MIN_AREA = 20         # px (drop specks)
CELL_MAX_EXPAND = 32      # px cap on nucleus->cell radius (~10 um; Voronoi-fills tissue otherwise)
MEM_RING_PX = 4           # membrane ring thickness (wider -> no empty rings on big cells)


def sample_list():
    env = os.environ.get("HD_SAMPLES", "").strip()
    if env:
        want = set(env.split(","))
        return [s for s in C.list_samples() if s in want]
    return C.list_samples()


def segment(dapi, asgr1):
    # --- nuclei (sensitive threshold to catch dim hepatocyte nuclei) ---
    d = gaussian(dapi, NUC_SIGMA, preserve_range=True)
    pos = d[d > 0]
    thr = (threshold_otsu(pos) if pos.size else 0.0) * NUC_THR_FACTOR
    nuc_mask = remove_small_objects(d > thr, NUC_MIN_AREA)
    dist = ndi.distance_transform_edt(nuc_mask)
    coords = peak_local_max(dist, min_distance=NUC_MIN_DIST, labels=nuc_mask)
    markers = np.zeros(dist.shape, np.int32)
    if len(coords):
        markers[tuple(coords.T)] = 1
    markers = ndi.label(markers)[0]
    nuclei = watershed(-dist, markers, mask=nuc_mask).astype(np.int32)

    # --- tissue mask (include cytoplasm between nuclei) ---
    a = gaussian(asgr1, NUC_SIGMA, preserve_range=True)
    apos = a[a > 0]
    a_thr = (threshold_otsu(apos) if apos.size else np.inf) * 0.4
    tissue = nuc_mask | (a > a_thr)
    tissue = ndi.binary_closing(tissue, structure=np.ones((9, 9)))
    tissue = ndi.binary_fill_holes(tissue)

    # --- cell bodies: tissue-filling Voronoi watershed from nuclei, radius-capped ---
    bg_dist = ndi.distance_transform_edt(nuclei == 0)         # distance from nearest nucleus
    cells = watershed(bg_dist, markers=nuclei, mask=tissue).astype(np.int32)
    cells[bg_dist > CELL_MAX_EXPAND] = 0                      # cap territory of isolated nuclei
    cells[nuclei > 0] = nuclei[nuclei > 0]
    return nuclei, cells


def main():
    for s in sample_list():
        stack = tiff.imread(str(C.DATA_ROOT / s / "AF_removal" / "fused_decon_refused.ome.tif")).astype(np.float32)
        nuclei, cells = segment(stack[C.CH["DAPI"]], stack[C.CH["ASGR1"]])
        mr = seg.make_membrane_ring(cells, MEM_RING_PX)

        out = C.DATA_ROOT / s / "segmentation_v4"; out.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(str(out / "whole_cell_label_v4.tiff"), cells.astype(np.uint32), compression="zlib")
        tiff.imwrite(str(out / "nuclei_v4.tiff"), nuclei.astype(np.uint32), compression="zlib")
        tiff.imwrite(str(out / "membrane_ring_v4.tiff"), mr.astype(np.uint32), compression="zlib")

        cs = seg.quantify(cells, mr, stack, s)
        cs.to_csv(str(out / "cell_stats_v4.csv"), index=False)

        areas = cs["Area"].to_numpy(float)
        diam_um = 2 * np.sqrt(areas / np.pi) * PX_UM
        # quick ASGR1+ estimate (nan-safe Otsu) for a sanity check on hepatocyte yield
        a = cs["ASGR1_mem_median"].to_numpy(float); cd = cs["CD45_mem_median"].to_numpy(float)
        fin = np.isfinite(a) & np.isfinite(cd)
        a_thr = threshold_otsu(a[np.isfinite(a)]); c_thr = threshold_otsu(cd[np.isfinite(cd)])
        hep = np.mean(fin & (a > a_thr) & (cd <= c_thr)) * 100
        empty = 100 * np.mean(~np.isfinite(a))
        print(f"  {s}: {int(cells.max())} cells | diam med={np.median(diam_um):.1f}um "
              f"p5-95={np.percentile(diam_um,5):.1f}-{np.percentile(diam_um,95):.1f} | "
              f"empty_ring={empty:.0f}% | hepato~{hep:.0f}%", flush=True)

        # overview
        step = 8
        dn = stack[C.CH["DAPI"]][::step, ::step]; dn = np.clip(dn / max(np.percentile(dn, 99), 1), 0, 1)
        b = find_boundaries(cells[::step, ::step], mode="outer")
        rgb = np.stack([dn, dn, dn], -1); rgb[b] = [1, 0.3, 0.1]
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(dn, cmap="gray"); ax[0].set_title("DAPI"); ax[0].axis("off")
        ax[1].imshow(rgb); ax[1].set_title(f"v4 cells n={int(cells.max())}"); ax[1].axis("off")
        fig.tight_layout(); fig.savefig(str(out / f"seg_v4_overview_{s.replace('sample_','')}.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
