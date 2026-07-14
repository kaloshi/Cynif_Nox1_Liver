"""Pipeline 6 - high-resolution per-scene review figures (2x2 panels, JPG).

For every scene, on the de-seamed clean mosaic:
  1  DAPI (blue) + NOX1 (red) composite
  2  NOX1 alone (magma)
  3  NOX1 with the hepatocyte mask overlaid at 50% transparency (green)
  4  ASGR1 alone (hepatocyte marker, viridis)
Title = "<sample> (<group>)" (group MUT/WT). Optionally (HD_MAKE_PDF=1) the JPGs are assembled
into one PDF.

Run order: after pipeline/4 (needs cell_stats_v4_gated.parquet) and pipeline/2 (NOX1_refused_norm).
Env: HD_DATA_ROOT, HD_SAMPLES (subset), HD_FIG_DS (default 2), HD_FIG_DPI (250), HD_FIG_IN (30),
     HD_FIG_JPGQ (90), HD_MAKE_PDF (0).
Output: <OUT_ROOT>/scene_figures/<sample>.jpg  (+ NOX1_scene_figures.pdf if HD_MAKE_PDF=1)
"""
import os
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import block_reduce
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from hdcycif import config as C

GROUP = {"LXT40": "MUT", "LXT47": "MUT", "LXT52": "MUT",
         "LXT31": "WT", "LXT32": "WT", "LXT38": "WT", "LXT41": "WT", "LXT48": "WT"}
DS = int(os.environ.get("HD_FIG_DS", "2"))
DPI = int(os.environ.get("HD_FIG_DPI", "250"))
FIGIN = float(os.environ.get("HD_FIG_IN", "30"))   # figure side in inches (panel px ~= FIGIN*DPI/2)
JPG_Q = int(os.environ.get("HD_FIG_JPGQ", "90"))
CLEAN6 = "AF_removal/fused_decon_refused.ome.tif"
NOX1N = "AF_removal/NOX1_refused_norm.tif"
WCELL = "segmentation_v4/whole_cell_label_v4.tiff"
OUTDIR = C.OUT_ROOT / "scene_figures"


def sample_list():
    env = os.environ.get("HD_SAMPLES", "").strip()
    return [s for s in C.list_samples() if (not env) or s in set(env.split(","))]


def norm(a, lo=1, hi=99.5):
    a = a.astype(np.float32)
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / max(p2 - p1, 1.0), 0, 1)


def dsmean(a):
    return block_reduce(a.astype(np.float32), (DS, DS), np.mean) if DS > 1 else a.astype(np.float32)


def hepatocyte_mask(sample, hep_ids):
    """Boolean full-res mask of the hepatocytes (whole-cell labels in hep_ids)."""
    wc = tiff.imread(str(C.DATA_ROOT / sample / WCELL))
    mx = int(wc.max())
    lut = np.zeros(mx + 1, bool)
    ids = np.asarray(sorted(i for i in hep_ids if 0 < i <= mx), dtype=np.int64)
    lut[ids] = True
    return lut[wc]


def make_fig(s, hep_ids):
    grp = GROUP.get(C.patient_of(s), "?")
    st = tiff.imread(str(C.DATA_ROOT / s / CLEAN6))
    dN = norm(dsmean(st[C.CH["DAPI"]]))
    aN = norm(dsmean(st[C.CH["ASGR1"]]))
    nN = norm(dsmean(tiff.imread(str(C.DATA_ROOT / s / NOX1N))))
    hep = hepatocyte_mask(s, hep_ids)
    hep = block_reduce(hep, (DS, DS), np.max) if DS > 1 else hep
    H = min(dN.shape[0], nN.shape[0], hep.shape[0]); W = min(dN.shape[1], nN.shape[1], hep.shape[1])
    dN, aN, nN, hep = dN[:H, :W], aN[:H, :W], nN[:H, :W], hep[:H, :W]

    fig, ax = plt.subplots(2, 2, figsize=(FIGIN, FIGIN + 1))
    comp = np.zeros((H, W, 3), np.float32); comp[..., 0] = nN; comp[..., 2] = dN
    ax[0, 0].imshow(comp, interpolation="nearest"); ax[0, 0].set_title("DAPI (blue) + NOX1 (red)", fontsize=22)
    ax[0, 1].imshow(nN, cmap="magma", interpolation="nearest"); ax[0, 1].set_title("NOX1", fontsize=22)
    base = plt.cm.magma(nN)[..., :3].astype(np.float32)
    ov = base.copy(); ov[hep] = 0.5 * base[hep] + 0.5 * np.array([0.1, 1.0, 0.2], np.float32)
    ax[1, 0].imshow(ov, interpolation="nearest")
    ax[1, 0].set_title("NOX1 + hepatocyte mask (50%)", fontsize=22)
    ax[1, 1].imshow(aN, cmap="viridis", interpolation="nearest"); ax[1, 1].set_title("ASGR1 (hepatocyte marker)", fontsize=22)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{s.replace('sample_','')}  ({grp})", fontsize=30, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gated = pd.read_parquet(C.OUT_ROOT / "cell_stats_v4_gated.parquet")
    jpgs = []
    for s in sample_list():
        hep_ids = set(gated.loc[(gated["Sample"] == s) & gated["is_hepato"], "CellID"].astype(int))
        fig = make_fig(s, hep_ids)
        jpg = OUTDIR / f"{s}.jpg"
        fig.savefig(str(jpg), dpi=DPI, pil_kwargs={"quality": JPG_Q})
        plt.close(fig)
        jpgs.append(jpg)
        print(f"  {s} ({GROUP.get(C.patient_of(s),'?')}) -> {jpg.name} ({jpg.stat().st_size/1e6:.1f} MB)", flush=True)
    if os.environ.get("HD_MAKE_PDF", "0") == "1" and len(jpgs) > 1:
        ims = [Image.open(str(p)).convert("RGB") for p in jpgs]
        out_pdf = C.OUT_ROOT / "NOX1_scene_figures.pdf"
        ims[0].save(str(out_pdf), save_all=True, append_images=ims[1:], resolution=DPI)
        print(f"Wrote {out_pdf} ({out_pdf.stat().st_size/1e6:.1f} MB, {len(ims)} pages)", flush=True)


if __name__ == "__main__":
    main()
