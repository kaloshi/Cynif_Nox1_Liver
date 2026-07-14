"""Pipeline 2 - robust cross-sample additive background normalization of the re-fused NOX1.

The original BaSiC "time-series" fit can DIVERGE on a single sample and assign an absurd baseline
(observed: LXT32_scene_2 baseline ~5280 vs ~150 for the rest -> the whole image clips to 0).
Instead we align the per-sample NOX1 BACKGROUND directly: the median NOX1 in ECM (non-cell)
regions, shifted to the cohort median. This matches the documented "additive baseline" intent
(BaSiC baseline correlated with ECM-p50, see verify.py) but is drift/outlier-proof.

    offset_s = ecm_bg_s - median(ecm_bg) ;  NOX1_norm_s = clip(NOX1_refused_s - offset_s, 0)

Output: <sample>/AF_removal/NOX1_refused_norm.tif ; OUT_ROOT/refuse_norm_offsets.csv
"""
import numpy as np
import pandas as pd
import tifffile as tiff
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from hdcycif import config as C

IN_NAME = "NOX1_refused.tif"
OUT_NAME = "NOX1_refused_norm.tif"


def background(s):
    """Robust per-sample NOX1 background = median in ECM (non-cell) regions; fallback p50."""
    nox1 = tiff.imread(str(C.DATA_ROOT / s / "AF_removal" / IN_NAME)).astype(np.float32)
    try:
        ecm = C.load_mask(s, C.ECM)
        H = min(nox1.shape[0], ecm.shape[0]); W = min(nox1.shape[1], ecm.shape[1])
        vals = nox1[:H, :W][ecm[:H, :W] > 0]
        bg = float(np.median(vals)) if vals.size > 1000 else float(np.percentile(nox1, 50))
    except Exception:
        bg = float(np.percentile(nox1, 50))
    return bg


def main():
    samples = C.list_samples()
    bgs = {}
    for s in samples:
        bgs[s] = background(s)
        print(f"  {s}: ECM-bg={bgs[s]:.1f}", flush=True)
    ref = float(np.median(list(bgs.values())))
    print(f"  cohort reference (median ECM-bg) = {ref:.1f}", flush=True)

    rows = []
    for s in samples:
        off = bgs[s] - ref
        nox1 = tiff.imread(str(C.DATA_ROOT / s / "AF_removal" / IN_NAME)).astype(np.float32)
        corr = np.clip(nox1 - off, 0, 65535).astype(np.uint16)
        tiff.imwrite(str(C.DATA_ROOT / s / "AF_removal" / OUT_NAME), corr, compression="zlib")
        rows.append(dict(sample=s, patient=C.patient_of(s), background=bgs[s],
                         reference=ref, offset=off))
        print(f"  {s}: offset={off:+.1f} -> {OUT_NAME}", flush=True)
    pd.DataFrame(rows).to_csv(C.OUT_ROOT / "refuse_norm_offsets.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
