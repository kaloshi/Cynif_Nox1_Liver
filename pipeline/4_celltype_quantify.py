"""Pipeline 4 - cell typing + hepatocyte NOX1 quantification on the segmentation_v4 masks with a
data-driven hepatocyte gate.

Cell typing: CD45+ = membrane-ring median CD45 > Otsu. Hepatocyte = ASGR1+ & CD45-. ASGR1 is a
graded marker (no clean bimodal split), so the ASGR1+ threshold per sample is the 95th percentile of
ASGR1 in the CD45+ (ASGR1-negative) population, clamped to the [5th, 15th] percentile of that
sample's ASGR1 distribution to keep it stable (-> ~75% hepatocytes).

NOX1 is cross-sample-normalized analytically with the ECM offset from pipeline/2 (additive; exact for
the per-cell mean). Reported readout: the per-sample / per-patient median hepatocyte NOX1.

Run order: pipeline 1 -> 2 -> 3 -> 4.
Output: <sample>/segmentation_v4/cell_stats_v4_gated.csv ; OUT_ROOT/cell_stats_v4_gated.parquet,
        nox1_v4_per_{sample,patient}.csv
"""
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from hdcycif import config as C

CELLSTATS = "segmentation_v4/cell_stats_v4.csv"
CD45_NEG_PCTILE = 95
# ASGR1+ threshold = p95 of ASGR1 in CD45+ cells, clamped to [lo, hi] percentiles of the sample's
# ASGR1 distribution. [5, 15] = inclusive (most ASGR1+ tissue counts as hepatocyte, ~75%); raise
# for stricter (e.g. [10, 25] -> ~70%, [25, 50] -> ~56%).
CLAMP_PCT = (5, 15)
NOX1_COLS = ["NOX1_mean", "NOX1_mem_mean", "NOX1_mem_median"]


def main():
    offs = pd.read_csv(C.OUT_ROOT / "refuse_norm_offsets.csv").set_index("sample")["offset"]
    parts = []
    for s in C.list_samples():
        cs = pd.read_csv(C.DATA_ROOT / s / CELLSTATS)
        off = float(offs[s])
        for col in NOX1_COLS:
            if col in cs:
                cs[col + "_norm"] = (cs[col] - off).clip(lower=0)   # cross-sample additive norm

        a = cs["ASGR1_mem_median"].to_numpy(float)
        cd = cs["CD45_mem_median"].to_numpy(float)
        af = a[np.isfinite(a)]
        ct = threshold_otsu(cd[np.isfinite(cd)])
        cs["is_cd45"] = np.isfinite(cd) & (cd > ct)
        neg = a[np.isfinite(a) & (cd > ct)]              # ASGR1 in CD45+ (ASGR1-negative) cells
        raw = float(np.percentile(neg, CD45_NEG_PCTILE)) if neg.size > 20 else threshold_otsu(af)
        # robust clamp to the lower-middle of the ASGR1 distribution -> prevents per-sample blow-ups
        # (e.g. when the CD45+ reference is contaminated; LXT41_s2 went 160 a.u./6% -> stable)
        lo, hi = np.percentile(af, CLAMP_PCT)
        a_thr = float(np.clip(raw, lo, hi))
        cs["is_asgr1"] = np.isfinite(a) & (a > a_thr)
        cs["is_hepato"] = cs["is_asgr1"] & ~cs["is_cd45"]
        cs["celltype"] = np.where(cs["is_hepato"], "hepatocyte",
                          np.where(cs["is_cd45"] & ~cs["is_asgr1"], "leukocyte_CD45", "other"))
        cs["patient"] = C.patient_of(s)
        cs.to_csv(C.DATA_ROOT / s / "segmentation_v4" / "cell_stats_v4_gated.csv", index=False)
        parts.append(cs)
        nh = int(cs["is_hepato"].sum())
        print(f"  {s}: n={len(cs)} hepato={nh} ({100*nh/len(cs):.0f}%) "
              f"leuko={int((cs['celltype']=='leukocyte_CD45').sum())} | ASGR1_thr={a_thr:.0f} CD45_thr={ct:.0f}", flush=True)

    allc = pd.concat(parts, ignore_index=True)
    allc.to_parquet(C.OUT_ROOT / "cell_stats_v4_gated.parquet", index=False)

    sub = allc[allc["celltype"].isin(["hepatocyte", "leukocyte_CD45"])].copy()

    def summ(df, keys):
        return df.groupby(keys).agg(n_cells=("CellID", "size"),
                                    nox1_median=("NOX1_mean_norm", "median"),
                                    nox1_mean=("NOX1_mean_norm", "mean")).reset_index().round(2)

    per_sample = summ(sub, ["patient", "Sample", "celltype"]).sort_values(["celltype", "Sample"])
    per_patient = summ(sub, ["patient", "celltype"]).sort_values(["celltype", "patient"])
    per_sample.to_csv(C.OUT_ROOT / "nox1_v4_per_sample.csv", index=False)
    per_patient.to_csv(C.OUT_ROOT / "nox1_v4_per_patient.csv", index=False)
    print("\n=== Per patient (segmentation_v4, CD45-anchored hepatocyte gate, normalized NOX1) ===")
    print(per_patient.to_string(index=False))

    hep = allc[allc["celltype"] == "hepatocyte"]
    frac = 100 * (allc["celltype"] == "hepatocyte").mean()
    hp = per_patient[per_patient["celltype"] == "hepatocyte"]["nox1_median"].to_numpy()
    print(f"\n  hepatocytes: {len(hep)} cells ({frac:.0f}% of all) | per-patient NOX1 "
          f"fold={hp.max()/max(hp.min(),1e-6):.2f}x CV={hp.std()/hp.mean():.3f}")


if __name__ == "__main__":
    main()
