"""8 — NOX1 evaluation on segmentation_v5 celltypes: per-cell-type median NOX1 and
NOX1+ fraction (per-celltype GMM cutoff), per-sample/patient tables + NOX1+ masks.

Mirrors nox1_norm/step18_nox1_positivity_v4.py but on the corrected v5 celltypes.
NOX1 = NOX1_mean_norm (cross-sample additive-normalized whole-cell mean). The
NOX1+ threshold is fit WITHIN each cell type (hepatocyte vs leukocyte_CD45), not
globally, so a hepatocyte-dominated cutoff can't swamp the leukocytes.

Outputs (_nox1_normalization/): nox1_v5_positivity_per_{sample,patient}.csv,
  nox1_v5_distribution.png, cell_stats_v5_positivity.parquet ;
  per sample (segmentation_v5/): nox1pos_hepato_v5.tiff, nox1pos_cd45_v5.tiff.

Run:
  HD_DATA_ROOT=/path/to/HD \
    micromamba run -n cynif python 8_nox1_positivity_v5.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from sklearn.mixture import GaussianMixture

DATA_ROOT = Path(os.environ.get("HD_DATA_ROOT", "."))
OUT_ROOT = DATA_ROOT / "_nox1_normalization"
SEG_DIR = "segmentation_v5"
WC = f"{SEG_DIR}/whole_cell_label_v5.tiff"
NOX1 = "NOX1_mean_norm"
CTYPES = ["hepatocyte", "leukocyte_CD45"]
THRESH = ("sd", -1.0)     # low_mean + f*low_sd ; f<0 = looser (more NOX1+). Or ("crossover",).


def fit_threshold(vals):
    v = vals.clip(upper=float(vals.quantile(0.995))).to_numpy(float).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(v)
    mu = gm.means_.ravel(); sd = np.sqrt(gm.covariances_.ravel())
    i = np.argsort(mu); mu, sd = mu[i], sd[i]
    if THRESH[0] == "crossover":
        xs = np.linspace(vals.min(), vals.quantile(0.995), 4000).reshape(-1, 1)
        pr = gm.predict(xs); ch = np.where(np.diff(pr) != 0)[0]
        return float(xs[ch[0], 0]) if len(ch) else float(mu.mean())
    return float(mu[0] + THRESH[1] * sd[0])


def build_lut(cell_ids, max_id):
    lut = np.zeros(max_id + 1, dtype=np.uint32)
    ids = np.asarray([c for c in cell_ids if 0 < c <= max_id], dtype=np.int64)
    lut[ids] = ids.astype(np.uint32)
    return lut


def summ(g):
    out = []
    for ct in CTYPES:
        s = g[g["celltype"] == ct]; n = len(s)
        out.append(dict(celltype=ct, n_cells=n,
                        nox1_median=round(s[NOX1].median(), 2) if n else np.nan,
                        nox1_mean=round(s[NOX1].mean(), 2) if n else np.nan,
                        nox1_z_mean=round(s["NOX1_z"].mean(), 3) if n else np.nan,
                        nox1_pos_n=int(s["nox1_positive"].sum()),
                        nox1_pos_pct=round(100 * s["nox1_positive"].mean(), 1) if n else np.nan))
    return out


def main():
    df = pd.read_parquet(OUT_ROOT / "cell_stats_v5_gated.parquet")
    if NOX1 not in df:
        raise SystemExit(f"{NOX1} missing — run 4b gating first")
    thr = {}
    for ct in CTYPES:
        vals = df.loc[df["celltype"] == ct, NOX1].dropna()
        if len(vals) < 50:
            thr[ct] = np.inf
            continue
        thr[ct] = fit_threshold(vals)
        m = df["celltype"] == ct
        mu_ct, sd_ct = df.loc[m, NOX1].mean(), df.loc[m, NOX1].std() or 1.0
        df.loc[m, "NOX1_z"] = (df.loc[m, NOX1] - mu_ct) / sd_ct
        df.loc[m, "nox1_positive"] = df.loc[m, NOX1] > thr[ct]
        print(f"  {ct}: thr={thr[ct]:.0f}  median={vals.median():.0f}  "
              f"NOX1+={100*(vals>thr[ct]).mean():.0f}%  (n={len(vals)})", flush=True)
    df["nox1_positive"] = df.get("nox1_positive", False).fillna(False)

    df.to_parquet(OUT_ROOT / "cell_stats_v5_positivity.parquet", index=False)

    ps = pd.DataFrame([dict(patient=p, Sample=s, **r)
                       for (p, s), g in df.groupby(["patient", "Sample"]) for r in summ(g)])
    pp = pd.DataFrame([dict(patient=p, **r) for p, g in df.groupby("patient") for r in summ(g)])
    ps.sort_values(["celltype", "Sample"]).to_csv(OUT_ROOT / "nox1_v5_positivity_per_sample.csv", index=False)
    pp.sort_values(["celltype", "patient"]).to_csv(OUT_ROOT / "nox1_v5_positivity_per_patient.csv", index=False)
    print("\n=== Per patient (v5): NOX1 median / z / %+ ===")
    print(pp.sort_values(["celltype", "patient"]).to_string(index=False), flush=True)

    # optional distribution figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for a, ct in zip(ax, CTYPES):
            vals = df.loc[df["celltype"] == ct, NOX1].dropna()
            if len(vals) < 50:
                continue
            lo, hi = vals.quantile(0.005), vals.quantile(0.995)
            xs = np.linspace(lo, hi, 500); ys = gaussian_kde(vals.clip(lo, hi))(xs)
            a.fill_between(xs, ys, alpha=0.2); a.plot(xs, ys, lw=1.8)
            a.axvline(thr[ct], color="crimson", ls="--", lw=2,
                      label=f"thr={thr[ct]:.0f} -> {100*(vals>thr[ct]).mean():.0f}%+")
            a.set_title(f"{ct} (median {vals.median():.0f})"); a.legend(fontsize=8)
            a.set_xlabel("NOX1_mean_norm")
        fig.suptitle("NOX1+ cutoff per cell type (v5)")
        fig.tight_layout(); fig.savefig(OUT_ROOT / "nox1_v5_distribution.png", dpi=140)
        plt.close(fig)
        print("wrote nox1_v5_distribution.png", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(distribution figure skipped: {e})", flush=True)

    # per-sample NOX1+ label masks
    for sample in sorted(df["Sample"].unique()):
        sdf = df[df["Sample"] == sample]
        wcp = DATA_ROOT / sample / WC
        if not wcp.exists():
            continue
        wc = tifffile.imread(str(wcp)).astype(np.int64)
        mx = int(wc.max())
        hep_pos = sdf.loc[(sdf.celltype == "hepatocyte") & sdf.nox1_positive, "CellID"].astype(int)
        cd_pos = sdf.loc[(sdf.celltype == "leukocyte_CD45") & sdf.nox1_positive, "CellID"].astype(int)
        seg = DATA_ROOT / sample / SEG_DIR
        tifffile.imwrite(str(seg / "nox1pos_hepato_v5.tiff"), build_lut(hep_pos, mx)[wc], compression="zlib")
        tifffile.imwrite(str(seg / "nox1pos_cd45_v5.tiff"), build_lut(cd_pos, mx)[wc], compression="zlib")
        print(f"  {sample}: NOX1+ hepato={len(hep_pos)} | NOX1+ CD45={len(cd_pos)}", flush=True)


if __name__ == "__main__":
    main()
