"""9 — NOX1-Figuren + CSV-Export aus der v5-Kohorte (cellular/double-negative Norm).

Erzeugt in _nox1_normalization/:
  nox1_dotplot_wt_vs_mut_v5.png       (Szenen n=16 + Patient-Mittel n=8)
  nox1_hepato_distribution_v5.png     (Hepato vs Leuko NOX1, %+ Schwelle)
  nox1_representative_LXT32_vs_LXT47.png (analyse-konsistent, feste Kontrastgrenze)
  nox1_v5_per_patient.csv / nox1_v5_per_scene.csv

Run: HD_DATA_ROOT=... micromamba run -n cynif python 9_nox1_figures_v5.py
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu
from skimage.segmentation import find_boundaries
from PIL import Image, ImageDraw

D = Path(os.environ.get("HD_DATA_ROOT", "."))
O = D / "_nox1_normalization"
NOX1 = "NOX1_mean_norm"
CLIM = (150.0, 400.0)
grp = pd.read_csv(D / "sample_groups.csv").set_index("patient")["group"]
df = pd.read_parquet(O / "cell_stats_v5_gated.parquet")
off = pd.read_csv(O / "nox1_cell_offsets_v5.csv").set_index("sample")["offset"]

# ---------- Verteilung ----------
hep = df.loc[df.celltype == "hepatocyte", NOX1].dropna().to_numpy(float)
leu = df.loc[df.celltype == "leukocyte_CD45", NOX1].dropna().to_numpy(float)
leu_p95 = np.percentile(leu, 95)
fig, ax = plt.subplots(figsize=(10, 6)); lo, hi = 0, np.percentile(hep, 99.5); xs = np.linspace(lo, hi, 400)
for v, l, c in [(leu, "CD45+ Leukozyten (NOX1-neg Referenz)", "#c0392b"), (hep, "Hepatozyten", "#2980b9")]:
    k = gaussian_kde(np.clip(v, lo, hi)); ax.fill_between(xs, k(xs), alpha=.25, color=c); ax.plot(xs, k(xs), color=c, lw=2, label=l)
ax.axvline(np.median(hep), color="#2980b9", ls=":", lw=1.5, label=f"Hepato-Median={np.median(hep):.0f}")
ax.axvline(leu_p95, color="black", ls="--", lw=2, label=f"Leuko-p95 (NOX1+-Schwelle)={leu_p95:.0f} → {100*(hep>leu_p95).mean():.0f}%")
ax.set_xlabel("NOX1 (zell-normalisiert)"); ax.set_ylabel("Dichte"); ax.legend(fontsize=9)
ax.set_title("Hepatozyten vs Leukozyten NOX1 — breit & überlappend, kein sauberer +/- Split")
fig.tight_layout(); fig.savefig(O / "nox1_hepato_distribution_v5.png", dpi=150); plt.close(fig)

# ---------- Dotplot ----------
def per_scene(ct): return df[df.celltype == ct].groupby(["patient", "Sample"])[NOX1].median().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 6)); rng = np.random.RandomState(1); COL = {"WT": "#2980b9", "MUT": "#c0392b"}
for ax, ct, title in zip(axes, ["hepatocyte", "leukocyte_CD45"], ["Hepatozyten", "CD45+ Leukozyten"]):
    sc = per_scene(ct); sc["group"] = sc.patient.map(grp)
    pp = sc.groupby("patient").agg(nox1=(NOX1, "mean")).reset_index(); pp["group"] = pp.patient.map(grp)
    for gi, g in enumerate(["WT", "MUT"]):
        s2 = sc[sc.group == g]; ax.scatter(gi + (rng.rand(len(s2))-.5)*.22, s2[NOX1], s=45, color=COL[g], alpha=.35, edgecolor="none", zorder=2, label="Szenen (n=16)" if gi == 0 else None)
        p2 = pp[pp.group == g]; ax.scatter(gi + (rng.rand(len(p2))-.5)*.12, p2.nox1, s=150, color=COL[g], edgecolor="black", lw=1.1, zorder=4, label="Patient-Ø (n=8)" if gi == 0 else None)
        m = np.median(p2.nox1); ax.plot([gi-.26, gi+.26], [m, m], color=COL[g], lw=3, zorder=3)
    wt = pp[pp.group == "WT"].nox1; mut = pp[pp.group == "MUT"].nox1; p = mannwhitneyu(wt, mut, alternative="two-sided")[1]
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"WT (n={len(wt)})", f"MUT (n={len(mut)})"])
    ax.set_title(f"{title}\nMann-Whitney (Patienten) p={p:.2f}"); ax.set_ylabel("NOX1 (zell-norm., Median)")
    ax.grid(axis="y", ls=":", alpha=.4); ax.set_xlim(-.6, 1.6)
    if ct == "hepatocyte": ax.legend(fontsize=8, loc="upper right")
fig.suptitle("NOX1 WT vs MUT — kleine Punkte = Szenen (n=16), große = Patient-Mittel (n=8)")
fig.tight_layout(); fig.savefig(O / "nox1_dotplot_wt_vs_mut_v5.png", dpi=150); plt.close(fig)

# ---------- CSV ----------
sc_h = per_scene("hepatocyte").rename(columns={NOX1: "hepato_NOX1_median"})
sc_l = per_scene("leukocyte_CD45").rename(columns={NOX1: "leuko_NOX1_median"})
scene = sc_h.merge(sc_l[["Sample", "leuko_NOX1_median"]], on="Sample")
pos = [(s, 100*(g[g.celltype == "hepatocyte"][NOX1] > leu_p95).mean()) for s, g in df.groupby("Sample")]
scene = scene.merge(pd.DataFrame(pos, columns=["Sample", "hepato_NOX1pos_pct_vs_leukoP95"]), on="Sample")
scene["group"] = scene.patient.map(grp)
scene.round(1).sort_values(["group", "patient", "Sample"]).to_csv(O / "nox1_v5_per_scene.csv", index=False)
pat = scene.groupby(["patient", "group"]).agg(hepato_NOX1_median=("hepato_NOX1_median", "mean"),
    leuko_NOX1_median=("leuko_NOX1_median", "mean"),
    hepato_NOX1pos_pct=("hepato_NOX1pos_pct_vs_leukoP95", "mean")).round(1).reset_index()
pat.sort_values(["group", "patient"]).to_csv(O / "nox1_v5_per_patient.csv", index=False)

# ---------- repräsentatives Bild (analyse-konsistent) ----------
def panel(s, sz=1200):
    sd = D / f"sample_{s}"; o = float(off.get(f"sample_{s}", 0))
    img = next(p for p in [sd/"AF_removal/fused_decon_refused.ome.tif"] if p.exists())
    dapi = tifffile.imread(str(img), key=0).astype("float32")
    nox1 = tifffile.imread(str(img), key=5).astype("float32") - o
    hepm = tifffile.imread(str(sd/"segmentation_v5/hepato_cells_v5.tiff"))
    H, W = hepm.shape; r = sz//2; best = None
    for cy in range(r, H-r, 400):
        for cx in range(r, W-r, 400):
            nh = np.count_nonzero(np.unique(hepm[cy-r:cy+r, cx-r:cx+r]))
            if best is None or nh > best[0]: best = (nh, cy, cx)
    cy, cx = best[1], best[2]; sl = (slice(cy-r, cy+r), slice(cx-r, cx+r))
    d = np.clip(dapi[sl]/np.percentile(dapi[sl], 99.5), 0, 1)
    n = np.clip((nox1[sl]-CLIM[0])/(CLIM[1]-CLIM[0]), 0, 1)
    rgb = np.zeros((sz, sz, 3), np.float32); rgb[..., 2] += d*.7; rgb[..., 0] += n; rgb[..., 2] += n
    b = find_boundaries(hepm[sl], mode="inner"); rgb[..., 0][b] = 1; rgb[..., 1][b] = 1; rgb[..., 2][b] = 0
    im = Image.fromarray((np.clip(rgb, 0, 1)*255).astype("uint8"))
    ImageDraw.Draw(im).text((15, 15), f"{s}  offset {o:+.0f}, clim {int(CLIM[0])}-{int(CLIM[1])}", fill=(255, 255, 255))
    return np.array(im)
combo = np.concatenate([panel("LXT32_scene_1"), np.full((1200, 8, 3), 255, "uint8"), panel("LXT47_scene_1")], axis=1)
Image.fromarray(combo).save(O / "nox1_representative_LXT32_vs_LXT47.png")

print("Figuren + CSV neu erzeugt in", O)
print(pat.sort_values(["group", "patient"]).to_string(index=False))
