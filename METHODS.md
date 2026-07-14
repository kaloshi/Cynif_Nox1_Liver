# Methods — HD CyCIF NOX1 image analysis

All analysis code is in this repository (`hdcycif/` helper package, `pipeline/1…6` run in order);
module and parameter names below refer to it.

## Samples and imaging
Formalin-fixed liver tissue from 8 patients (2 scenes each; n = 16 imaged regions) was analysed,
comprising a **MUT** group (LXT40, LXT47, LXT52) and a **WT** group (LXT31, LXT32, LXT38, LXT41,
LXT48). Single-cycle 6-plex cyclic immunofluorescence (CyCIF) imaged, in order, DAPI, AF1
(Atto490LS), an autofluorescence channel (AF2), ASGR1 (ATTO532), CD45 (SO) and NOX1 (ATTO643) at
0.325 µm/pixel, as a 5 × 5 grid of 2048-pixel tiles with 10 % overlap and 3 z-planes. Raw tile
export, per-channel deconvolution, extended-depth-of-field projection and stitching were performed
with an in-house pipeline (Cycif_pipeline_V3); the middle focal plane (z01) was used here. All
subsequent steps were run on CPU with the software listed below.

## Illumination re-correction and re-fusion (`pipeline/1_deseam_refuse.py`)
The original per-tile flat-field correction had been fitted per sample without a dark-field term,
leaving an uncorrected additive per-tile background that appeared as tile seams in the fused mosaics.
We therefore re-corrected the raw tiles: for every sample and channel a **BaSiC** model (BaSiCPy)
was fitted on the raw tiles (4× block-mean-downsampled) with `get_darkfield=True` and
`smoothness_flatfield=1.0`, and each tile was corrected as `(raw − darkfield) / flatfield`. The exact
position of each tile in the mosaic was recovered by normalized cross-correlation template matching
of the single DAPI tiles against the mosaic (coarse 8× then local fine 2×; `hdcycif.tiles`).
Residual per-tile additive offsets were removed by weighted least squares matching the median
intensity of every pair of overlapping tiles (constrained to zero mean), and tiles were combined by
linear-ramp (feathered) blending. All six channels were re-fused on the mosaic coordinate frame.

## Autofluorescence removal (`hdcycif/af_removal.py`, NNLS v3)
On the de-seamed channels, AF1 and AF2 (Gaussian-smoothed, σ = 2) served as autofluorescence donors.
A tissue mask was defined as AF2 above its 15th percentile (holes filled). For each marker (ASGR1,
CD45, NOX1), non-negative least-squares coefficients were estimated on the lower 60 % of tissue-pixel
intensities (AF2 first, then AF1 on the residual); each coefficient was capped at the smaller of the
NNLS estimate, a 99.9-percentile intensity ratio, and 2.0. The donors were subtracted globally with
clipping at zero, and a second identical pass was applied when the residual Pearson correlation with
autofluorescence exceeded 0.3.

## Cross-sample normalization (`pipeline/2_crosssample_norm.py`)
NOX1 background was equalized across samples by an additive offset equal to the median NOX1 in
extracellular-matrix (non-cell) regions of a sample minus the cohort median of those values; the
offset was subtracted (clipped at zero). This robust background anchor was preferred over a
per-frame BaSiC baseline, which diverged on one sample.

## Segmentation (`pipeline/3_segment.py`)
Nuclei were segmented on the de-seamed DAPI (Gaussian σ = 1) by thresholding at 0.6 × Otsu and
distance-transform watershed (peak minimum distance 7 px, minimum object area 20 px). Whole cells
were obtained by a tissue-filling Voronoi watershed that grew each nucleus within a tissue mask
(nuclei ∪ ASGR1 > 0.4 × Otsu, morphologically closed and hole-filled), with the per-nucleus radius
capped at 32 px; a 4-px membrane ring was derived by erosion (`hdcycif.segmentation`). The median
whole-cell diameter was ≈ 18 µm. Per cell, whole-cell mean and membrane-ring mean/median were
recorded for all six channels.

## Cell typing (`pipeline/4_celltype_quantify.py`)
Cells were classified from the de-seamed, background-normalized markers. CD45⁺ leukocytes were
defined by an Otsu threshold on the membrane-ring median CD45. **Hepatocytes were defined as ASGR1⁺
and CD45⁻**; the per-sample ASGR1⁺ threshold was the 95th percentile of ASGR1 (membrane-ring median)
within the CD45⁺ (ASGR1-negative) population, clamped to the 5th–15th percentile of that sample's
ASGR1 distribution to prevent unstable thresholds. This identified 285 824 hepatocytes (75 % of all
segmented cells).

## NOX1 quantification (`pipeline/4`, exported by `pipeline/5_export_excel.py`)
Per-cell NOX1 was the whole-cell mean of the de-seamed, cross-sample-normalized NOX1 channel. The
primary readout was the **median hepatocyte NOX1 per sample and per patient**; CD45⁺ leukocytes were
reported as a negative-population reference. Per-scene review figures (DAPI+NOX1, NOX1, NOX1 with the
hepatocyte mask, ASGR1) were produced by `pipeline/6_scene_figures.py`. The biological unit for
comparison was the patient (n = 8; MUT vs WT).

## Software
Python 3.10.18 with BaSiCPy 1.2.0 (JAX 0.4.23), scikit-image 0.25.2, scikit-learn 1.7.0, SciPy
1.12.0, tifffile 2023.2.28, NumPy 1.26.4, pandas 2.3.0, Matplotlib 3.10.3, Pillow 11.2.1 and
openpyxl 3.1.5. All steps ran on CPU. Code and exact versions are available at &lt;REPO_URL&gt;.
