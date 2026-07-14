# Methods — HD CyCIF NOX1 image analysis

Formalin-fixed liver tissue from eight patients was imaged over two scenes each (n = 16 imaged
regions), forming a mutant (MUT: LXT40, LXT47, LXT52) and a wild-type (WT: LXT31, LXT32, LXT38,
LXT41, LXT48) group. Single-cycle six-plex cyclic immunofluorescence (CyCIF)¹ acquired, in order,
DAPI, AF1 (Atto490LS), an autofluorescence channel (AF2), ASGR1 (ATTO532), CD45 (SO) and NOX1
(ATTO643) at 0.325 µm per pixel, as a 5 × 5 grid of 2,048-pixel tiles with 10 % overlap and three
z-planes; the middle focal plane was used. Raw tile export, per-channel deconvolution,
extended-depth-of-field projection and stitching were performed with an in-house pipeline, and all
subsequent analysis was run on CPU.

Because the original per-tile flat-field correction had been fitted per sample without a dark-field
term — leaving an additive per-tile background that appeared as tile seams — the raw tiles were
re-corrected. For each sample and channel a BaSiC² model was fitted with dark-field estimation
enabled (flat-field smoothness 1.0) on 4×-downsampled tiles, and every tile was corrected as
(raw − dark-field)/flat-field. Each tile's position was recovered by normalized cross-correlation
template matching³ of the single DAPI tiles against the mosaic (coarse eightfold, then local
twofold); residual per-tile additive offsets were removed by weighted least squares matching the
median of every overlapping tile pair (constrained to zero mean); and tiles were combined by
feathered blending, re-fusing all six channels. Autofluorescence was then removed from each marker
by non-negative least squares: AF1 and AF2 were Gaussian-smoothed (σ = 2), a tissue mask was taken as
AF2 above its 15th percentile, coefficients were estimated on the lowest 60 % of tissue-pixel
intensities (AF2 first, then AF1 on the residual) and capped at the smaller of the fit, a
99.9-percentile intensity ratio and 2.0; the smoothed donors were subtracted with clipping at zero,
and a second pass was applied where the residual correlation with autofluorescence exceeded 0.3.
NOX1 was equalized across samples by subtracting, per sample, the median NOX1 of extracellular
(non-cell) regions minus the cohort median of that quantity.

Nuclei were segmented on the de-seamed DAPI (Gaussian σ = 1) by thresholding at 0.6 × the Otsu
level⁴ and distance-transform watershed³ (peak minimum distance 7 px, minimum area 20 px). Whole
cells were obtained by a tissue-filling Voronoi watershed that grew each nucleus within a tissue mask
(nuclei ∪ ASGR1 above 0.4 × Otsu, morphologically closed and hole-filled), with the per-nucleus
radius capped at 32 px, and a 4-px membrane ring was derived by erosion; the median cell diameter was
≈ 18 µm. Whole-cell mean and membrane-ring statistics were recorded per cell for all six channels.
CD45⁺ leukocytes were defined by an Otsu threshold on the membrane-ring median CD45. Hepatocytes were
defined as ASGR1⁺ and CD45⁻, where the per-sample ASGR1⁺ threshold was the 95th percentile of ASGR1
(membrane-ring median) within the CD45⁺ (ASGR1-negative) population, clamped to the 5th–15th
percentile of that sample's ASGR1 distribution for stability; this identified 285,824 hepatocytes
(75 % of segmented cells). Per-cell NOX1 was the whole-cell mean of the de-seamed,
cross-sample-normalized channel, and the reported readout was the median hepatocyte NOX1 per sample
and per patient, with CD45⁺ leukocytes as a negative-population reference; the patient (n = 8; MUT
versus WT) was the unit of comparison.

Analysis used Python 3.10.18 with BaSiCPy 1.2.0 (JAX 0.4.23), scikit-image 0.25.2³, SciPy 1.12.0⁵,
NumPy 1.26.4⁶, pandas 2.3.0⁷, Matplotlib 3.10.3⁸, tifffile 2023.2.28, Pillow 11.2.1 and openpyxl
3.1.5, entirely on CPU. The complete analysis code and exact package versions are available at
https://github.com/kaloshi/Cynif_Nox1_Liver. Portions of the analysis code were prepared with the
assistance of a large language model (Claude, Anthropic); the study design, all parameters and the
results were defined and verified by the authors, who take full responsibility for the work.

## References
1. Lin, J.-R. et al. Highly multiplexed immunofluorescence imaging of human tissues and tumors using t-CyCIF and conventional optical microscopes. *eLife* **7**, e31657 (2018).
2. Peng, T. et al. A BaSiC tool for background and shading correction of optical microscopy images. *Nat. Commun.* **8**, 14836 (2017).
3. van der Walt, S. et al. scikit-image: image processing in Python. *PeerJ* **2**, e453 (2014).
4. Otsu, N. A threshold selection method from gray-level histograms. *IEEE Trans. Syst. Man Cybern.* **9**, 62–66 (1979).
5. Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat. Methods* **17**, 261–272 (2020).
6. Harris, C. R. et al. Array programming with NumPy. *Nature* **585**, 357–362 (2020).
7. McKinney, W. Data structures for statistical computing in Python. In *Proc. 9th Python in Science Conf.* 56–61 (2010).
8. Hunter, J. D. Matplotlib: a 2D graphics environment. *Comput. Sci. Eng.* **9**, 90–95 (2007).
