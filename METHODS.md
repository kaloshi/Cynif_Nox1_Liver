# Methods — HD CyCIF NOX1 image analysis

Formalin-fixed liver tissue from eight patients was imaged over two scenes each (n = 16 imaged
regions), forming a mutant (MUT: LXT40, LXT47, LXT52) and a wild-type (WT: LXT31, LXT32, LXT38,
LXT41, LXT48) group. Single-cycle six-plex cyclic immunofluorescence (CyCIF)¹ acquired, in order,
DAPI, AF1 (Atto490LS), an autofluorescence channel (AF2), ASGR1 (ATTO532), CD45 (SO) and NOX1
(ATTO643) at 0.26 µm per pixel, as a 5 × 5 grid of 2,048-pixel tiles with ~10 % overlap and three
z-planes spaced 2 µm apart; the middle focal plane was used. Raw tile export, per-channel deconvolution,
extended-depth-of-field projection and stitching were performed with an in-house pipeline.

Because the original per-tile flat-field correction had been fitted per sample without a dark-field
term — leaving an additive per-tile background that appeared as tile seams — the raw tiles were
re-corrected. For each sample and channel a BaSiC² model was fitted with dark-field estimation
enabled (flat-field smoothness 1.0) on 4×-downsampled tiles, and every tile was corrected as
(raw − dark-field)/flat-field. Each tile's position was recovered by normalized cross-correlation
template matching³ of the single DAPI tiles against the mosaic (coarse eightfold, then local
twofold); residual per-tile additive offsets were removed by weighted least squares matching the
median of every overlapping tile pair (constrained to zero mean); and tiles were combined by
feathered blending, re-fusing all six channels. Autofluorescence was then removed from each marker by non-negative least
squares: AF1 and AF2 were Gaussian-smoothed (σ = 2), a tissue mask was taken as AF2 above its 15th
percentile, coefficients were estimated on the lowest 60 % of tissue-pixel intensities (AF2 first,
then AF1 on the residual) and capped at the smaller of the fit, a 99.9-percentile intensity ratio
and 2.0; the smoothed donors were subtracted with clipping at zero, and a second pass was applied
where the residual correlation with autofluorescence exceeded 0.3.

Nuclei were segmented on the de-seamed DAPI channel with Cellpose-SAM⁵,⁶ (the generalist "cpsam"
model; flow threshold 0.6, per-tile normalization block 128 px, automatic diameter) on an NVIDIA
GPU, and over-merged nuclei were split by a distance-based watershed refinement (minimum peak
distance 9 px). Whole cells were formed by growing each nucleus up to 31 px (≈ 8 µm) within a tissue
mask (nuclei ∪ ASGR1 above 0.4 × Otsu⁴, morphologically closed and hole-filled), one cell per
nucleus; cells whose nuclear DAPI fell below 1.1 × the 20th image percentile or whose area was below
120 px were dropped, and a 4-px membrane ring plus a matched cytoplasm compartment were derived.
Whole-cell and cytoplasm mean intensities were recorded per cell for all six channels. Cell types were
then assigned per sample from marker intensities by mutual anchoring. ASGR1⁺ cells were identified
by a two-component Gaussian mixture⁷ on log1p(cytoplasmic ASGR1), taking the bright component with
its positive fraction constrained to 0.45–0.75 (percentile fallback otherwise); CD45⁺ cells were
identified by a threshold at the hepatocyte-pedestal median plus five median absolute deviations,
raised to the Gaussian-mixture valley of the ASGR1-negative cells where their CD45 distribution was
clearly bimodal. Double-positive cells were resolved to the marker with the larger relative distance
above its threshold. Cells were labelled hepatocyte (ASGR1⁺ CD45⁻), CD45⁺ leukocyte, or
double-negative ("other").

NOX1 was equalized across samples by subtracting, per sample, the median NOX1 of the double-negative
(ASGR1⁻ CD45⁻) cells minus the cohort median of that quantity (falling back to all non-hepatocyte
cells where fewer than 200 double-negative cells were present), with clipping at zero. The
double-negative population was used as the NOX1-negative reference (leukocytes carry their own
NADPH-oxidase activity); the alignment was assessed by scene concordance (the two scenes of a patient
are biological replicates) and a leukocyte-reference cross-check. Per-cell NOX1 was the whole-cell
mean of the de-seamed, AF-removed and cross-sample-normalized channel, and the reported readout was
the median hepatocyte NOX1 per sample and per patient, with CD45⁺ leukocytes as a reference
population; the continuous median was used rather than NOX1⁺ fractions. The patient (n = 8; MUT
versus WT) was the unit of comparison, and groups were compared by the two-sided Mann–Whitney U test.

De-seaming used Python 3.11 with BaSiCPy 2.0.0 (JAX 0.10.1). Segmentation, cell typing and downstream
analysis used Python 3.11 with Cellpose 4.1.1 (Cellpose-SAM) on PyTorch 2.12 (CUDA 13.0, NVIDIA RTX
PRO 5000 Blackwell GPU), scikit-image 0.26.0³, scikit-learn 1.9.0⁷, SciPy 1.17.1⁸, NumPy 2.4.6⁹,
pandas 2.3.3¹⁰, Matplotlib 3.10.9¹¹, tifffile 2026.3.3 and openpyxl. The complete analysis code and
exact package versions are available at https://github.com/kaloshi/Cynif_Nox1_Liver. Portions of the
analysis code were prepared with the assistance of a large language model (Claude, Anthropic); the
study design, all parameters and the results were defined and verified by the authors, who take full
responsibility for the work.

## References
1. Lin, J.-R. et al. Highly multiplexed immunofluorescence imaging of human tissues and tumors using t-CyCIF and conventional optical microscopes. *eLife* **7**, e31657 (2018).
2. Peng, T. et al. A BaSiC tool for background and shading correction of optical microscopy images. *Nat. Commun.* **8**, 14836 (2017).
3. van der Walt, S. et al. scikit-image: image processing in Python. *PeerJ* **2**, e453 (2014).
4. Otsu, N. A threshold selection method from gray-level histograms. *IEEE Trans. Syst. Man Cybern.* **9**, 62–66 (1979).
5. Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. *Nat. Methods* **18**, 100–106 (2021).
6. Pachitariu, M., Rariden, M. & Stringer, C. Cellpose-SAM: superhuman generalization for cellular segmentation. *bioRxiv* (2025). doi:10.1101/2025.04.28.651001.
7. Pedregosa, F. et al. Scikit-learn: machine learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).
8. Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat. Methods* **17**, 261–272 (2020).
9. Harris, C. R. et al. Array programming with NumPy. *Nature* **585**, 357–362 (2020).
10. McKinney, W. Data structures for statistical computing in Python. In *Proc. 9th Python in Science Conf.* 56–61 (2010).
11. Hunter, J. D. Matplotlib: a 2D graphics environment. *Comput. Sci. Eng.* **9**, 90–95 (2007).
