# HD CyCIF NOX1 analysis pipeline

Image-analysis code for quantifying **NOX1** in **hepatocytes** from single-cycle 6-plex cyclic
immunofluorescence (CyCIF) liver imaging: illumination re-correction / de-seaming → autofluorescence
removal → **Cellpose-SAM segmentation (GPU)** → intensity-based cell typing → double-negative NOX1
normalization → per-patient median NOX1 and review figures.

See [`METHODS.md`](METHODS.md) for the manuscript methods text with all parameters and versions.

## Pipeline (current, v5)

| Step | Script | Does | Key output (per sample, under `<HD_DATA_ROOT>`) |
|---|---|---|---|
| 1 | `pipeline/1_deseam_refuse.py` | Per-sample BaSiC flat-/dark-field on raw tiles, tile-position detection, overlap balancing, feathered re-fusion of all 6 channels; then NNLS-v3 autofluorescence removal | `<sample>/AF_removal/fused_decon_refused.ome.tif`, `NOX1_refused.tif` |
| 2 | `pipeline/3b_segment_cellpose_v5.py` | **Cellpose-SAM** nuclei (GPU) → over-merge split → tissue-bounded cell bodies (grow ≈8 µm) → DNA-gate + min-area → compartments; per-cell 6-channel means + SPARQ local contrast | `<sample>/segmentation_v5/{nuclei,whole_cell_label,matched_cytoplasm,membrane_ring}_v5.tiff`, `cell_stats_v5.csv` |
| 3 | `pipeline/4b_celltype_quantify_v5.py` | Cell typing by intensity mutual-anchoring: hepatocyte (ASGR1⁺ CD45⁻), CD45⁺ leukocyte, double-negative ("other") | `<sample>/segmentation_v5/cell_stats_v5_gated.csv`, per-type label TIFFs, `_nox1_normalization/cell_stats_v5_gated.parquet` |
| 4 | `pipeline/2b_nox1_cellnorm_v5.py` | Additive cross-sample NOX1 normalization to the **double-negative cellular background** (replaces the ECM anchor) | overwrites `NOX1_mean_norm` in the gated CSV/parquet, `_nox1_normalization/nox1_cell_offsets_v5.csv` |
| 5 | `pipeline/7_validate_v5.py` | Visual QC crops (ASGR1+hepatocyte / CD45+leukocyte) and per-sample metrics | `<sample>/segmentation_v5/qc/`, `qc/metrics.csv` |
| 6 | `pipeline/8_nox1_positivity_v5.py` | Median NOX1 and NOX1⁺ fraction per cell type / patient | `_nox1_normalization/nox1_v5_positivity_per_{sample,patient}.csv` |
| 7 | `pipeline/9_nox1_figures_v5.py` | WT-vs-MUT dot plot, distribution, per-patient/scene CSV and multi-sheet Excel | `_nox1_normalization/nox1_dotplot_wt_vs_mut_v5.png`, `NOX1_v5_results.xlsx`, `nox1_v5_per_{patient,scene}.csv` |

Shared helpers live in the importable package `hdcycif/` (`config`, `tiles`, `af_removal`,
`segmentation`). Step 1 imports only from `hdcycif`. The v5 segmentation/typing scripts additionally
import the **`cynif`** package (`segmentation`, `features`, `utils`) from the companion project
repository — put it on the `PYTHONPATH` before running steps 2–3.

> The original **v4** scripts (`pipeline/{2_crosssample_norm,3_segment,4_celltype_quantify,5_export_excel,6_scene_figures}.py`)
> are retained for reference. They used ECM-anchored normalization and watershed segmentation and are
> **superseded** by the v5 scripts above; step 1 (de-seaming) is shared by both.

## Installation

The published v5 run uses two environments:

- **De-seaming** (`pipeline/1`): BaSiCPy 2.0.0 / JAX 0.10.1 in a separate environment (BaSiCPy and
  the GPU/Cellpose stack below do not coexist cleanly).
- **Segmentation, typing, analysis** (`pipeline/3b,4b,2b,7,8,9`): Cellpose 4.1.1 (Cellpose-SAM) on
  **CUDA-enabled PyTorch 2.12**, plus scikit-image, scikit-learn, SciPy, NumPy, pandas, Matplotlib,
  tifffile (the `cynif` conda env, Python 3.11, run on an NVIDIA RTX PRO 5000 GPU).

Exact versions are pinned in `requirements.txt`. A CUDA-enabled PyTorch build (matching your GPU/
driver) and the `cynif` package must be installed separately — they are not pip-pinnable here.

## Running

Set `HD_DATA_ROOT` to the dataset root (the folder containing the per-scene `sample_*` directories)
and run the steps in order:

```bash
export HD_DATA_ROOT=/path/to/HD          # Windows PowerShell:  $env:HD_DATA_ROOT = "D:\CycIF\HD"

# de-seaming (BaSiCPy env):
python pipeline/1_deseam_refuse.py

# segmentation → typing → NOX1 normalization → QC → readout (Cellpose-SAM / GPU env):
python pipeline/3b_segment_cellpose_v5.py
python pipeline/4b_celltype_quantify_v5.py
python pipeline/2b_nox1_cellnorm_v5.py
python pipeline/7_validate_v5.py
python pipeline/8_nox1_positivity_v5.py
python pipeline/9_nox1_figures_v5.py
```

Optional environment variables: `HD_SAMPLES` (comma-separated subset), `HD_SEG_VERSION`, and the
figure knobs used by the individual scripts. Outputs are written under `<HD_DATA_ROOT>` as listed
above.

For the group labels (MUT/WT), the figure/readout scripts read `<HD_DATA_ROOT>/sample_groups.csv`
(columns `patient,group`; see `sample_groups.example.csv`). This mapping is study metadata and is
**not** part of the repository.

### Inputs
Step 1 reads the raw exported single tiles (`<sample>/cyc001/Z-Stacks/fileseries_export/z01/tiles/`)
and uses the upstream DAPI mosaic (`<sample>/AF_removal/…nnls_v3.ome.tif`) only as the coordinate
frame for tile-position detection. The v5 segmentation and typing run on the de-seamed
`fused_decon_refused.ome.tif`; NOX1 normalization uses the cell-type calls (double-negative cells),
not an ECM mask. Upstream products are generated by the acquisition pipeline (Cycif_pipeline_V3,
referenced), not by this repository.

## Data availability
This repository contains **code only**. Raw and processed imaging data are not included (size and
patient confidentiality) and are available from the authors on reasonable request.

## Notes
- Segmentation uses **Cellpose-SAM** on a GPU (`pipeline/3b`) to capture dim and densely packed
  immune nuclei; cell bodies are grown within a tissue mask so the ASGR1 membrane is captured for
  hepatocyte typing.
- Upstream acquisition and preprocessing (illumination, deconvolution, EDF, stitching, and the
  original AF removal) were performed with an in-house pipeline (Cycif_pipeline_V3) and are cited,
  not reproduced here.

## Citation
See [`CITATION.cff`](CITATION.cff). Please cite this repository and the associated publication.

## License
[MIT](LICENSE).
