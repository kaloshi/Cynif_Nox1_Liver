"""Dataset paths, channel map and loaders for the HD CyCIF NOX1 pipeline.

The dataset root is taken from the ``HD_DATA_ROOT`` environment variable and must point to the
folder that contains the per-scene ``sample_*`` directories, e.g.::

    export HD_DATA_ROOT=/path/to/HD            # Linux/macOS
    $env:HD_DATA_ROOT = "D:\\CycIF\\HD"        # Windows PowerShell

Channel order of the fused 6-plex mosaics (single cycle):
    0 DAPI, 1 AF1, 2 AF2, 3 ASGR1, 4 CD45, 5 NOX1
"""
import os
from pathlib import Path
import numpy as np
import tifffile as tiff

DATA_ROOT = Path(os.environ.get("HD_DATA_ROOT", ".")).resolve()
OUT_ROOT = DATA_ROOT / "_nox1_normalization"
if os.environ.get("HD_DATA_ROOT"):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

CH = {"DAPI": 0, "AF1": 1, "AF2": 2, "ASGR1": 3, "CD45": 4, "NOX1": 5}

# Relative paths inside each <sample>/ directory (produced by the upstream acquisition pipeline
# and by this pipeline). ECM = extracellular / non-cell regions used for background normalization.
AF_TIF = "AF_removal/fused_decon_AF_cleaned_nnls_v3.ome.tif"   # upstream AF-cleaned mosaic (coord frame)
WHOLE_CELL = "segmentation_v2/whole_cell_label.tiff"
MEM_RING = "segmentation_v2/membrane_ring_label.tiff"
ECM = "segmentation_v2/ecm_label.tiff"
CELL_STATS = "segmentation_v2/cell_stats.csv"
IEL_LOG = "segmentation_v2/iel_split_log.csv"


def list_samples():
    """All sample-scene folders under DATA_ROOT, sorted."""
    return sorted(p.name for p in DATA_ROOT.glob("sample_*") if p.is_dir())


def patient_of(sample):
    """'sample_LXT31_scene_1' -> 'LXT31'."""
    return sample.split("_")[1]


def load_channel(sample, ch_name):
    """Load one channel (full res, float32) from the AF_removal OME-TIFF."""
    arr = tiff.imread(str(DATA_ROOT / sample / AF_TIF), key=CH[ch_name])
    return arr.astype(np.float32)


def load_nox1(sample):
    return load_channel(sample, "NOX1")


def load_mask(sample, which):
    return tiff.imread(str(DATA_ROOT / sample / which))


def imread(path, **kw):
    return tiff.imread(str(path), **kw)
