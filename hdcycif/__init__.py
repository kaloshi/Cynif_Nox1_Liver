"""hdcycif - shared helpers for the HD CyCIF NOX1 analysis pipeline.

Modules
-------
config        dataset paths (via the HD_DATA_ROOT env var), channel map, image/mask loaders
tiles         tile-origin detection (DAPI template matching), feathering, seam metric
af_removal    NNLS v3 autofluorescence removal
segmentation  membrane-ring construction and per-cell multi-channel quantification
"""
__version__ = "1.0.0"
