"""Membrane-ring construction and per-cell multi-channel quantification (used by pipeline/3).

``quantify`` returns, for every whole-cell label, morphology plus per-channel whole-cell mean and
membrane-ring mean/median for all six channels (DAPI, AF1, AF2, ASGR1, CD45, NOX1).
"""
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import disk, erosion

CH_NAMES = ["DAPI", "AF1", "AF2", "ASGR1", "CD45", "NOX1"]


def make_membrane_ring(wc_label, ring_px):
    """Vectorised membrane ring: erode the whole-cell labels; the removed rim is the ring."""
    se = disk(ring_px)
    eroded = erosion(wc_label, se)
    ring = wc_label.copy()
    ring[eroded == wc_label] = 0
    return ring


def quantify(wc_label, mr_label, stack, sample):
    """Quantify all channels per whole-cell and membrane-ring region -> per-cell DataFrame."""
    def med_prop(mask, intensity):
        return np.median(intensity[mask])

    wc_props = regionprops_table(
        wc_label.astype(np.int32), intensity_image=stack[0],
        properties=["label", "area", "perimeter", "eccentricity", "solidity",
                    "centroid", "bbox", "intensity_mean"],
        extra_properties=(med_prop,),
    )
    df = pd.DataFrame({
        "CellID": wc_props["label"], "Area": wc_props["area"], "Perimeter": wc_props["perimeter"],
        "Eccentricity": wc_props["eccentricity"], "Solidity": wc_props["solidity"],
        "Y_centroid": wc_props["centroid-0"], "X_centroid": wc_props["centroid-1"],
        "bbox-0": wc_props["bbox-0"], "bbox-1": wc_props["bbox-1"],
        "bbox-2": wc_props["bbox-2"], "bbox-3": wc_props["bbox-3"],
        "Sample": sample, "DAPI_mean": wc_props["intensity_mean"],
    })

    for ci, chname in enumerate(CH_NAMES):
        ch_img = stack[ci].astype(np.float32)
        rp = regionprops_table(wc_label.astype(np.int32), intensity_image=ch_img,
                               properties=["label", "intensity_mean"])
        df[f"{chname}_mean"] = rp["intensity_mean"]
        mr_rp = regionprops_table(mr_label.astype(np.int32), intensity_image=ch_img,
                                  properties=["label", "intensity_mean"], extra_properties=(med_prop,))
        mr_df = pd.DataFrame({"CellID": mr_rp["label"],
                              f"{chname}_mem_mean": mr_rp["intensity_mean"],
                              f"{chname}_mem_median": mr_rp["med_prop"]})
        df = df.merge(mr_df, on="CellID", how="left")
    return df
