"""Faithful port of the upstream NNLS V3 AF-removal (Cycif_pipeline_part_3_AF_removal_NNLS_V3),
restricted to the single-cycle HD case and exposed as a reusable function so pipeline/1 can apply
the SAME autofluorescence removal to the freshly de-seamed NOX1/AF1/AF2 channels.

Algorithm (identical parameters to the original notebook):
  donors AF1/AF2 are gaussian-smoothed (sigma=2); a tissue mask is taken from AF2 (>p15);
  per marker the lower ESTIMATION_QUANTILE (60%) of tissue pixels estimate NNLS coefficients
  (AF2 first, then AF1 on the residual), each capped by a 99.9-percentile ratio and MAX_COEFF;
  final subtraction is the GLOBAL linear  clean = max(0, marker - w2*AF2 - w1*AF1);
  if the residual Pearson correlation with AF still exceeds 0.3 a 2nd pass is applied.

Because the upstream pre-AF input (spillover_cleaned) was an intermediate that was cleaned up
and is not bit-identical to decon2D_fused, the OLD cleaned mosaic cannot be reproduced exactly;
re-estimating on the clean inputs is the correct, seam-free equivalent.
"""
import numpy as np
from scipy.optimize import nnls
from scipy import ndimage as ndi
from scipy.stats import pearsonr

# === PARAMETERS V3 (verbatim from part_3_AF_removal_NNLS_V3) ===
MIN_REDUCTION_PCT = 0.5
MIN_COEFF = 0.002
MAX_COEFF = 2.0
PERCENTILE_Q = 99.9
SAFETY_FACTOR = 1.0
TISSUE_PERCENTILE = 15
ESTIMATION_QUANTILE = 0.60
DONOR_SIGMA = 2.0
RESIDUAL_CORR_THRESHOLD = 0.3


def nnls_coeff_1d(donor, target):
    A = donor.reshape(-1, 1)
    x, _ = nnls(A, target.ravel())
    return float(x[0])


def percentile_cap_coeff(donor, target, raw_coeff):
    d_q = float(np.percentile(donor, PERCENTILE_Q))
    t_q = float(np.percentile(target, PERCENTILE_Q))
    if d_q <= 1e-6:
        return min(raw_coeff, MAX_COEFF)
    cap = (t_q / d_q) * SAFETY_FACTOR
    return max(0.0, min(raw_coeff, cap, MAX_COEFF))


def make_tissue_mask(af2):
    mask = af2 > np.percentile(af2, TISSUE_PERCENTILE)
    return ndi.binary_fill_holes(mask)


def get_estimation_mask(marker_tissue, quantile=ESTIMATION_QUANTILE):
    return marker_tissue <= np.percentile(marker_tissue, quantile * 100)


def compute_af_correlation(channel, af1, af2, sample_size=100000):
    ch = channel.ravel().astype(np.float64)
    a1 = af1.ravel().astype(np.float64)
    a2 = af2.ravel().astype(np.float64)
    valid = (ch > 0) & ((a1 + a2) > 0)
    if valid.sum() < 1000:
        return {"pearson_af1": 0.0, "pearson_af2": 0.0, "max_corr": 0.0}
    ch, a1, a2 = ch[valid], a1[valid], a2[valid]
    if ch.size > sample_size:
        idx = np.random.choice(ch.size, sample_size, replace=False)
        ch, a1, a2 = ch[idx], a1[idx], a2[idx]
    r1, _ = pearsonr(ch, a1)
    r2, _ = pearsonr(ch, a2)
    return {"pearson_af1": float(r1), "pearson_af2": float(r2),
            "max_corr": float(max(abs(r1), abs(r2)))}


def _single_pass(marker_2d, af2_2d, af1_2d, af2_tissue, af1_tissue, marker_tissue, est_mask):
    marker_est = marker_tissue[est_mask]
    af2_est = af2_tissue[est_mask]
    w2_raw = nnls_coeff_1d(af2_est, marker_est)
    w2 = percentile_cap_coeff(af2_est, marker_est, w2_raw)
    w1 = w1_raw = 0.0
    if af1_2d is not None and af1_tissue is not None:
        residual_est = np.maximum(0.0, marker_est - w2 * af2_est)
        af1_est = af1_tissue[est_mask]
        w1_raw = nnls_coeff_1d(af1_est, residual_est)
        w1 = percentile_cap_coeff(af1_est, residual_est, w1_raw)
    if af1_2d is not None:
        clean = np.maximum(0.0, marker_2d - w2 * af2_2d - w1 * af1_2d)
    else:
        clean = np.maximum(0.0, marker_2d - w2 * af2_2d)
    return clean, w2, w1, w2_raw, w1_raw


def remove_af(marker_raw, af1_raw, af2_raw):
    """Run NNLS V3 AF-removal on one marker channel (float32 2D, raw = NOT yet smoothed).

    Returns (clean_float32, report_dict). Mirrors the notebook's per-marker logic incl. the
    skip rule and the residual-correlation-triggered 2nd pass.
    """
    marker = marker_raw.astype(np.float32)
    af2 = ndi.gaussian_filter(af2_raw.astype(np.float32), DONOR_SIGMA) if DONOR_SIGMA > 0 else af2_raw.astype(np.float32)
    af1 = ndi.gaussian_filter(af1_raw.astype(np.float32), DONOR_SIGMA) if (af1_raw is not None and DONOR_SIGMA > 0) \
        else (af1_raw.astype(np.float32) if af1_raw is not None else None)

    tissue = make_tissue_mask(af2)
    af2_tissue = af2[tissue]
    af1_tissue = af1[tissue] if af1 is not None else None
    marker_tissue = marker[tissue]

    original_sum = float(marker.sum())
    est_mask = get_estimation_mask(marker_tissue)
    clean1, w2, w1, w2_raw, w1_raw = _single_pass(
        marker, af2, af1, af2_tissue, af1_tissue, marker_tissue, est_mask)
    clean1_sum = float(clean1.sum())
    reduction1 = (original_sum - clean1_sum) / (original_sum + 1e-6) * 100
    apply = (w2 >= MIN_COEFF or w1 >= MIN_COEFF) and reduction1 >= MIN_REDUCTION_PCT

    rep = {"af2_coeff": w2, "af1_coeff": w1, "af2_coeff_raw": w2_raw, "af1_coeff_raw": w1_raw,
           "reduction_pct": reduction1, "applied": bool(apply), "action": None, "pass2": None}

    if not apply:
        rep["action"] = "skipped"
        return marker, rep

    corr1 = compute_af_correlation(clean1, af2, af1 if af1 is not None else af2)
    rep["corr_after_pass1"] = corr1
    if corr1["max_corr"] > RESIDUAL_CORR_THRESHOLD:
        clean1_tissue = clean1[tissue]
        est_mask2 = get_estimation_mask(clean1_tissue)
        clean2, w2b, w1b, _, _ = _single_pass(
            clean1, af2, af1, af2_tissue, af1_tissue, clean1_tissue, est_mask2)
        corr2 = compute_af_correlation(clean2, af2, af1 if af1 is not None else af2)
        rep["pass2"] = {"af2_coeff": w2b, "af1_coeff": w1b,
                        "total_reduction_pct": (original_sum - float(clean2.sum())) / (original_sum + 1e-6) * 100}
        rep["corr_after_pass2"] = corr2
        rep["action"] = "processed_2pass"
        return clean2, rep
    rep["action"] = "processed_1pass"
    return clean1, rep
