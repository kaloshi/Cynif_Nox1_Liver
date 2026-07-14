"""Pipeline 5 - export the hepatocyte NOX1 result tables to a multi-sheet Excel workbook.

Reads the per-sample / per-patient summaries written by pipeline/4 (median and mean normalized
NOX1 per cell type). The primary readout is the MEDIAN NOX1 in hepatocytes (ASGR1+ / CD45-);
CD45+ leukocytes are reported as a negative-population reference.

Run order: 1 -> 2 -> 3 -> 4 -> 5.  Output: <HD_DATA_ROOT>/_nox1_normalization/nox1_v4_results.xlsx
"""
import pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from hdcycif import config as C


def main():
    ps = pd.read_csv(C.OUT_ROOT / "nox1_v4_per_sample.csv")    # patient, Sample, celltype, n_cells, nox1_median, nox1_mean
    pp = pd.read_csv(C.OUT_ROOT / "nox1_v4_per_patient.csv")   # patient, celltype, n_cells, nox1_median, nox1_mean

    # per-sample wide pivot (one row per sample, hepatocyte + CD45 columns)
    wide = ps.pivot_table(index=["patient", "Sample"], columns="celltype",
                          values=["n_cells", "nox1_median", "nox1_mean"]).round(2)
    wide.columns = [f"{v}_{ct}" for v, ct in wide.columns]
    wide = wide.reset_index()

    # summary per patient: hepatocyte (primary) + CD45 reference side by side
    def side(ct, tag):
        return pp[pp["celltype"] == ct].drop(columns="celltype").rename(columns={
            "n_cells": f"{tag}_n", "nox1_median": f"{tag}_nox1_median", "nox1_mean": f"{tag}_nox1_mean"})
    summary = side("hepatocyte", "hepato").merge(side("leukocyte_CD45", "CD45"), on="patient")

    out = C.OUT_ROOT / "nox1_v4_results.xlsx"
    with pd.ExcelWriter(out) as xl:
        summary.to_excel(xl, sheet_name="summary_per_patient", index=False)
        pp.sort_values(["celltype", "patient"]).to_excel(xl, sheet_name="per_patient_long", index=False)
        ps.sort_values(["celltype", "Sample"]).to_excel(xl, sheet_name="per_sample_long", index=False)
        wide.to_excel(xl, sheet_name="per_sample_wide", index=False)
    print(f"Wrote {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
