# -*- coding: utf-8 -*-
"""
02_build_future_total_co2eq_v2.py

Build total CO2-equivalent flux files from predicted CO2, CH4 and N2O NetCDF files.

Inputs:
- CO2 flux: mg CO2 m-2 h-1
- CH4 flux: mg CH4 m-2 h-1
- N2O flux: ug N2O m-2 h-1

Output:
- co2eq_flux: mg CO2-eq m-2 h-1

Important:
Your N2O model output is treated as gas mass, i.e., ug N2O m-2 h-1.
Therefore, this script does NOT apply a 44/28 conversion.
It only converts ug to mg by dividing by 1000.
"""

from aquaculture_paths import project_path, external_path

import os
import warnings
from typing import List

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")


# ============================================================
# 1. Config
# ============================================================

GWP_CO2 = 1.0
GWP_CH4 = 27.2
GWP_N2O = 273.0

DEFAULT_TARGET_MODELS = [
    "MRI-ESM2-0",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "UKESM1-0-LL",
]
TARGET_MODELS = [
    value.strip()
    for value in os.environ.get("GHG_TARGET_MODELS", ",".join(DEFAULT_TARGET_MODELS)).split(",")
    if value.strip()
]

BASELINE_SCENARIOS = []
MANAGEMENT_SCENARIOS = [
    value.strip()
    for value in os.environ.get("GHG_MANAGEMENT_SCENARIOS", "ssp126,ssp245,ssp585").split(",")
    if value.strip()
]
WORK_DIR = os.environ.get(
    "GHG_WORK_DIR",
    str(project_path('meta/Scenario_management_pathway_discussion/Optimization_water_quality_bounds')),
)
CANDIDATE_CSV = os.environ.get(
    "GHG_CANDIDATE_CSV",
    os.path.join(WORK_DIR, "data", "optimization_candidates_pilot.csv"),
)
# To run all 320 combinations after screening, change this to optimization_candidates_screening.csv or optimization_candidates_full_grid.csv
MAX_CANDIDATES = None


def load_management_candidates(candidate_csv: str):
    if not os.path.exists(candidate_csv):
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")
    import pandas as pd
    df = pd.read_csv(candidate_csv)
    if MAX_CANDIDATES is not None:
        df = df.head(MAX_CANDIDATES).copy()
    return [{"id": str(r["Candidate_ID"]), "label": str(r["Candidate_label"])} for _, r in df.iterrows()]


MANAGEMENT_LEVELS = load_management_candidates(CANDIDATE_CSV)

OUTPUT_ROOT = os.environ.get(
    "GHG_OUTPUT_ROOT",
    str(project_path('meta/Scenario_management_pathway_discussion/Optimization_water_quality_bounds/Future_Run_optimization_candidates')),
)

DIRS_BASELINE = {
    "CO2": f"{OUTPUT_ROOT}/CO2_Baseline",
    "CH4": f"{OUTPUT_ROOT}/CH4_Baseline",
    "N2O": f"{OUTPUT_ROOT}/N2O_Baseline",
}

DIRS_S3 = {
    "CO2": f"{OUTPUT_ROOT}/CO2_Management",
    "CH4": f"{OUTPUT_ROOT}/CH4_Management",
    "N2O": f"{OUTPUT_ROOT}/N2O_Management",
}

OUTPUT_DIR = f"{OUTPUT_ROOT}/Total_GHG"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Helpers
# ============================================================

def year_month_keys(da: xr.DataArray) -> np.ndarray:
    years = da.time.dt.year.values
    months = da.time.dt.month.values
    return np.array([f"{int(y):04d}-{int(m):02d}" for y, m in zip(years, months)])


def subset_da_by_keys(da: xr.DataArray, common_keys: List[str]) -> xr.DataArray:
    keys = year_month_keys(da)
    key_to_idx = {}
    for i, k in enumerate(keys):
        if k not in key_to_idx:
            key_to_idx[k] = i
    indices = [key_to_idx[k] for k in common_keys if k in key_to_idx]
    return da.isel(time=indices)


def align_three_fluxes(co2: xr.DataArray, ch4: xr.DataArray, n2o: xr.DataArray):
    """
    Align three gases by year-month and interpolate CH4/N2O to CO2 grid if needed.
    """
    keys_co2 = set(year_month_keys(co2))
    keys_ch4 = set(year_month_keys(ch4))
    keys_n2o = set(year_month_keys(n2o))
    common_keys = sorted(list(keys_co2 & keys_ch4 & keys_n2o))

    if not common_keys:
        raise ValueError("CO2, CH4 and N2O have no common year-month keys.")

    co2 = subset_da_by_keys(co2, common_keys)
    ch4 = subset_da_by_keys(ch4, common_keys)
    n2o = subset_da_by_keys(n2o, common_keys)

    ch4 = ch4.assign_coords(time=co2.time)
    n2o = n2o.assign_coords(time=co2.time)

    if not ch4.lat.equals(co2.lat) or not ch4.lon.equals(co2.lon):
        ch4 = ch4.interp(lat=co2.lat, lon=co2.lon, method="linear")

    if not n2o.lat.equals(co2.lat) or not n2o.lon.equals(co2.lon):
        n2o = n2o.interp(lat=co2.lat, lon=co2.lon, method="linear")

    return co2, ch4, n2o


def open_flux(path: str, var_name: str) -> xr.DataArray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    if var_name not in ds.data_vars:
        raise KeyError(f"{path} does not contain variable {var_name}. Current variables: {list(ds.data_vars)}")
    return ds[var_name]


def build_one_total(file_co2: str, file_ch4: str, file_n2o: str, output_file: str, mode: str, scenario: str, model: str):
    co2 = open_flux(file_co2, "co2_flux")
    ch4 = open_flux(file_ch4, "ch4_flux")
    n2o = open_flux(file_n2o, "n2o_flux")

    co2, ch4, n2o = align_three_fluxes(co2, ch4, n2o)

    # Units:
    # CO2: mg CO2 m-2 h-1
    # CH4: mg CH4 m-2 h-1
    # N2O: ug N2O m-2 h-1 -> mg N2O m-2 h-1
    co2eq_flux = (co2 * GWP_CO2) + (ch4 * GWP_CH4) + ((n2o / 1000.0) * GWP_N2O)

    co2eq_flux.name = "co2eq_flux"
    co2eq_flux.attrs = {
        "units": "mg CO2-eq m-2 h-1",
        "description": "Total CO2-equivalent greenhouse gas flux from CO2, CH4 and N2O.",
        "formula": "co2eq_flux = co2_flux + ch4_flux*GWP_CH4 + (n2o_flux/1000)*GWP_N2O",
        "N2O_note": "N2O flux is treated as gas mass, ug N2O m-2 h-1; no 44/28 conversion applied.",
        "GWP_CO2": GWP_CO2,
        "GWP_CH4": GWP_CH4,
        "GWP_N2O": GWP_N2O,
        "mode": mode,
        "scenario": scenario,
        "gcm": model,
        "output_root": OUTPUT_ROOT,
    }

    ds_out = co2eq_flux.to_dataset()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds_out.to_netcdf(output_file)

    print(f"      ✅ Saved: {output_file}")


# ============================================================
# 3. Main
# ============================================================

def build_total_co2eq():
    print("==================================================")
    print("🌍 Building total CO2-eq files for bounded optimization candidates")
    print(f"   GWP100: CO2={GWP_CO2}, CH4={GWP_CH4}, N2O={GWP_N2O}")
    print(f"   Output folder: {OUTPUT_DIR}")
    print("==================================================\n")

    success = 0

    # Baseline scenarios.
    for model in TARGET_MODELS:
        for scenario in BASELINE_SCENARIOS:
            f_co2 = os.path.join(DIRS_BASELINE["CO2"], f"CO2_China_{model}_{scenario}.nc")
            f_ch4 = os.path.join(DIRS_BASELINE["CH4"], f"CH4_China_{model}_{scenario}.nc")
            f_n2o = os.path.join(DIRS_BASELINE["N2O"], f"N2O_China_{model}_{scenario}.nc")
            out = os.path.join(OUTPUT_DIR, f"Total_CO2eq_China_{model}_{scenario}.nc")

            if not (os.path.exists(f_co2) and os.path.exists(f_ch4) and os.path.exists(f_n2o)):
                print(f"⚠️ Skip baseline {model} {scenario}: missing one or more gas files.")
                continue

            print(f"🔄 Mixing baseline: {model} | {scenario}")
            try:
                build_one_total(f_co2, f_ch4, f_n2o, out, mode="baseline", scenario=scenario, model=model)
                success += 1
            except Exception as e:
                print(f"      ❌ Failed: {e}")

    # Management pathway matrix.
    for management in MANAGEMENT_LEVELS:
        for model in TARGET_MODELS:
            for scenario in MANAGEMENT_SCENARIOS:
                scenario_id = f"{scenario}_{management['id']}"
                f_co2 = os.path.join(DIRS_S3["CO2"], f"CO2_China_{model}_{scenario_id}_OptimizedBounds.nc")
                f_ch4 = os.path.join(DIRS_S3["CH4"], f"CH4_China_{model}_{scenario_id}_OptimizedBounds.nc")
                f_n2o = os.path.join(DIRS_S3["N2O"], f"N2O_China_{model}_{scenario_id}_OptimizedBounds.nc")
                out = os.path.join(OUTPUT_DIR, f"Total_CO2eq_China_{model}_{scenario_id}_OptimizedBounds.nc")

                if not (os.path.exists(f_co2) and os.path.exists(f_ch4) and os.path.exists(f_n2o)):
                    print(f"Skip management {model} {scenario_id}: missing one or more gas files.")
                    continue

                print(f"Mixing management: {model} | {scenario_id}")
                try:
                    build_one_total(f_co2, f_ch4, f_n2o, out, mode="management", scenario=scenario_id, model=model)
                    success += 1
                except Exception as e:
                    print(f"      Failed: {e}")
    print(f"\n✅ Finished. Successfully generated {success} total CO2-eq files.")
    print(f"📁 Output folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    build_total_co2eq()
