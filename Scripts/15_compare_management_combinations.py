# -*- coding: utf-8 -*-
"""
01_predict_future_flux_all_gases_v2.py

Bounded water-quality optimization candidate prediction pipeline for aquaculture GHG fluxes in China.

This script predicts:
1. Baseline CMIP6 scenarios:
   - historical
   - ssp126
   - ssp245
   - ssp585

2. Management pathway matrix:
   - SSP126, SSP245 and SSP585 climate backgrounds
   - Low, medium and high management levels
   - DO increased gradually from 2025 to 2060
   - NH4N and NO3N reduced gradually from 2025 to 2060
   - pH is varied within a bounded sensitivity range.
     This pH-median version uses a fixed empirical median pH value of 8.04.

Outputs:
- CO2 flux: mg CO2 m-2 h-1
- CH4 flux: mg CH4 m-2 h-1
- N2O flux: ug N2O m-2 h-1

Key scientific assumptions:
- DO is represented by a temperature- and elevation-driven oxygen saturation proxy.
- pH is represented by a fixed empirical median value:
      pH = 8.04
  This avoids using the weak empirical DO-pH proxy in the main prediction.
- CMIP6 precipitation pr is assumed to be kg m-2 s-1 and converted to mm month-1
  using the actual number of days in each month.
- sfcWind is used dynamically.
"""

from aquaculture_paths import project_path, external_path

import os
import warnings
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 1. User configuration
# ============================================================

INPUT_ROOT = str(project_path('meta/CMIP6_Data_Organized'))
ELEVATION_FILE = str(project_path("meta/elev.1-deg.nc"))

# New output root. This avoids overwriting the original Fig. 6 results.
OUTPUT_ROOT = os.environ.get(
    "GHG_OUTPUT_ROOT",
    str(project_path('meta/Scenario_management_pathway_discussion/Optimization_water_quality_bounds/Future_Run_optimization_candidates')),
)

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

BASELINE_SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]
S3_SCENARIOS = [
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
MAX_CANDIDATES = None  # Set to an integer for a small test run.


def load_management_candidates(candidate_csv: str) -> List[dict]:
    if not os.path.exists(candidate_csv):
        raise FileNotFoundError(
            f"Candidate CSV not found: {candidate_csv}. Run 00_build_optimization_candidates.py first."
        )
    df = pd.read_csv(candidate_csv)
    required = ["Candidate_ID", "Candidate_label", "DO_increase", "NH4_reduction", "NO3_reduction", "pH_target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate CSV missing columns: {missing}")
    if MAX_CANDIDATES is not None:
        df = df.head(MAX_CANDIDATES).copy()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id": str(r["Candidate_ID"]),
            "label": str(r["Candidate_label"]),
            "do_increase": float(r["DO_increase"]),
            "nh4_reduction": float(r["NH4_reduction"]),
            "no3_reduction": float(r["NO3_reduction"]),
            "ph_target": float(r["pH_target"]),
        })
    return rows


MANAGEMENT_LEVELS = load_management_candidates(CANDIDATE_CSV)
CURRENT_MANAGEMENT = MANAGEMENT_LEVELS[0]

CHINA_LAT_RANGE = (15, 55)
CHINA_LON_RANGE = (70, 140)

# Run switches
RUN_BASELINE = False
RUN_S3 = True

# First test with 12 months; then change TEST_MODE to False for full run.
TEST_MODE = False
TEST_N_STEPS = 12

# Management intervention timing
S3_START_YEAR = 2025
S3_FULL_YEAR = 2060

# pH treatment in management scenarios:
# pH is fixed at the empirical median from the original observation datasets.
# This keeps historical and future treatment consistent after dropping the weak DO-pH proxy.
S3_PH_MODE = "optimized_target"
PH_MEDIAN_VALUE = 8.04
PH_BASELINE_VALUE = 8.04

# Smoothing after 2025 to reduce abrupt RF/LGBM boundary effects.
USE_S3_MONTE_CARLO_SMOOTHING = False
SMOOTHING_SAMPLES = 10
SMOOTHING_RELATIVE_SD = 0.005
RANDOM_SEED = 42

# Historical water quality background for N2O
BASELINE_NH4N = 0.57
BASELINE_NO3N = 0.93

# Static Chl-a for CO2 model
CHLA_STATIC_VALUE = 50.43


# ============================================================
# 2. Gas model configuration
# ============================================================

@dataclass
class GasConfig:
    gas: str
    model_path: str
    output_dir_baseline: str
    output_dir_s3: str
    var_name: str
    output_unit: str
    smearing_factor: float
    feature_columns: List[str]


GAS_CONFIGS: Dict[str, GasConfig] = {
    "CO2": GasConfig(
        gas="CO2",
        model_path="LGBM_China_CO2_Best_Param_v2.pkl",
        output_dir_baseline=f"{OUTPUT_ROOT}/CO2_Baseline",
        output_dir_s3=f"{OUTPUT_ROOT}/CO2_Management",
        var_name="co2_flux",
        output_unit="mg CO2 m-2 h-1",
        smearing_factor=1.57,
        feature_columns=[
            "DO",
            "pH",
            "chla",
            "Watertempreature",
            "Airtemperature_Filled",
            "Monthly_Precipitation_ERA5",
            "Windspeed_Filled",
            "ERA5_Radiation",
            "Atomsphericpressure_Filled",
            "ERA5_SoilTemp",
        ],
    ),

    "CH4": GasConfig(
        gas="CH4",
        model_path="RF_China_CH4_Best_Param_v2.pkl",
        output_dir_baseline=f"{OUTPUT_ROOT}/CH4_Baseline",
        output_dir_s3=f"{OUTPUT_ROOT}/CH4_Management",
        var_name="ch4_flux",
        output_unit="mg CH4 m-2 h-1",
        smearing_factor=1.304119,
        feature_columns=[
            "Longitude",
            "Watertempreature",
            "DO",
            "MAT_Final",
            "Latitude",
            "Atomsphericpressure_Filled",
            "ERA5_SoilTemp",
            "Airtemperature_Filled",
            "pH",
            "MAP_Final",
        ],
    ),

    "N2O": GasConfig(
        gas="N2O",
        model_path="RF_China_N2O_BayesTuned.pkl",
        output_dir_baseline=f"{OUTPUT_ROOT}/N2O_Baseline",
        output_dir_s3=f"{OUTPUT_ROOT}/N2O_Management",
        var_name="n2o_flux",
        output_unit="ug N2O m-2 h-1",
        smearing_factor=2.012610,
        feature_columns=[
            "Watertempreature",
            "pH",
            "DO",
            "NH4N",
            "NO3N",
            "MAT_Final",
            "MAP_Final",
            "Monthly_Precipitation_ERA5",
            "Windspeed_Filled",
            "Longitude",
        ],
    ),
}


# ============================================================
# 3. Scientific helper functions
# ============================================================

def calculate_pressure_kPa(elevation_m):
    return 101.325 * np.power((1 - 2.25577e-5 * elevation_m), 5.25588)


def calculate_pressure_hPa(elevation_m):
    return calculate_pressure_kPa(elevation_m) * 10.0


def calculate_solar_radiation_FAO56(lat_array, day_of_year):
    doy = np.array(day_of_year)
    phi = np.deg2rad(lat_array)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    tan_phi_delta = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(tan_phi_delta)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    gsc = 0.0820
    ra = (
        (24 * 60 / np.pi)
        * gsc
        * dr
        * (
            ws * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(ws)
        )
    )
    return np.maximum(0, ra * 0.6 * 11.574)


def calculate_do_graham2025(temp_k, elevation_m):
    """
    Temperature- and elevation-driven DO saturation proxy.
    temp_k: water/air temperature proxy in K
    elevation_m: elevation in m
    """
    tw = np.maximum(temp_k, 273.15)
    c0 = -139.34411
    c1 = 1.575701e5
    c2 = -6.642308e7
    c3 = 1.243800e10
    c4 = -8.621949e11
    c_elev = 0.0001148
    exponent = c0 + c1 / tw + c2 / tw**2 + c3 / tw**3 + c4 / tw**4
    do_sat = np.exp(exponent)
    return np.maximum(do_sat * (1 - c_elev * elevation_m), 0)


def estimate_ph_from_do_empirical(do_val):
    """
    Fixed empirical median pH from the original observation datasets.
    The input shape is preserved for compatibility with the existing feature pipeline.
    """
    return np.full_like(do_val, PH_MEDIAN_VALUE, dtype=float)


def inverse_log_modulus(x):
    return np.sign(x) * (np.power(10, np.abs(x)) - 1)


def policy_factor(year: int) -> float:
    if year <= S3_START_YEAR:
        return 0.0
    if year <= S3_FULL_YEAR:
        return (year - S3_START_YEAR) / (S3_FULL_YEAR - S3_START_YEAR)
    return 1.0


# ============================================================
# 4. NetCDF utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise KeyError("Dataset must contain lon/lat or longitude/latitude coordinates.")

    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)

    return ds.sortby(["lat", "lon"])


def crop_china(ds: xr.Dataset) -> xr.Dataset:
    ds = normalize_coords(ds)
    lat_min, lat_max = CHINA_LAT_RANGE
    lon_min, lon_max = CHINA_LON_RANGE

    if ds.lat.values[0] <= ds.lat.values[-1]:
        return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))


def find_files(model_dir: str, variable_keyword: str, scenario: str) -> List[str]:
    files = []
    for f in os.listdir(model_dir):
        lower = f.lower()
        if lower.endswith(".nc") and variable_keyword.lower() in lower and scenario.lower() in lower:
            files.append(os.path.join(model_dir, f))
    return sorted(files)


def open_cmip6_variable(files: List[str], variable_name: str) -> xr.Dataset:
    if not files:
        raise FileNotFoundError(f"No files found for variable {variable_name}")

    ds = xr.open_mfdataset(files, combine="by_coords", use_cftime=True, parallel=False)
    ds = crop_china(ds)

    if variable_name not in ds.data_vars:
        data_vars = list(ds.data_vars)
        matched = [v for v in data_vars if v.lower() == variable_name.lower()]
        if matched:
            ds = ds.rename({matched[0]: variable_name})
        elif len(data_vars) == 1:
            ds = ds.rename({data_vars[0]: variable_name})
        else:
            raise KeyError(f"Variable {variable_name} not found. Current variables: {data_vars}")

    return ds[[variable_name]]


def year_month_keys(ds: xr.Dataset) -> np.ndarray:
    years = ds.time.dt.year.values
    months = ds.time.dt.month.values
    return np.array([f"{int(y):04d}-{int(m):02d}" for y, m in zip(years, months)])


def subset_by_keys(ds: xr.Dataset, common_keys: List[str]) -> xr.Dataset:
    keys = year_month_keys(ds)
    key_to_idx = {}
    for i, k in enumerate(keys):
        if k not in key_to_idx:
            key_to_idx[k] = i
    indices = [key_to_idx[k] for k in common_keys if k in key_to_idx]
    return ds.isel(time=indices)


def align_climate_datasets(ds_tas: xr.Dataset, ds_pr: xr.Dataset, ds_wind: xr.Dataset):
    """
    Align by year-month, not exact timestamp.
    This avoids the 00:00 vs 12:00 CMIP6 timestamp trap.
    """
    keys_tas = set(year_month_keys(ds_tas))
    keys_pr = set(year_month_keys(ds_pr))
    keys_wind = set(year_month_keys(ds_wind))
    common_keys = sorted(list(keys_tas & keys_pr & keys_wind))

    if not common_keys:
        raise ValueError("tas, pr and sfcWind have no common year-month keys.")

    ds_tas = subset_by_keys(ds_tas, common_keys)
    ds_pr = subset_by_keys(ds_pr, common_keys)
    ds_wind = subset_by_keys(ds_wind, common_keys)

    # Force pr and wind time coordinates to tas time after year-month alignment.
    ds_pr = ds_pr.assign_coords(time=ds_tas.time)
    ds_wind = ds_wind.assign_coords(time=ds_tas.time)

    target_lat = ds_tas.lat
    target_lon = ds_tas.lon

    if not ds_pr.lat.equals(target_lat) or not ds_pr.lon.equals(target_lon):
        ds_pr["pr"] = ds_pr["pr"].interp(lat=target_lat, lon=target_lon, method="linear")

    if not ds_wind.lat.equals(target_lat) or not ds_wind.lon.equals(target_lon):
        ds_wind["sfcWind"] = ds_wind["sfcWind"].interp(lat=target_lat, lon=target_lon, method="linear")

    if TEST_MODE:
        ds_tas = ds_tas.isel(time=slice(0, TEST_N_STEPS))
        ds_pr = ds_pr.isel(time=slice(0, TEST_N_STEPS))
        ds_wind = ds_wind.isel(time=slice(0, TEST_N_STEPS))

    return ds_tas, ds_pr, ds_wind


def open_climate_group(model_name: str, scenario: str):
    model_dir = os.path.join(INPUT_ROOT, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files_tas = find_files(model_dir, "tas", scenario)
    files_pr = find_files(model_dir, "pr", scenario)
    files_wind = find_files(model_dir, "sfcWind", scenario)

    if not files_tas:
        raise FileNotFoundError(f"Missing tas files for {model_name} {scenario}")
    if not files_pr:
        raise FileNotFoundError(f"Missing pr files for {model_name} {scenario}")
    if not files_wind:
        raise FileNotFoundError(f"Missing sfcWind files for {model_name} {scenario}; this pipeline uses dynamic wind.")

    ds_tas = open_cmip6_variable(files_tas, "tas")
    ds_pr = open_cmip6_variable(files_pr, "pr")
    ds_wind = open_cmip6_variable(files_wind, "sfcWind")

    return align_climate_datasets(ds_tas, ds_pr, ds_wind)


def load_elevation_for_grid(target_da: xr.DataArray) -> np.ndarray:
    if not os.path.exists(ELEVATION_FILE):
        print("      ⚠️ Elevation file not found. Using default 500 m.")
        return np.full(target_da.shape, 500.0, dtype=np.float32)

    try:
        ds_elev = xr.open_dataset(ELEVATION_FILE, decode_times=False).squeeze()
        ds_elev = crop_china(ds_elev)
        elev_var = list(ds_elev.data_vars)[0]
        elev_grid = ds_elev[elev_var].interp(
            lat=target_da.lat,
            lon=target_da.lon,
            method="linear"
        ).values

        if elev_grid.ndim > 2:
            elev_grid = np.squeeze(elev_grid)

        return elev_grid.astype(np.float32)

    except Exception as e:
        print(f"      ⚠️ Elevation reading failed. Using default 500 m. Error: {e}")
        return np.full(target_da.shape, 500.0, dtype=np.float32)


def get_days_in_month(ds: xr.Dataset) -> np.ndarray:
    try:
        return ds.time.dt.days_in_month.values.astype(int)
    except Exception:
        return np.full(len(ds.time), 30, dtype=int)


def get_dayofyear(ds: xr.Dataset) -> np.ndarray:
    try:
        return ds.time.dt.dayofyear.values.astype(int)
    except Exception:
        return np.full(len(ds.time), 15, dtype=int)


def get_years(ds: xr.Dataset) -> np.ndarray:
    return ds.time.dt.year.values.astype(int)


# ============================================================
# 5. Feature builder and prediction
# ============================================================

def get_model_feature_columns(model, expected_columns: List[str]) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        missing = [c for c in model_cols if c not in expected_columns]
        if not missing:
            return model_cols
    return expected_columns


def build_features(
    gas: str,
    tas_k: np.ndarray,
    pr_flux: np.ndarray,
    wind: np.ndarray,
    elev_grid: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    mask: np.ndarray,
    day_of_year: int,
    days_in_month: int,
    year: int,
    mode: str,
) -> pd.DataFrame:

    air_c = tas_k[mask] - 273.15
    elev_m = elev_grid[mask]
    lon_val = lon_grid[mask]
    lat_val = lat_grid[mask]

    # CMIP6 pr: kg m-2 s-1. 1 kg m-2 = 1 mm.
    monthly_pr_mm = pr_flux[mask] * 86400.0 * float(days_in_month)
    wind_val = wind[mask]

    water_c = 0.4180 * air_c + 12.9454
    base_do = calculate_do_graham2025(tas_k[mask], elev_m)

    s_factor = policy_factor(year) if mode == "s3" else 0.0
    nominal_do = base_do * (1.0 + CURRENT_MANAGEMENT["do_increase"] * s_factor)
    target_do = np.where(base_do < 10.0, np.minimum(nominal_do, 10.0), base_do)

    if mode == "s3":
        # pH is treated as a bounded water-chemistry sensitivity variable.
        # It moves linearly from the baseline median pH to the candidate target by 2060.
        ph_target = CURRENT_MANAGEMENT.get("ph_target", PH_BASELINE_VALUE)
        ph_val = np.full_like(base_do, PH_BASELINE_VALUE + (ph_target - PH_BASELINE_VALUE) * s_factor, dtype=float)
    else:
        ph_val = estimate_ph_from_do_empirical(base_do)

    if gas == "CO2":
        df = pd.DataFrame({
            "DO": target_do,
            "pH": ph_val,
            "chla": np.full_like(water_c, CHLA_STATIC_VALUE, dtype=float),
            "Watertempreature": water_c,
            "Airtemperature_Filled": air_c,
            "Monthly_Precipitation_ERA5": monthly_pr_mm,
            "Windspeed_Filled": wind_val,
            "ERA5_Radiation": calculate_solar_radiation_FAO56(lat_val, day_of_year),
            "Atomsphericpressure_Filled": calculate_pressure_kPa(elev_m),
            "ERA5_SoilTemp": air_c,
        })

    elif gas == "CH4":
        df = pd.DataFrame({
            "Longitude": lon_val,
            "Watertempreature": water_c,
            "DO": target_do,
            "MAT_Final": air_c,
            "Latitude": lat_val,
            "Atomsphericpressure_Filled": calculate_pressure_hPa(elev_m),
            "ERA5_SoilTemp": air_c,
            "Airtemperature_Filled": air_c,
            "pH": ph_val,
            "MAP_Final": monthly_pr_mm * 12.0,
        })

    elif gas == "N2O":
        nh4 = BASELINE_NH4N * (1.0 - CURRENT_MANAGEMENT["nh4_reduction"] * s_factor)
        no3 = BASELINE_NO3N * (1.0 - CURRENT_MANAGEMENT["no3_reduction"] * s_factor)

        df = pd.DataFrame({
            "Watertempreature": water_c,
            "pH": ph_val,
            "DO": target_do,
            "NH4N": nh4,
            "NO3N": no3,
            "MAT_Final": air_c,
            "MAP_Final": monthly_pr_mm * 12.0,
            "Monthly_Precipitation_ERA5": monthly_pr_mm,
            "Windspeed_Filled": wind_val,
            "Longitude": lon_val,
        })

    else:
        raise ValueError(f"Unsupported gas: {gas}")

    return df


def predict_values(model, gas: str, config: GasConfig, x_clean: pd.DataFrame, year: int, mode: str, rng: np.random.Generator):
    if mode != "s3" or year <= S3_START_YEAR or not USE_S3_MONTE_CARLO_SMOOTHING:
        y_log = model.predict(x_clean)
        return inverse_log_modulus(y_log) * config.smearing_factor

    preds = []
    for _ in range(SMOOTHING_SAMPLES):
        x_mc = x_clean.copy()
        x_mc["DO"] *= rng.normal(1.0, SMOOTHING_RELATIVE_SD, size=len(x_mc))

        if gas == "N2O":
            x_mc["NH4N"] *= rng.normal(1.0, SMOOTHING_RELATIVE_SD, size=len(x_mc))
            x_mc["NO3N"] *= rng.normal(1.0, SMOOTHING_RELATIVE_SD, size=len(x_mc))

        # pH is not perturbed because pH is not a direct management intervention.
        y_log = model.predict(x_mc)
        preds.append(inverse_log_modulus(y_log))

    return np.mean(preds, axis=0) * config.smearing_factor


def output_filename(gas: str, model_name: str, scenario: str, mode: str):
    if mode == "baseline":
        return f"{gas}_China_{model_name}_{scenario}.nc"
    return f"{gas}_China_{model_name}_{scenario}_{CURRENT_MANAGEMENT['id']}_OptimizedBounds.nc"


def output_file_has_expected_time(path: str, expected_n_steps: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with xr.open_dataset(path) as ds:
            return ds.sizes.get("time") == expected_n_steps
    except Exception:
        return False


def predict_one_gas_for_group(
    gas: str,
    model_obj,
    ds_tas: xr.Dataset,
    ds_pr: xr.Dataset,
    ds_wind: xr.Dataset,
    elev_grid: np.ndarray,
    model_name: str,
    scenario: str,
    mode: str,
    rng: np.random.Generator,
):
    config = GAS_CONFIGS[gas]
    out_dir = config.output_dir_baseline if mode == "baseline" else config.output_dir_s3
    out_name = output_filename(gas, model_name, scenario, mode)
    out_path = os.path.join(out_dir, out_name)

    if output_file_has_expected_time(out_path, ds_tas.sizes["time"]):
        print(f"      Skip existing complete file: {out_path}")
        return

    feature_columns = get_model_feature_columns(model_obj, config.feature_columns)

    lon_grid, lat_grid = np.meshgrid(ds_tas.lon.values, ds_tas.lat.values)
    times = ds_tas.time.values
    years = get_years(ds_tas)
    days_arr = get_days_in_month(ds_tas)
    doy_arr = get_dayofyear(ds_tas)

    pred_list = []

    for i in tqdm(range(len(times)), desc=f"      {gas}-{model_name}-{scenario}-{mode}", leave=False):
        tas_k = ds_tas["tas"].isel(time=i).values
        pr_flux = ds_pr["pr"].isel(time=i).values
        wind = ds_wind["sfcWind"].isel(time=i).values

        mask = ~np.isnan(tas_k)
        if not mask.any():
            pred_list.append(np.full_like(tas_k, np.nan, dtype=np.float32))
            continue

        x_all = build_features(
            gas=gas,
            tas_k=tas_k,
            pr_flux=pr_flux,
            wind=wind,
            elev_grid=elev_grid,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            mask=mask,
            day_of_year=int(doy_arr[i]),
            days_in_month=int(days_arr[i]),
            year=int(years[i]),
            mode=mode,
        )

        missing = [c for c in feature_columns if c not in x_all.columns]
        if missing:
            raise ValueError(f"{gas} missing feature columns: {missing}")

        x_input = x_all[feature_columns].replace([np.inf, -np.inf], np.nan)
        valid_idx = x_input.dropna().index

        result_1d = np.full(len(x_input), np.nan, dtype=np.float32)

        if len(valid_idx) > 0:
            x_clean = x_input.loc[valid_idx, feature_columns].copy()
            y_real = predict_values(
                model=model_obj,
                gas=gas,
                config=config,
                x_clean=x_clean,
                year=int(years[i]),
                mode=mode,
                rng=rng,
            )
            result_1d[valid_idx] = y_real.astype(np.float32)

        frame = np.full_like(tas_k, np.nan, dtype=np.float32)
        frame[mask] = result_1d
        pred_list.append(frame)

    da = xr.DataArray(
        np.array(pred_list, dtype=np.float32),
        coords={"time": times, "lat": ds_tas.lat.values, "lon": ds_tas.lon.values},
        dims=["time", "lat", "lon"],
        name=config.var_name,
        attrs={
            "units": config.output_unit,
            "gas": gas,
            "mode": mode,
            "scenario": scenario,
            "gcm": model_name,
            "model_path": config.model_path,
            "smearing_factor": config.smearing_factor,
            "DO_proxy": "Temperature- and elevation-driven oxygen saturation proxy",
            "pH_proxy": "Bounded fixed-pH sensitivity target from baseline median pH = 8.04",
            "S3_policy": (
                f"{scenario} management pathway: "
                f"DO +{CURRENT_MANAGEMENT['do_increase'] * 100:.0f}%, "
                f"NH4N -{CURRENT_MANAGEMENT['nh4_reduction'] * 100:.0f}%, "
                f"NO3N -{CURRENT_MANAGEMENT['no3_reduction'] * 100:.0f}% for N2O, "
                f"pH target {CURRENT_MANAGEMENT['ph_target']:.2f}."
            ),
            "management_level": CURRENT_MANAGEMENT["label"],
            "management_id": CURRENT_MANAGEMENT["id"],
            "management_DO_increase": CURRENT_MANAGEMENT["do_increase"],
            "management_NH4_reduction": CURRENT_MANAGEMENT["nh4_reduction"],
            "management_NO3_reduction": CURRENT_MANAGEMENT["no3_reduction"],
            "management_pH_target": CURRENT_MANAGEMENT["ph_target"],
            "S3_PH_MODE": S3_PH_MODE,
            "PH_MEDIAN_VALUE": PH_MEDIAN_VALUE,
            "NH4N_baseline": BASELINE_NH4N,
            "NO3N_baseline": BASELINE_NO3N,
            "uses_dynamic_sfcWind": "true",
            "output_root": OUTPUT_ROOT,
        },
    )

    ensure_dir(out_dir)
    da.to_netcdf(out_path)

    print(f"      ✅ Saved: {out_path}")


# ============================================================
# 6. Main execution
# ============================================================

def run_pipeline():
    print("==================================================")
    print("🚀 Unified aquaculture GHG future flux prediction v2")
    print(f"   Output root: {OUTPUT_ROOT}")
    print("   Baseline: historical / ssp126 / ssp245 / ssp585")
    print("   Bounded water-quality optimization candidates under SSP126/SSP245/SSP585")
    print(f"   S3_PH_MODE = {S3_PH_MODE}")
    print("==================================================\n")

    rng = np.random.default_rng(RANDOM_SEED)

    ensure_dir(OUTPUT_ROOT)

    # Load models once.
    models = {}
    for gas, config in GAS_CONFIGS.items():
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"{gas} model not found: {config.model_path}")
        print(f">>> Loading {gas} model: {config.model_path}")
        models[gas] = joblib.load(config.model_path)

    # Baseline scenarios.
    if RUN_BASELINE:
        print("\n================ Baseline CMIP6 scenarios ================\n")
        for model_name in TARGET_MODELS:
            for scenario in BASELINE_SCENARIOS:
                print(f"\n🌊 Reading climate: {model_name} | {scenario}")
                try:
                    ds_tas, ds_pr, ds_wind = open_climate_group(model_name, scenario)
                    elev_grid = load_elevation_for_grid(ds_tas["tas"].isel(time=0))

                    for gas in ["CO2", "CH4", "N2O"]:
                        predict_one_gas_for_group(
                            gas=gas,
                            model_obj=models[gas],
                            ds_tas=ds_tas,
                            ds_pr=ds_pr,
                            ds_wind=ds_wind,
                            elev_grid=elev_grid,
                            model_name=model_name,
                            scenario=scenario,
                            mode="baseline",
                            rng=rng,
                        )

                    ds_tas.close()
                    ds_pr.close()
                    ds_wind.close()

                except Exception as e:
                    print(f"      ❌ Failed: {model_name} | {scenario}: {e}")
                    raise

    # Management pathway matrix.
    if RUN_S3:
        print("\n================ Bounded optimization candidate pathways ================\n")
        global CURRENT_MANAGEMENT
        for management in MANAGEMENT_LEVELS:
            CURRENT_MANAGEMENT = management
            print(
                f"\n================ {management['id']} "
                f"(DO +{management['do_increase'] * 100:.0f}%, "
                f"NH4 -{management['nh4_reduction'] * 100:.0f}%, "
                f"NO3 -{management['no3_reduction'] * 100:.0f}%, "
                f"pH {management['ph_target']:.2f}) ================\n"
            )
            for model_name in TARGET_MODELS:
                for scenario in S3_SCENARIOS:
                    print(f"\nReading climate for management: {model_name} | {scenario} | {management['id']}")
                    try:
                        ds_tas, ds_pr, ds_wind = open_climate_group(model_name, scenario)
                        elev_grid = load_elevation_for_grid(ds_tas["tas"].isel(time=0))

                        for gas in ["CO2", "CH4", "N2O"]:
                            predict_one_gas_for_group(
                                gas=gas,
                                model_obj=models[gas],
                                ds_tas=ds_tas,
                                ds_pr=ds_pr,
                                ds_wind=ds_wind,
                                elev_grid=elev_grid,
                                model_name=model_name,
                                scenario=scenario,
                                mode="s3",
                                rng=rng,
                            )

                        ds_tas.close()
                        ds_pr.close()
                        ds_wind.close()

                    except Exception as e:
                        print(f"      Failed management: {model_name} | {scenario} | {management['id']}: {e}")
                        raise

    print("\nAll prediction tasks finished!")


if __name__ == "__main__":
    run_pipeline()
