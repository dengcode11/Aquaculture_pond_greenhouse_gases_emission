# -*- coding: utf-8 -*-
"""
Analyze the main future projection with fixed empirical median pH.

This script reads existing NetCDF outputs only. It does not modify any NetCDF file.

Main tasks:
- Convert monthly gridded fluxes to China annual national emissions using
  aquaculture pond area weights.
- Summarize future scenarios and SSP585-Mitigation effects.
- Save CSV tables and Fig 6-ready figures.
"""

from __future__ import annotations

from aquaculture_paths import project_path, external_path

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from scipy.spatial import cKDTree

matplotlib.use("Agg")


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(str(project_path('meta')))
OUTPUT_ROOT = BASE_DIR / "Future_Run_v2_pH_median"
FIG_DIR = BASE_DIR / "Fig 6" / "6.4"

POND_SHP = Path(
    str(external_path('download/全球水产养殖池塘遥感监测分布数据集（2020年）-数据实体/全球水产养殖池塘遥感监测分布数据集（2020年）-数据实体/Asia/aqu_AS_ds.shp'))
)

GCM_MODELS = [
    "MRI-ESM2-0",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "UKESM1-0-LL",
]

BASELINE_SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]
FUTURE_SCENARIOS = ["ssp126", "ssp245", "ssp585", "ssp585_S3"]
PERIODS = [
    ("2025-2060", 2025, 2060),
    ("2061-2100", 2061, 2100),
    ("2025-2100", 2025, 2100),
]

GWP = {
    "CO2": 1.0,
    "CH4": 27.2,
    "N2O": 273.0,
}

GAS_CONFIG = {
    "CO2": {
        "var": "co2_flux",
        "baseline_dir": "CO2_Baseline",
        "s3_dir": "CO2_S3_DO20_NoPH",
        "s3_name": "CO2_China_{model}_ssp585_S3_Smoothed_No_pH.nc",
        "unit_scale_to_mg": 1.0,
    },
    "CH4": {
        "var": "ch4_flux",
        "baseline_dir": "CH4_Baseline",
        "s3_dir": "CH4_S3_DO20_NoPH",
        "s3_name": "CH4_China_{model}_ssp585_S3_Smoothed_No_pH.nc",
        "unit_scale_to_mg": 1.0,
    },
    "N2O": {
        "var": "n2o_flux",
        "baseline_dir": "N2O_Baseline",
        "s3_dir": "N2O_S3_DO20_Nred10_NoPH",
        "s3_name": "N2O_China_{model}_ssp585_S3_Smoothed_No_pH.nc",
        "unit_scale_to_mg": 1.0 / 1000.0,
    },
}

TOTAL_VAR = "co2eq_flux"
ANALYSIS_START_YEAR = 2025
ANALYSIS_END_YEAR = 2100
K_NEIGHBORS = 5

ALBERS_CHINA = CRS.from_proj4(
    "+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +datum=WGS84 +units=m +no_defs"
)

CSV_ANNUAL_TOTAL = FIG_DIR / "future_main_pH_median_annual_total_emissions.csv"
CSV_PERIOD_TOTAL = FIG_DIR / "future_main_pH_median_period_total_summary.csv"
CSV_S3_ANNUAL = FIG_DIR / "future_main_pH_median_s3_annual_mitigation.csv"
CSV_S3_PERIOD = FIG_DIR / "future_main_pH_median_s3_period_mitigation.csv"
CSV_GAS_ANNUAL = FIG_DIR / "future_main_pH_median_gas_annual_co2eq.csv"
CSV_GAS_PERIOD = FIG_DIR / "future_main_pH_median_gas_period_co2eq.csv"

FIG_TIMESERIES = FIG_DIR / "future_main_pH_median_total_timeseries"
FIG_CUMULATIVE = FIG_DIR / "future_main_pH_median_period_cumulative_emissions"
FIG_MITIGATION = FIG_DIR / "future_main_pH_median_s3_avoided_emissions"
FIG_GAS_STACKED = FIG_DIR / "future_main_pH_median_gas_contribution_2025_2100"

SCENARIO_LABELS = {
    "historical": "Historical",
    "ssp126": "SSP126",
    "ssp245": "SSP245",
    "ssp585": "SSP585",
    "ssp585_S3": "SSP585-Mitigation",
}

SCENARIO_COLORS = {
    "ssp126": "#2ca25f",
    "ssp245": "#3182bd",
    "ssp585": "#de2d26",
    "ssp585_S3": "#756bb1",
}

GAS_COLORS = {
    "CO2": "#4daf4a",
    "CH4": "#377eb8",
    "N2O": "#984ea3",
}


# ============================================================
# 2. File helpers
# ============================================================

def total_file(model: str, scenario: str) -> Path:
    if scenario == "ssp585_S3":
        return OUTPUT_ROOT / "Total_GHG" / f"Total_CO2eq_China_{model}_ssp585_S3_Smoothed_No_pH.nc"
    return OUTPUT_ROOT / "Total_GHG" / f"Total_CO2eq_China_{model}_{scenario}.nc"


def gas_file(gas: str, model: str, scenario: str) -> Path:
    config = GAS_CONFIG[gas]
    if scenario == "ssp585_S3":
        return OUTPUT_ROOT / config["s3_dir"] / config["s3_name"].format(model=model)
    return OUTPUT_ROOT / config["baseline_dir"] / f"{gas}_China_{model}_{scenario}.nc"


def save_figure(path_stem: Path) -> None:
    plt.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path_stem.with_suffix('.pdf')}")
    print(f"Saved figure: {path_stem.with_suffix('.png')}")


def startup_checks() -> None:
    print("==================================================")
    print("Future main pH-median analysis startup checks")
    print(f"Input root: {OUTPUT_ROOT}")
    print(f"Fig 6 output directory: {FIG_DIR}")
    print(f"Pond shapefile exists: {POND_SHP.exists()} | {POND_SHP}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/available: {FIG_DIR}")

    for scenario in BASELINE_SCENARIOS + ["ssp585_S3"]:
        files = [total_file(model, scenario) for model in GCM_MODELS]
        found = sum(path.exists() for path in files)
        print(f"Total_GHG {scenario}: {found} / {len(GCM_MODELS)} files found")
        for model, path in zip(GCM_MODELS, files):
            if not path.exists():
                print(f"WARNING: missing {scenario} total file for {model}: {path}")
    print("==================================================\n")


# ============================================================
# 3. Pond area weights
# ============================================================

def load_china_pond_points() -> pd.DataFrame:
    if not POND_SHP.exists():
        raise FileNotFoundError(POND_SHP)

    print(f"Reading pond shapefile: {POND_SHP}")
    ponds = gpd.read_file(POND_SHP)

    if "NAME_0" not in ponds.columns:
        raise KeyError(f"{POND_SHP} does not contain NAME_0. Columns: {list(ponds.columns)}")

    if ponds.crs is None:
        print("WARNING: Pond shapefile has no CRS. Assuming EPSG:4326.")
        ponds = ponds.set_crs("EPSG:4326")

    china = ponds.loc[ponds["NAME_0"] == "China"].copy()
    if china.empty:
        raise ValueError('No pond polygons found with NAME_0 == "China".')

    china = china.explode(index_parts=False).reset_index(drop=True)
    china_albers = china.to_crs(ALBERS_CHINA)
    china_albers["area_m2"] = china_albers.geometry.area

    centroids_albers = gpd.GeoSeries(china_albers.geometry.centroid, crs=ALBERS_CHINA)
    centroids_wgs84 = centroids_albers.to_crs("EPSG:4326")

    pond_points = pd.DataFrame(
        {
            "lon": centroids_wgs84.x.to_numpy(dtype=np.float64),
            "lat": centroids_wgs84.y.to_numpy(dtype=np.float64),
            "area_m2": china_albers["area_m2"].to_numpy(dtype=np.float64),
        }
    )
    pond_points = pond_points.replace([np.inf, -np.inf], np.nan).dropna()
    pond_points = pond_points.loc[pond_points["area_m2"] > 0].reset_index(drop=True)

    print(f"China pond polygons used: {len(pond_points):,}")
    print(f"Total pond area: {pond_points['area_m2'].sum() / 1e6:,.2f} km2\n")
    return pond_points


def nearest_indices(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord)
    values = np.asarray(values)

    if coord[0] > coord[-1]:
        reversed_idx = nearest_indices(coord[::-1], values)
        return len(coord) - 1 - reversed_idx

    positions = np.searchsorted(coord, values, side="left")
    positions = np.clip(positions, 0, len(coord) - 1)
    previous = np.clip(positions - 1, 0, len(coord) - 1)
    use_previous = np.abs(values - coord[previous]) <= np.abs(values - coord[positions])
    return np.where(use_previous, previous, positions)


def grid_area_weights(ds: xr.Dataset, pond_points: pd.DataFrame) -> xr.DataArray:
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_idx = nearest_indices(lats, pond_points["lat"].values)
    lon_idx = nearest_indices(lons, pond_points["lon"].values)

    weights = np.zeros((len(lats), len(lons)), dtype=np.float64)
    np.add.at(weights, (lat_idx, lon_idx), pond_points["area_m2"].values)

    return xr.DataArray(
        weights,
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
        dims=("lat", "lon"),
        name="pond_area_m2",
    )


def knn_area_weights(ds: xr.Dataset, pond_points: pd.DataFrame, var_name: str) -> xr.DataArray:
    reference = ds[var_name].mean(dim="time", skipna=True).transpose("lat", "lon")
    lats = ds["lat"].values
    lons = ds["lon"].values
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    valid_mask = np.isfinite(reference.values)
    valid_points = np.column_stack((grid_lon[valid_mask], grid_lat[valid_mask]))

    if len(valid_points) == 0:
        raise ValueError(f"No valid grid cells found for {var_name} in {ds.encoding.get('source', 'dataset')}.")

    k = min(K_NEIGHBORS, len(valid_points))
    tree = cKDTree(valid_points)
    pond_xy = np.column_stack((pond_points["lon"].values, pond_points["lat"].values))
    distances, indices = tree.query(pond_xy, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    weights = 1.0 / (distances + 1e-12)
    weights = weights / weights.sum(axis=1, keepdims=True)

    weighted_area_valid = np.zeros(len(valid_points), dtype=np.float64)
    area_contrib = pond_points["area_m2"].values[:, None] * weights
    np.add.at(weighted_area_valid, indices.ravel(), area_contrib.ravel())

    weighted_area_grid = np.zeros(valid_mask.shape, dtype=np.float64)
    weighted_area_grid[valid_mask] = weighted_area_valid

    return xr.DataArray(
        weighted_area_grid,
        coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
        dims=("lat", "lon"),
        name=f"pond_area_m2_knn{K_NEIGHBORS}",
    )


# ============================================================
# 4. Annual emission calculations
# ============================================================

def annual_emissions_from_flux(
    nc_file: Path,
    var_name: str,
    pond_points: pd.DataFrame,
    flux_scale: float = 1.0,
) -> pd.Series:
    with xr.open_dataset(nc_file) as ds:
        if var_name not in ds.data_vars:
            raise KeyError(f"{nc_file} does not contain variable {var_name}. Variables: {list(ds.data_vars)}")

        area_weights = knn_area_weights(ds, pond_points, var_name)
        flux = ds[var_name] * flux_scale
        weighted_flux_area = (flux * area_weights).sum(dim=("lat", "lon"), skipna=True)
        days = ds["time"].dt.days_in_month
        monthly_gg = (weighted_flux_area * 24.0 * days) / 1e12
        annual = monthly_gg.groupby("time.year").sum(skipna=True)

        return pd.Series(
            annual.values.astype(float),
            index=annual["year"].values.astype(int),
            name="Annual_Gg_CO2eq_yr",
        )


def build_total_annual_table(pond_points: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    for model in GCM_MODELS:
        for scenario in BASELINE_SCENARIOS + ["ssp585_S3"]:
            path = total_file(model, scenario)
            if not path.exists():
                print(f"WARNING: skip missing total file: {path}")
                continue

            print(f"Calculating total annual emissions: {model} | {scenario}")
            try:
                annual = annual_emissions_from_flux(path, TOTAL_VAR, pond_points)
            except Exception as exc:
                print(f"WARNING: failed total annual calculation for {model} | {scenario}: {exc}")
                continue

            for year, value in annual.items():
                records.append(
                    {
                        "Model": model,
                        "Scenario": scenario,
                        "Scenario_label": SCENARIO_LABELS[scenario],
                        "Year": int(year),
                        "Annual_Gg_CO2eq_yr": float(value),
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No total annual emissions were calculated.")

    df = df.sort_values(["Scenario", "Model", "Year"]).reset_index(drop=True)
    df.to_csv(CSV_ANNUAL_TOTAL, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {CSV_ANNUAL_TOTAL}")
    return df


def build_gas_annual_table(pond_points: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    for gas, config in GAS_CONFIG.items():
        for model in GCM_MODELS:
            for scenario in FUTURE_SCENARIOS:
                path = gas_file(gas, model, scenario)
                if not path.exists():
                    print(f"WARNING: skip missing gas file: {path}")
                    continue

                print(f"Calculating gas annual CO2-eq emissions: {gas} | {model} | {scenario}")
                try:
                    flux_scale = config["unit_scale_to_mg"] * GWP[gas]
                    annual = annual_emissions_from_flux(path, config["var"], pond_points, flux_scale=flux_scale)
                except Exception as exc:
                    print(f"WARNING: failed gas annual calculation for {gas} | {model} | {scenario}: {exc}")
                    continue

                annual = annual.loc[
                    (annual.index >= ANALYSIS_START_YEAR) & (annual.index <= ANALYSIS_END_YEAR)
                ]
                for year, value in annual.items():
                    records.append(
                        {
                            "Gas": gas,
                            "Model": model,
                            "Scenario": scenario,
                            "Scenario_label": SCENARIO_LABELS[scenario],
                            "Year": int(year),
                            "Annual_Gg_CO2eq_yr": float(value),
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No gas annual emissions were calculated.")

    df = df.sort_values(["Gas", "Scenario", "Model", "Year"]).reset_index(drop=True)
    df.to_csv(CSV_GAS_ANNUAL, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {CSV_GAS_ANNUAL}")
    return df


# ============================================================
# 5. Summaries
# ============================================================

def summarize_periods(
    annual_df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for keys, group in annual_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for period_name, start_year, end_year in PERIODS:
            sub = group.loc[(group["Year"] >= start_year) & (group["Year"] <= end_year)]
            if sub.empty:
                continue
            record = {
                **base,
                "Period": period_name,
                "Start_year": start_year,
                "End_year": end_year,
                "N_years": int(sub["Year"].nunique()),
                "Cumulative_Gg_CO2eq": float(sub[value_col].sum()),
                "Mean_annual_Gg_CO2eq_yr": float(sub[value_col].mean()),
            }
            records.append(record)
    return pd.DataFrame(records)


def add_mme_rows(period_df: pd.DataFrame, scenario_col: str = "Scenario") -> pd.DataFrame:
    metric_cols = ["Cumulative_Gg_CO2eq", "Mean_annual_Gg_CO2eq_yr"]
    rows = [period_df]
    for stat_name, func in [("MME_Mean", "mean"), ("MME_Std", "std")]:
        grouped = (
            period_df.groupby([scenario_col, "Period", "Start_year", "End_year", "N_years"], as_index=False)[metric_cols]
            .agg(func)
        )
        grouped["Model"] = stat_name
        if "Scenario_label" in period_df.columns:
            label_map = period_df.drop_duplicates(scenario_col).set_index(scenario_col)["Scenario_label"].to_dict()
            grouped["Scenario_label"] = grouped[scenario_col].map(label_map)
        rows.append(grouped[period_df.columns])
    return pd.concat(rows, ignore_index=True)


def build_period_tables(total_annual_df: pd.DataFrame, gas_annual_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    future_total = total_annual_df.loc[
        (total_annual_df["Scenario"].isin(FUTURE_SCENARIOS))
        & (total_annual_df["Year"] >= ANALYSIS_START_YEAR)
        & (total_annual_df["Year"] <= ANALYSIS_END_YEAR)
    ].copy()

    total_period = summarize_periods(
        future_total,
        value_col="Annual_Gg_CO2eq_yr",
        group_cols=["Model", "Scenario", "Scenario_label"],
    )
    total_period = add_mme_rows(total_period)
    total_period = total_period.sort_values(["Scenario", "Period", "Model"]).reset_index(drop=True)
    total_period.to_csv(CSV_PERIOD_TOTAL, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {CSV_PERIOD_TOTAL}")

    gas_period = summarize_periods(
        gas_annual_df,
        value_col="Annual_Gg_CO2eq_yr",
        group_cols=["Gas", "Model", "Scenario", "Scenario_label"],
    )
    mme_gas = (
        gas_period.groupby(["Gas", "Scenario", "Scenario_label", "Period", "Start_year", "End_year", "N_years"], as_index=False)
        [["Cumulative_Gg_CO2eq", "Mean_annual_Gg_CO2eq_yr"]]
        .mean()
    )
    mme_gas["Model"] = "MME_Mean"
    gas_period = pd.concat([gas_period, mme_gas[gas_period.columns]], ignore_index=True)
    gas_period = gas_period.sort_values(["Gas", "Scenario", "Period", "Model"]).reset_index(drop=True)
    gas_period.to_csv(CSV_GAS_PERIOD, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {CSV_GAS_PERIOD}")

    return total_period, gas_period


def build_s3_mitigation_tables(total_annual_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = total_annual_df.loc[total_annual_df["Scenario"] == "ssp585"].copy()
    s3 = total_annual_df.loc[total_annual_df["Scenario"] == "ssp585_S3"].copy()

    merged = baseline.merge(
        s3,
        on=["Model", "Year"],
        suffixes=("_ssp585", "_s3"),
    )
    merged = merged.loc[
        (merged["Year"] >= ANALYSIS_START_YEAR) & (merged["Year"] <= ANALYSIS_END_YEAR)
    ].copy()
    merged["SSP585_Gg_CO2eq"] = merged["Annual_Gg_CO2eq_yr_ssp585"]
    merged["S3_Gg_CO2eq"] = merged["Annual_Gg_CO2eq_yr_s3"]
    merged["Difference_S3_minus_SSP585_Gg_CO2eq"] = merged["S3_Gg_CO2eq"] - merged["SSP585_Gg_CO2eq"]
    merged["Avoided_Gg_CO2eq"] = merged["SSP585_Gg_CO2eq"] - merged["S3_Gg_CO2eq"]
    merged["Avoided_percent"] = np.where(
        merged["SSP585_Gg_CO2eq"].abs() > 0,
        merged["Avoided_Gg_CO2eq"] / merged["SSP585_Gg_CO2eq"].abs() * 100.0,
        np.nan,
    )
    annual = merged[
        [
            "Model",
            "Year",
            "SSP585_Gg_CO2eq",
            "S3_Gg_CO2eq",
            "Difference_S3_minus_SSP585_Gg_CO2eq",
            "Avoided_Gg_CO2eq",
            "Avoided_percent",
        ]
    ].sort_values(["Model", "Year"])
    annual.to_csv(CSV_S3_ANNUAL, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {CSV_S3_ANNUAL}")

    records: List[Dict[str, float]] = []
    for model, group in annual.groupby("Model"):
        for period_name, start_year, end_year in PERIODS:
            sub = group.loc[(group["Year"] >= start_year) & (group["Year"] <= end_year)]
            if sub.empty:
                continue
            ssp585_sum = float(sub["SSP585_Gg_CO2eq"].sum())
            s3_sum = float(sub["S3_Gg_CO2eq"].sum())
            avoided = ssp585_sum - s3_sum
            records.append(
                {
                    "Model": model,
                    "Period": period_name,
                    "Start_year": start_year,
                    "End_year": end_year,
                    "N_years": int(sub["Year"].nunique()),
                    "SSP585_cumulative_Gg_CO2eq": ssp585_sum,
                    "S3_cumulative_Gg_CO2eq": s3_sum,
                    "Difference_S3_minus_SSP585_Gg_CO2eq": s3_sum - ssp585_sum,
                    "Avoided_cumulative_Gg_CO2eq": avoided,
                    "Avoided_percent": avoided / abs(ssp585_sum) * 100.0 if ssp585_sum else np.nan,
                }
            )

    period = pd.DataFrame(records)
    metric_cols = [
        "SSP585_cumulative_Gg_CO2eq",
        "S3_cumulative_Gg_CO2eq",
        "Difference_S3_minus_SSP585_Gg_CO2eq",
        "Avoided_cumulative_Gg_CO2eq",
        "Avoided_percent",
    ]
    rows = [period]
    for stat_name, func in [("MME_Mean", "mean"), ("MME_Std", "std")]:
        grouped = period.groupby(["Period", "Start_year", "End_year", "N_years"], as_index=False)[metric_cols].agg(func)
        grouped["Model"] = stat_name
        rows.append(grouped[period.columns])
    period = pd.concat(rows, ignore_index=True)
    period = period.sort_values(["Period", "Model"]).reset_index(drop=True)
    period.to_csv(CSV_S3_PERIOD, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {CSV_S3_PERIOD}")
    return annual, period


# ============================================================
# 6. Figures
# ============================================================

def plot_total_timeseries(total_annual_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    future = total_annual_df.loc[
        (total_annual_df["Scenario"].isin(FUTURE_SCENARIOS))
        & (total_annual_df["Year"] >= ANALYSIS_START_YEAR)
        & (total_annual_df["Year"] <= ANALYSIS_END_YEAR)
    ]

    for scenario in FUTURE_SCENARIOS:
        sub = future.loc[future["Scenario"] == scenario]
        if sub.empty:
            continue
        stats = sub.groupby("Year")["Annual_Gg_CO2eq_yr"].agg(["mean", "min", "max"]).reset_index()
        color = SCENARIO_COLORS[scenario]
        ax.plot(stats["Year"], stats["mean"], color=color, lw=2.2, label=SCENARIO_LABELS[scenario])
        ax.fill_between(stats["Year"], stats["min"], stats["max"], color=color, alpha=0.16, lw=0)

    ax.set_xlim(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual total GHG emissions (Gg CO$_2$-eq yr$^{-1}$)")
    ax.set_title("Future aquaculture GHG emissions with fixed median pH")
    ax.grid(True, color="#d9d9d9", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(FIG_TIMESERIES)
    plt.close(fig)


def plot_period_cumulative(total_period_df: pd.DataFrame) -> None:
    mme = total_period_df.loc[total_period_df["Model"].isin(["MME_Mean", "MME_Std"])].copy()
    mean_df = mme.loc[mme["Model"] == "MME_Mean"]
    std_df = mme.loc[mme["Model"] == "MME_Std"]

    periods = [period[0] for period in PERIODS]
    scenarios = FUTURE_SCENARIOS
    x = np.arange(len(periods))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, scenario in enumerate(scenarios):
        means = []
        stds = []
        for period in periods:
            mean_row = mean_df.loc[(mean_df["Scenario"] == scenario) & (mean_df["Period"] == period)]
            std_row = std_df.loc[(std_df["Scenario"] == scenario) & (std_df["Period"] == period)]
            means.append(float(mean_row["Cumulative_Gg_CO2eq"].iloc[0]) if not mean_row.empty else np.nan)
            stds.append(float(std_row["Cumulative_Gg_CO2eq"].iloc[0]) if not std_row.empty else np.nan)
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylabel("Cumulative emissions (Gg CO$_2$-eq)")
    ax.set_title("Period cumulative total GHG emissions")
    ax.grid(True, axis="y", color="#d9d9d9", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(FIG_CUMULATIVE)
    plt.close(fig)


def plot_s3_mitigation(s3_annual_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    stats = s3_annual_df.groupby("Year")["Avoided_Gg_CO2eq"].agg(["mean", "min", "max"]).reset_index()

    ax.plot(stats["Year"], stats["mean"], color="#756bb1", lw=2.4, label="MME mean")
    ax.fill_between(stats["Year"], stats["min"], stats["max"], color="#756bb1", alpha=0.18, lw=0, label="GCM min-max")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Avoided emissions (Gg CO$_2$-eq yr$^{-1}$)")
    ax.set_title("SSP585-Mitigation avoided total GHG emissions")
    ax.grid(True, color="#d9d9d9", lw=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(FIG_MITIGATION)
    plt.close(fig)


def plot_gas_stacked(gas_period_df: pd.DataFrame) -> None:
    sub = gas_period_df.loc[
        (gas_period_df["Model"] == "MME_Mean")
        & (gas_period_df["Period"] == "2025-2100")
        & (gas_period_df["Scenario"].isin(["ssp126", "ssp245", "ssp585", "ssp585_S3"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    scenarios = FUTURE_SCENARIOS
    x = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios), dtype=float)

    for gas in ["CO2", "CH4", "N2O"]:
        values = []
        for scenario in scenarios:
            row = sub.loc[(sub["Gas"] == gas) & (sub["Scenario"] == scenario)]
            values.append(float(row["Cumulative_Gg_CO2eq"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x, values, bottom=bottom, color=GAS_COLORS[gas], label=gas, alpha=0.9)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[scenario] for scenario in scenarios])
    ax.set_ylabel("2025-2100 cumulative emissions (Gg CO$_2$-eq)")
    ax.set_title("Gas contributions to cumulative GHG emissions")
    ax.grid(True, axis="y", color="#d9d9d9", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save_figure(FIG_GAS_STACKED)
    plt.close(fig)


# ============================================================
# 7. Main
# ============================================================

def main() -> None:
    startup_checks()
    pond_points = load_china_pond_points()

    total_annual_df = build_total_annual_table(pond_points)
    gas_annual_df = build_gas_annual_table(pond_points)
    total_period_df, gas_period_df = build_period_tables(total_annual_df, gas_annual_df)
    s3_annual_df, s3_period_df = build_s3_mitigation_tables(total_annual_df)

    plot_total_timeseries(total_annual_df)
    plot_period_cumulative(total_period_df)
    plot_s3_mitigation(s3_annual_df)
    plot_gas_stacked(gas_period_df)

    key = s3_period_df.loc[
        (s3_period_df["Model"] == "MME_Mean") & (s3_period_df["Period"] == "2025-2060")
    ]
    if not key.empty:
        avoided = float(key["Avoided_cumulative_Gg_CO2eq"].iloc[0])
        pct = float(key["Avoided_percent"].iloc[0])
        print(
            f"\n2025-2060 MME SSP585-Mitigation avoided emissions: "
            f"{avoided:,.2f} Gg CO2-eq ({pct:.2f}%)."
        )

    print("\nFuture main pH-median Fig 6 analysis finished.")


if __name__ == "__main__":
    main()
