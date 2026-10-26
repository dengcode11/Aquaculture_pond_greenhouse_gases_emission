from __future__ import annotations

from aquaculture_paths import project_path, external_path

import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(str(project_path('meta')))
WORK_DIR = Path(os.environ["GHG_WORK_DIR"])
OUTPUT_ROOT = Path(os.environ["GHG_OUTPUT_ROOT"])
CANDIDATE_CSV = Path(os.environ["GHG_CANDIDATE_CSV"])
TARGET_MODELS = [v.strip() for v in os.environ["GHG_TARGET_MODELS"].split(",") if v.strip()]
SCENARIOS = [v.strip() for v in os.environ["GHG_MANAGEMENT_SCENARIOS"].split(",") if v.strip()]

DATA_DIR = WORK_DIR / "data"
TABLE_DIR = WORK_DIR / "tables"
FIGURE_DIR = WORK_DIR / "figures"
TOTAL_DIR = OUTPUT_ROOT / "Total_GHG"
BASELINE_CSV = BASE_DIR / "Fig 6" / "6.4" / "future_main_pH_median_annual_total_emissions.csv"
BASELINE_GAS_CSV = BASE_DIR / "Fig 6" / "6.4" / "future_main_pH_median_gas_annual_co2eq.csv"
ORIGINAL_ANALYSIS = BASE_DIR / "03_analyze_future_main_pH_median.py"

PERIODS = [
    ("2025-2060", 2025, 2060),
    ("2061-2100", 2061, 2100),
    ("2025-2100", 2025, 2100),
]
SSP_LABELS = {"ssp126": "SSP126", "ssp245": "SSP245", "ssp585": "SSP585"}
GG_TO_TG = 1.0 / 1000.0
GAS_CONFIG = {
    "CO2": {"directory": "CO2_Management", "variable": "co2_flux", "scale": 1.0},
    "CH4": {"directory": "CH4_Management", "variable": "ch4_flux", "scale": 27.2},
    "N2O": {"directory": "N2O_Management", "variable": "n2o_flux", "scale": 273.0 / 1000.0},
}

for directory in [DATA_DIR, TABLE_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_original_analysis():
    import future_integration
    return future_integration


def total_path(model: str, scenario: str, candidate_id: str) -> Path:
    return TOTAL_DIR / f"Total_CO2eq_China_{model}_{scenario}_{candidate_id}_OptimizedBounds.nc"


def build_annual_table(candidates: pd.DataFrame) -> pd.DataFrame:
    original = load_original_analysis()
    pond_points = original.load_china_pond_points()
    rows = []

    for _, candidate in candidates.iterrows():
        candidate_id = str(candidate["Candidate_ID"])
        for model in TARGET_MODELS:
            for scenario in SCENARIOS:
                path = total_path(model, scenario, candidate_id)
                if not path.exists():
                    raise FileNotFoundError(path)
                print(f"Integrating ponds: {model} | {scenario} | {candidate_id}")
                annual = original.annual_emissions_from_flux(path, "co2eq_flux", pond_points)
                annual = annual.loc[(annual.index >= 2025) & (annual.index <= 2100)]
                for year, value in annual.items():
                    rows.append(
                        {
                            "Model": model,
                            "SSP": scenario,
                            "SSP_label": SSP_LABELS[scenario],
                            "Candidate_ID": candidate_id,
                            "Candidate_label": candidate["Candidate_label"],
                            "DO_increase_percent": candidate["DO_increase"] * 100,
                            "NH4_reduction_percent": candidate["NH4_reduction"] * 100,
                            "NO3_reduction_percent": candidate["NO3_reduction"] * 100,
                            "pH_target": candidate["pH_target"],
                            "Year": int(year),
                            "Management_Gg_CO2eq_yr": float(value),
                        }
                    )

    annual = pd.DataFrame(rows)
    annual.to_csv(DATA_DIR / "annual_management_emissions.csv", index=False, encoding="utf-8-sig")
    return annual


def load_baseline() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline.loc[
        baseline["Model"].isin(TARGET_MODELS)
        & baseline["Scenario"].isin(SCENARIOS)
        & baseline["Year"].between(2025, 2100)
    ].copy()
    baseline = baseline.rename(
        columns={"Scenario": "SSP", "Annual_Gg_CO2eq_yr": "Baseline_Gg_CO2eq_yr"}
    )
    return baseline[["Model", "SSP", "Year", "Baseline_Gg_CO2eq_yr"]]


def summarize(annual: pd.DataFrame) -> pd.DataFrame:
    merged = annual.merge(load_baseline(), on=["Model", "SSP", "Year"], how="left")
    if merged["Baseline_Gg_CO2eq_yr"].isna().any():
        raise RuntimeError("Baseline emissions are missing for one or more model-scenario-year records.")
    merged["Avoided_Gg_CO2eq_yr"] = (
        merged["Baseline_Gg_CO2eq_yr"] - merged["Management_Gg_CO2eq_yr"]
    )
    merged.to_csv(TABLE_DIR / "Table_annual_avoided_emissions.csv", index=False, encoding="utf-8-sig")

    group_columns = [
        "Model",
        "SSP",
        "SSP_label",
        "Candidate_ID",
        "Candidate_label",
        "DO_increase_percent",
        "NH4_reduction_percent",
        "NO3_reduction_percent",
        "pH_target",
    ]
    rows = []
    for period, start, end in PERIODS:
        subset = merged.loc[merged["Year"].between(start, end)]
        for keys, group in subset.groupby(group_columns, dropna=False):
            baseline = group["Baseline_Gg_CO2eq_yr"].sum() * GG_TO_TG
            management = group["Management_Gg_CO2eq_yr"].sum() * GG_TO_TG
            row = dict(zip(group_columns, keys))
            row.update(
                {
                    "Period": period,
                    "Baseline_cumulative_Tg_CO2eq": baseline,
                    "Management_cumulative_Tg_CO2eq": management,
                    "Avoided_cumulative_Tg_CO2eq": baseline - management,
                    "Reduction_percent": (baseline - management) / baseline * 100,
                }
            )
            rows.append(row)

    by_model = pd.DataFrame(rows)
    by_model.to_csv(TABLE_DIR / "Table_period_results_by_GCM.csv", index=False, encoding="utf-8-sig")

    summary_columns = [
        "SSP",
        "SSP_label",
        "Candidate_ID",
        "Candidate_label",
        "DO_increase_percent",
        "NH4_reduction_percent",
        "NO3_reduction_percent",
        "pH_target",
        "Period",
    ]
    summary = by_model.groupby(summary_columns, as_index=False, dropna=False).agg(
        Baseline_cumulative_Tg_CO2eq_mean=("Baseline_cumulative_Tg_CO2eq", "mean"),
        Management_cumulative_Tg_CO2eq_mean=("Management_cumulative_Tg_CO2eq", "mean"),
        Avoided_cumulative_Tg_CO2eq_mean=("Avoided_cumulative_Tg_CO2eq", "mean"),
        Avoided_cumulative_Tg_CO2eq_sd=("Avoided_cumulative_Tg_CO2eq", "std"),
        Reduction_percent_mean=("Reduction_percent", "mean"),
        Reduction_percent_sd=("Reduction_percent", "std"),
    )
    summary.to_csv(TABLE_DIR / "Table_period_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def summarize_gas_contributions(candidates: pd.DataFrame) -> None:
    original = load_original_analysis()
    pond_points = original.load_china_pond_points()
    baseline = pd.read_csv(BASELINE_GAS_CSV)
    baseline = baseline.loc[
        baseline["Model"].isin(TARGET_MODELS)
        & baseline["Scenario"].isin(SCENARIOS)
        & baseline["Year"].between(2025, 2100)
    ].copy()

    rows = []
    for _, candidate in candidates.iterrows():
        candidate_id = str(candidate["Candidate_ID"])
        for gas, config in GAS_CONFIG.items():
            for model in TARGET_MODELS:
                for scenario in SCENARIOS:
                    path = (
                        OUTPUT_ROOT
                        / config["directory"]
                        / f"{gas}_China_{model}_{scenario}_{candidate_id}_OptimizedBounds.nc"
                    )
                    if not path.exists():
                        raise FileNotFoundError(path)
                    annual = original.annual_emissions_from_flux(
                        path,
                        config["variable"],
                        pond_points,
                        flux_scale=config["scale"],
                    )
                    annual = annual.loc[annual.index.to_series().between(2025, 2100)]
                    management = annual.sum() * GG_TO_TG
                    baseline_value = baseline.loc[
                        (baseline["Gas"] == gas)
                        & (baseline["Model"] == model)
                        & (baseline["Scenario"] == scenario),
                        "Annual_Gg_CO2eq_yr",
                    ].sum() * GG_TO_TG
                    rows.append(
                        {
                            "Gas": gas,
                            "Model": model,
                            "SSP": scenario,
                            "SSP_label": SSP_LABELS[scenario],
                            "Candidate_ID": candidate_id,
                            "Candidate_label": candidate["Candidate_label"],
                            "Baseline_cumulative_Tg_CO2eq": baseline_value,
                            "Management_cumulative_Tg_CO2eq": management,
                            "Avoided_cumulative_Tg_CO2eq": baseline_value - management,
                        }
                    )

    by_model = pd.DataFrame(rows)
    by_model.to_csv(
        TABLE_DIR / "Table_gas_avoided_emissions_by_GCM_2025_2100.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = by_model.groupby(
        ["Gas", "SSP", "SSP_label", "Candidate_ID", "Candidate_label"],
        as_index=False,
    ).agg(
        Avoided_cumulative_Tg_CO2eq_mean=("Avoided_cumulative_Tg_CO2eq", "mean"),
        Avoided_cumulative_Tg_CO2eq_sd=("Avoided_cumulative_Tg_CO2eq", "std"),
    )
    summary.to_csv(
        TABLE_DIR / "Table_gas_avoided_emissions_summary_2025_2100.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_summary(summary: pd.DataFrame) -> None:
    plot_data = summary.loc[summary["Period"] == "2025-2100"].copy()
    candidates = list(dict.fromkeys(plot_data["Candidate_label"]))
    scenarios = list(dict.fromkeys(plot_data["SSP_label"]))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(candidates)))

    figure, axis = plt.subplots(figsize=(max(7.2, len(scenarios) * 2.0), 4.8))
    x = np.arange(len(scenarios))
    width = 0.78 / max(len(candidates), 1)

    for index, (candidate, color) in enumerate(zip(candidates, colors)):
        values = []
        errors = []
        for scenario in scenarios:
            row = plot_data.loc[
                (plot_data["Candidate_label"] == candidate)
                & (plot_data["SSP_label"] == scenario)
            ]
            values.append(row["Avoided_cumulative_Tg_CO2eq_mean"].iloc[0])
            error = row["Avoided_cumulative_Tg_CO2eq_sd"].iloc[0]
            errors.append(0 if pd.isna(error) else error)
        offset = (index - (len(candidates) - 1) / 2) * width
        axis.bar(x + offset, values, width, yerr=errors, capsize=3, label=candidate, color=color)

    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set_xticks(x, scenarios)
    axis.set_ylabel("Cumulative avoided emissions (Tg CO$_2$-eq)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "Fig_management_summary.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "Fig_management_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    candidates = pd.read_csv(CANDIDATE_CSV)
    annual = build_annual_table(candidates)
    summary = summarize(annual)
    summarize_gas_contributions(candidates)
    plot_summary(summary)
    print(f"Completed. Results: {WORK_DIR}")


if __name__ == "__main__":
    main()
