from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"SSP126": "#2CA25F", "SSP245": "#3182BD", "SSP585": "#DE2D26", "SSP585-High": "#756BB1"}
PATHWAYS = ["Baseline", "Low", "Medium", "High"]


def future_panels(root, output):
    annual = pd.read_csv(root / "Source_Data/Fig_5/annual_emissions.csv")
    cumulative = pd.read_csv(root / "Source_Data/Fig_5/cumulative_emissions.csv")
    response = pd.read_csv(root / "Source_Data/Fig_5/reduction_rates.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for series, frame in annual.groupby("Series", sort=False):
        axes[0].plot(frame.Year, frame.Mean_Tg_CO2eq_yr, color=COLORS[series], label=series)
        axes[0].fill_between(frame.Year, frame.Min_Tg_CO2eq_yr, frame.Max_Tg_CO2eq_yr, color=COLORS[series], alpha=0.12)
    axes[0].axvline(2060, color="gray", linestyle="--")
    axes[0].set(xlabel="Year", ylabel="Annual emissions (Tg CO₂-eq yr⁻¹)", title="(a) Annual emissions")
    axes[0].legend(frameon=False, fontsize=7)
    positions = np.arange(3)
    for pathway_index, pathway in enumerate(PATHWAYS):
        frame = cumulative.loc[cumulative.Series.eq(pathway)].set_index("SSP").loc[["ssp126", "ssp245", "ssp585"]]
        axes[1].bar(positions + (pathway_index - 1.5) * 0.18, frame.Cumulative_mean, width=0.18, yerr=frame.Cumulative_sd, capsize=2, label=pathway, color=["#C9C9C9", "#9ECAE1", "#4292C6", "#08519C"][pathway_index])
    axes[1].set_xticks(positions, ["SSP126", "SSP245", "SSP585"])
    axes[1].set(ylabel="Cumulative emissions (Tg CO₂-eq)", title="(b) Cumulative emissions")
    axes[1].legend(frameon=False, fontsize=7)
    for scenario, frame in response.groupby("SSP", sort=False):
        frame = frame.set_index("Candidate_label").loc[["Low", "Medium", "High"]]
        offset = {"ssp126": -0.04, "ssp245": 0, "ssp585": 0.04}[scenario]
        axes[2].errorbar(positions + offset, frame.Reduction_percent_mean, yerr=frame.Reduction_percent_sd, marker="o", capsize=3, label=scenario.upper(), color=COLORS[scenario.upper()])
    axes[2].set_xticks(positions, ["Low", "Medium", "High"])
    axes[2].set(ylabel="Emission reduction (%)", title="(c) Management response")
    upper = (response.Reduction_percent_mean + response.Reduction_percent_sd).max()
    lower = (response.Reduction_percent_mean - response.Reduction_percent_sd).min()
    axes[2].set_ylim(lower - 1, upper + 1)
    axes[2].legend(frameon=False, fontsize=7)
    figure.savefig(output / "Fig5_abc_saved_data.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig5_abc_saved_data.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def management_panels(root, output):
    combinations = pd.read_csv(root / "Source_Data/Fig_6/management_combinations.csv")
    curves = pd.read_csv(root / "Source_Data/Fig_6/allocation_curves.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].barh(combinations.Label, combinations.Avoided_Tg, color="#506F9B")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Cumulative avoided emissions (Tg CO₂-eq)", title="(a) Management combinations")
    axes[1].fill_between(curves.Coverage_percent, curves.Random_p025, curves.Random_p975, color="#DFDFDF", label="Random 95% range")
    axes[1].plot(curves.Coverage_percent, curves.Random_mean, linestyle=":", color="gray", label="Random mean")
    axes[1].plot(curves.Coverage_percent, curves["Baseline priority"], linestyle="--", color="#506F9B", label="Baseline priority")
    axes[1].plot(curves.Coverage_percent, curves["Mitigation priority"], color="#43833F", label="Mitigation priority")
    axes[1].set(xlabel="Managed pond area (%)", ylabel="Avoided emissions (Tg CO₂-eq)", title="(b) Spatial prioritization", xlim=(0, 100))
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(output / "Fig6_ab_saved_data.pdf", bbox_inches="tight")
    figure.savefig(output / "Fig6_ab_saved_data.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Redraw selected non-map main panels from saved source values, without model reruns.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = arguments.output_dir.resolve() if arguments.output_dir else root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    future_panels(root, output)
    management_panels(root, output)
    print(f"Saved panel checks to {output}; maps and manuscript assembly are not regenerated.")


if __name__ == "__main__":
    main()
