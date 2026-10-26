"""
Pre-imputation water-quality comparison across aquaculture categories.

This script uses the raw enriched environmental dataset before imputation and
compares selected water-quality predictors across the same grouping variables
used in Fig. 2: location type, climate zone, system type and season.

Outputs:
  1. Summary statistics by variable and group.
  2. Kruskal-Wallis overall tests.
  3. Dunn-Bonferroni pairwise comparisons.
  4. Multi-panel violin/box plots for each grouping variable.
"""

from __future__ import annotations

from aquaculture_paths import project_path, external_path

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(str(project_path('meta')))
INPUT_CSV = project_path("literature/环境因子预测_Enriched_Final_v4_Corrected.csv")

OUTPUT_DIR = BASE_DIR / "Pre_imputation_water_quality_comparison"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"


VARIABLES = {
    "DO": {
        "source": "DO",
        "label": "Dissolved oxygen",
        "unit": "mg L\u207b\u00b9",
        "transform": "raw",
        "color": "#4C78A8",
    },
    "pH": {
        "source": "pH",
        "label": "pH",
        "unit": "",
        "transform": "raw",
        "color": "#7A5195",
    },
    "WaterTemp": {
        "source": "Watertempreature",
        "label": "Water temperature",
        "unit": "\u00b0C",
        "transform": "raw",
        "color": "#F58518",
    },
    "NH4N": {
        "source": "NH4N",
        "label": "NH\u2084\u207a-N",
        "unit": "mg L\u207b\u00b9",
        "transform": "log10p1",
        "color": "#54A24B",
    },
    "NO3N": {
        "source": "NO3N",
        "label": "NO\u2083\u207b-N",
        "unit": "mg L\u207b\u00b9",
        "transform": "log10p1",
        "color": "#E45756",
    },
    "Chla": {
        "source": "Chla",
        "label": "Chlorophyll-a",
        "unit": "\u00b5g L\u207b\u00b9",
        "transform": "log10p1",
        "color": "#72B7B2",
    },
}


CATEGORIES = {
    "Location_type": {
        "label": "Location type",
        "order": ["CAS", "IAS"],
    },
    "Climate_Zone": {
        "label": "Climate zone",
        "order": [
            "Warm-temperate\n(North)",
            "Subtropical\n(Central)",
            "Tropical/Subtropical\n(South)",
        ],
    },
    "System_type": {
        "label": "System type",
        "order": ["fish", "shrimp", "crab", "algae", "shellfish", "mixed"],
    },
    "Season_Clean": {
        "label": "Season",
        "order": ["Spring", "Summer", "Autumn", "Winter"],
    },
}


CLIMATE_LABELS = {
    "a": "Warm-temperate\n(North)",
    "b": "Subtropical\n(Central)",
    "c": "Tropical/Subtropical\n(South)",
}


def transform_values(values: pd.Series, transform: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if transform == "log10p1":
        numeric = numeric.where(numeric >= 0)
        return np.log10(numeric + 1)
    return numeric


def clean_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if "Row_ID" in df.columns and "Location_type" in df.columns:
        df.loc[df["Row_ID"] == 2081, "Location_type"] = "CPS"

    df["Season_Clean"] = (
        df["Season_Clean"]
        .astype("string")
        .str.lower()
        .str.title()
    )
    df = df[df["Season_Clean"] != "Annual"].copy()

    df["Climate_Zone"] = df["Climate_Zone"].replace(CLIMATE_LABELS)

    location_map = {"CPS": "CAS", "IPS": "IAS", "IAS": "IAS", "CAS": "CAS"}
    df["Location_type"] = df["Location_type"].replace(location_map)

    for category, meta in CATEGORIES.items():
        df[category] = pd.Categorical(df[category], categories=meta["order"], ordered=True)

    return df


def long_format(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, meta in VARIABLES.items():
        values_raw = pd.to_numeric(df[meta["source"]], errors="coerce")
        values_plot = transform_values(values_raw, meta["transform"])
        variable_df = df[
            ["Row_ID", "Study_ID", "Location_type", "Climate_Zone", "System_type", "Season_Clean"]
        ].copy()
        variable_df["Variable"] = variable
        variable_df["Variable_label"] = meta["label"]
        variable_df["Unit"] = meta["unit"]
        variable_df["Transform"] = meta["transform"]
        variable_df["Value_raw"] = values_raw
        variable_df["Value_plot"] = values_plot
        rows.append(variable_df)
    return pd.concat(rows, ignore_index=True)


def summarize_by_group(long_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for category, meta in CATEGORIES.items():
        total_counts = (
            long_df[["Row_ID", category]]
            .drop_duplicates()
            .groupby(category, observed=False)
            .size()
            .rename("n_total")
            .reset_index()
        )
        for variable, variable_df in long_df.groupby("Variable", sort=False):
            grouped = (
                variable_df
                .groupby(category, observed=False)["Value_raw"]
                .agg(
                    n_observed=lambda x: int(x.notna().sum()),
                    mean="mean",
                    median="median",
                    sd="std",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                    min="min",
                    max="max",
                )
                .reset_index()
            )
            grouped.insert(0, "Category", category)
            grouped.insert(0, "Variable", variable)
            grouped = grouped.merge(total_counts, on=category, how="left")
            grouped["missing_rate"] = 1 - grouped["n_observed"] / grouped["n_total"]
            grouped = grouped.rename(columns={category: "Group"})
            summary_rows.append(grouped)
    return pd.concat(summary_rows, ignore_index=True)


def dunn_bonferroni(groups: list[np.ndarray], labels: list[str]) -> pd.DataFrame:
    values = np.concatenate(groups)
    ranks = stats.rankdata(values)
    n_total = len(values)
    _, tie_counts = np.unique(values, return_counts=True)
    tie_correction = 1 - np.sum(tie_counts**3 - tie_counts) / (n_total**3 - n_total)

    starts = np.cumsum([0] + [len(group) for group in groups[:-1]])
    mean_ranks = {
        label: ranks[start:start + len(group)].mean()
        for label, group, start in zip(labels, groups, starts)
    }
    ns = {label: len(group) for label, group in zip(labels, groups)}

    rows = []
    for group1, group2 in combinations(labels, 2):
        denominator = np.sqrt(
            (n_total * (n_total + 1) / 12)
            * tie_correction
            * (1 / ns[group1] + 1 / ns[group2])
        )
        z_value = (mean_ranks[group1] - mean_ranks[group2]) / denominator
        p_value = 2 * stats.norm.sf(abs(z_value))
        rows.append(
            {
                "group1": group1,
                "group2": group2,
                "z": z_value,
                "p": p_value,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        _, p_adj, _, _ = multipletests(result["p"], method="bonferroni")
        result["p_adj"] = np.minimum(p_adj, 1.0)
        result["significance"] = pd.cut(
            result["p_adj"],
            bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
            labels=["***", "**", "*", "ns"],
        ).astype(str)
    return result


def run_statistics(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kw_rows = []
    dunn_rows = []

    for variable, variable_df in long_df.groupby("Variable", sort=False):
        for category, meta in CATEGORIES.items():
            stat_df = variable_df[[category, "Value_plot"]].dropna().copy()
            counts = stat_df.groupby(category, observed=False).size()
            valid_groups = [group for group in meta["order"] if counts.get(group, 0) >= 3]

            group_arrays = [
                stat_df.loc[stat_df[category] == group, "Value_plot"].to_numpy()
                for group in valid_groups
            ]

            if len(group_arrays) < 2:
                kw_rows.append(
                    {
                        "Variable": variable,
                        "Category": category,
                        "n_groups": len(group_arrays),
                        "n_observed": int(sum(len(group) for group in group_arrays)),
                        "statistic": np.nan,
                        "p": np.nan,
                    }
                )
                continue

            kw_stat, kw_p = stats.kruskal(*group_arrays)
            kw_rows.append(
                {
                    "Variable": variable,
                    "Category": category,
                    "n_groups": len(group_arrays),
                    "n_observed": int(sum(len(group) for group in group_arrays)),
                    "statistic": kw_stat,
                    "p": kw_p,
                }
            )

            pairwise = dunn_bonferroni(group_arrays, valid_groups)
            if not pairwise.empty:
                pairwise.insert(0, "Category", category)
                pairwise.insert(0, "Variable", variable)
                dunn_rows.append(pairwise)

    kw_results = pd.DataFrame(kw_rows)
    kw_results["p_label"] = kw_results["p"].apply(
        lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3g}")
    )
    dunn_results = pd.concat(dunn_rows, ignore_index=True) if dunn_rows else pd.DataFrame()
    return kw_results, dunn_results


def plot_category(long_df: pd.DataFrame, category: str, output_dir: Path) -> None:
    meta = CATEGORIES[category]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0))
    axes = axes.ravel()
    is_system_type = category == "System_type"
    min_n_for_violin = 10 if is_system_type else 1

    for ax, (variable, variable_meta) in zip(axes, VARIABLES.items()):
        plot_df = long_df[
            (long_df["Variable"] == variable)
            & long_df[category].notna()
            & long_df["Value_plot"].notna()
        ].copy()
        counts_all = plot_df.groupby(category, observed=False).size().reindex(meta["order"], fill_value=0)
        visible_order = [group for group in meta["order"] if counts_all.loc[group] >= min_n_for_violin]
        plot_df = plot_df[plot_df[category].isin(visible_order)].copy()
        plot_df[category] = pd.Categorical(plot_df[category], categories=visible_order, ordered=True)
        counts = counts_all.reindex(visible_order, fill_value=0)

        if not visible_order:
            ax.text(
                0.5,
                0.5,
                "No group with n \u2265 10",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#777777",
            )
            ax.set_title(variable_meta["label"], fontsize=10, fontweight="bold")
            ax.set_axis_off()
            continue

        sns.violinplot(
            data=plot_df,
            x=category,
            y="Value_plot",
            order=visible_order,
            color=variable_meta["color"],
            alpha=0.42 if is_system_type else 0.45,
            inner=None,
            linewidth=0,
            cut=0,
            ax=ax,
        )
        sns.boxplot(
            data=plot_df,
            x=category,
            y="Value_plot",
            order=visible_order,
            width=0.10 if is_system_type else 0.14,
            color="white",
            fliersize=0,
            linewidth=0.8,
            ax=ax,
            boxprops={"edgecolor": "#555555", "facecolor": "white", "alpha": 0.72 if is_system_type else 0.82},
            medianprops={"color": "#333333", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        label_y = y_min - 0.075 * y_range
        for tick_position, group_name in enumerate(visible_order):
            n_value = int(counts.loc[group_name])
            ax.text(
                tick_position,
                label_y,
                f"n={n_value}",
                ha="center",
                va="top",
                fontsize=6.8 if is_system_type else 7.5,
                color="#4A4A4A",
                clip_on=False,
            )
        ax.set_ylim(y_min - 0.17 * y_range, y_max)

        ylabel = variable_meta["label"]
        if variable_meta["unit"]:
            ylabel += f" ({variable_meta['unit']})"
        if variable_meta["transform"] == "log10p1":
            ylabel = f"log\u2081\u2080({variable_meta['label']} + 1)"

        ax.set_title(variable_meta["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
        ax.grid(axis="x", visible=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Pre-imputation water-quality distributions by {meta['label'].lower()}",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        (
            "Only observed pre-imputation values are shown. NH\u2084\u207a-N, NO\u2083\u207b-N and chlorophyll-a are plotted as log\u2081\u2080(x + 1). "
            + ("System-type groups with n < 10 for a given variable are not shown." if is_system_type else "")
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])

    safe_name = category.lower()
    for ext in ["png", "pdf", "svg"]:
        safe_savefig(fig, output_dir / f"Fig_pre_imputation_water_quality_by_{safe_name}.{ext}", dpi=600)
    plt.close(fig)


def plot_missing_rate_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    variable_order = list(VARIABLES.keys())
    category_order = list(CATEGORIES.keys())
    rows = []
    for variable in variable_order:
        for category in category_order:
            subset = summary[(summary["Variable"] == variable) & (summary["Category"] == category)]
            weighted_missing = (
                (subset["missing_rate"] * subset["n_total"]).sum() / subset["n_total"].sum()
                if subset["n_total"].sum() > 0
                else np.nan
            )
            rows.append({"Variable": variable, "Category": category, "Missing_rate": weighted_missing})
    heat_df = pd.DataFrame(rows).pivot(index="Variable", columns="Category", values="Missing_rate")
    heat_df = heat_df.loc[variable_order, category_order]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sns.heatmap(
        heat_df * 100,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Missing rate (%)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([CATEGORIES[c]["label"] for c in category_order], rotation=35, ha="right")
    ax.set_yticklabels([VARIABLES[v]["label"] for v in variable_order], rotation=0)
    ax.set_title("Pre-imputation missing rates of selected water-quality variables", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        safe_savefig(fig, output_dir / f"Fig_pre_imputation_water_quality_missing_rates.{ext}", dpi=600)
    plt.close(fig)


def safe_to_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Write a CSV, falling back to *_updated.csv if the target is open."""
    try:
        dataframe.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        dataframe.to_csv(fallback, index=False)
        print(f"Target CSV was locked; wrote fallback file: {fallback}")
        return fallback


def safe_savefig(fig: plt.Figure, path: Path, **kwargs) -> Path:
    """Save a figure, falling back to *_updated.ext if the target is open."""
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        fig.savefig(fallback, **kwargs)
        print(f"Target figure was locked; wrote fallback file: {fallback}")
        return fallback


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    df = clean_data()
    long_df = long_format(df)
    summary = summarize_by_group(long_df)
    kw_results, dunn_results = run_statistics(long_df)

    safe_to_csv(long_df, DATA_DIR / "pre_imputation_water_quality_long_data.csv")
    safe_to_csv(summary, TABLE_DIR / "Table_pre_imputation_water_quality_summary_by_group.csv")
    safe_to_csv(kw_results, TABLE_DIR / "Table_pre_imputation_water_quality_kruskal_wallis.csv")
    safe_to_csv(dunn_results, TABLE_DIR / "Table_pre_imputation_water_quality_dunn_bonferroni.csv")

    try:
        with pd.ExcelWriter(TABLE_DIR / "Tables_pre_imputation_water_quality_comparison.xlsx") as writer:
            summary.to_excel(writer, sheet_name="summary_by_group", index=False)
            kw_results.to_excel(writer, sheet_name="kruskal_wallis", index=False)
            dunn_results.to_excel(writer, sheet_name="dunn_bonferroni", index=False)
    except ModuleNotFoundError:
        print("Excel output skipped because no Excel writer engine is installed. CSV outputs were saved.")

    for category in CATEGORIES:
        plot_category(long_df, category, FIGURE_DIR)
    plot_missing_rate_heatmap(summary, FIGURE_DIR)

    print(f"Pre-imputation water-quality comparison complete: {OUTPUT_DIR}")
    print("Input rows after excluding annual records:", df.shape[0])
    print("Kruskal-Wallis significant tests:", int((kw_results["p"] < 0.05).sum()), "/", len(kw_results))


if __name__ == "__main__":
    main()
