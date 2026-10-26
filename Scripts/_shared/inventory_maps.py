# -*- coding: utf-8 -*-
"""
Common dashboard functions for aquaculture GHG maps.

This module is used by:
- actualCO2map_v2.py
- actualCH4map_v2.py
- actualN2Omap_v2.py
- actualGHGmap_v2.py

Main improvements compared with the original scripts:
1. Uses matplotlib Agg backend for stable PDF export.
2. Computes pond centroids in an equal-area projection, then converts back to WGS84.
3. Uses distance-weighted K-nearest grid sampling instead of simple K=5 average.
4. Uses 95% confidence intervals consistently.
5. Exports Monte Carlo mean values, not deterministic values mixed with Monte Carlo CIs.
6. Keeps flux map and emission bar chart clearly separated.
"""

from aquaculture_paths import project_path, external_path

import matplotlib
matplotlib.use("Agg")

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any

import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.spatial import cKDTree
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")


@dataclass
class GasMapConfig:
    gas: str
    nc_path: str
    flux_var: str
    flux_col: str
    flux_std_col: str
    emission_col: str
    flux_unit_label: str
    bar_xlabel: str
    map_title: str
    output_pdf_name: str
    output_csv_name: str
    conversion_divisor: float
    systematic_error: float
    output_float_format: str = "%.2f"
    clip_negative_flux_in_mc: bool = False
    use_diverging_cmap_if_negative: bool = True


# ==============================
# User-editable shared paths
# ==============================

# 请根据你自己的路径修改这里。
AQUACULTURE_SHP = str(external_path('download/全球水产养殖池塘遥感监测分布数据集（2020年）-数据实体/全球水产养殖池塘遥感监测分布数据集（2020年）-数据实体/Asia/aqu_AS_ds.shp'))

# 推荐以后下载到本地使用；如果这个路径不存在，代码会自动尝试使用阿里云在线边界。
CHINA_BOUNDARY_PATH = r""

OUTPUT_DIR = str(project_path("meta/Fig 5/6.3"))

# 分析时段
PERIOD_START = "2019-01-01"
PERIOD_END = "2024-12-31"

# 空间采样参数
K_NEIGHBORS = 5

# 蒙特卡洛参数
N_SIMULATIONS = 10000
RANDOM_SEED = 42
MC_CHUNK_SIZE = 2000

# 投影参数
AEA_CRS = "+proj=aea +lat_1=25 +lat_2=47 +lon_0=105"

# Match the tighter China map framing used in actualCH4map.py.
MAP_EXTENT = [77, 132, 16, 53]


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_china_boundary() -> Optional[gpd.GeoDataFrame]:
    """Read China/province boundary from local path or online fallback."""
    try:
        if CHINA_BOUNDARY_PATH and os.path.exists(CHINA_BOUNDARY_PATH):
            print(f">>> 读取本地中国边界: {CHINA_BOUNDARY_PATH}")
            boundary = gpd.read_file(CHINA_BOUNDARY_PATH).to_crs(epsg=4326)
        else:
            print(">>> 未设置本地边界或文件不存在，尝试读取阿里云在线中国边界...")
            aliyun_geojson = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
            boundary = gpd.read_file(aliyun_geojson).to_crs(epsg=4326)

        boundary["geometry"] = boundary.geometry.buffer(0)
        return boundary
    except Exception as e:
        print(f"      WARNING: 中国边界读取失败: {e}")
        return None


def prepare_ponds(shp_path: str, china_standard: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Read aquaculture ponds, filter China, calculate projected centroids and area."""
    print(">>> [1/7] 正在读取并提取中国养殖池塘区域...")

    gdf = gpd.read_file(shp_path)

    if "NAME_0" in gdf.columns:
        ponds = gdf[gdf["NAME_0"] == "China"].copy()
    else:
        print("      WARNING: 未找到 NAME_0 字段，默认使用全部矢量要素。")
        ponds = gdf.copy()

    ponds = ponds.explode(index_parts=False).reset_index(drop=True)
    ponds = ponds.to_crs(epsg=4326)

    # 先用等面积投影计算面积和 centroid，再转回 WGS84。
    ponds_aea = ponds.to_crs(AEA_CRS)
    ponds["area_m2"] = ponds_aea.geometry.area.astype(float)

    centroids_aea = ponds_aea.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_aea, crs=AEA_CRS).to_crs(epsg=4326)
    ponds["centroid_geom"] = centroids_wgs84.values
    ponds["lon"] = centroids_wgs84.x.values
    ponds["lat"] = centroids_wgs84.y.values

    # 用标准中国边界裁剪 centroid。
    if china_standard is not None:
        try:
            china_union = china_standard.geometry.unary_union
            within_mask = gpd.GeoSeries(ponds["centroid_geom"], crs=4326).within(china_union)
            ponds = ponds[within_mask.values].copy().reset_index(drop=True)
            print("      OK: 空间裁剪成功！")
        except Exception as e:
            print(f"      WARNING: 空间裁剪失败: {e}")

    return ponds


def read_period_mean_grid(nc_path: str, flux_var: str) -> xr.DataArray:
    """Read NetCDF and calculate mean flux during target period."""
    print(f">>> [2/7] 正在读取通量 NetCDF: {nc_path}")

    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"找不到 NetCDF 文件: {nc_path}")

    ds = xr.open_dataset(nc_path)

    if flux_var not in ds.data_vars:
        raise KeyError(f"NetCDF 中找不到变量 {flux_var}，当前变量包括: {list(ds.data_vars)}")

    if "time" in ds.coords:
        ds_period = ds.sel(time=slice(PERIOD_START, PERIOD_END))
    else:
        raise KeyError("NetCDF 缺少 time 坐标。")

    da = ds_period[flux_var].mean(dim="time", skipna=True)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise KeyError("通量数据必须包含 lat 和 lon 坐标。")

    return da


def distance_weighted_grid_sample(
    mean_grid: xr.DataArray,
    lon: np.ndarray,
    lat: np.ndarray,
    k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Sample gridded flux to pond centroids using inverse-distance weighted KNN."""
    print(f">>> 正在执行 K={k} 距离加权空间采样...")

    grid_lon, grid_lat = np.meshgrid(mean_grid.lon.values, mean_grid.lat.values)
    grid_vals = mean_grid.values

    valid_mask = ~np.isnan(grid_vals)
    valid_vals = grid_vals[valid_mask].astype(float)
    grid_points = np.column_stack((grid_lon[valid_mask], grid_lat[valid_mask]))

    if len(valid_vals) == 0:
        raise ValueError("通量网格没有有效值，无法采样。")

    kk = min(k, len(valid_vals))
    tree = cKDTree(grid_points)
    pond_points = np.column_stack((lon, lat))

    distances, indices = tree.query(pond_points, k=kk)

    if kk == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    sampled_vals = valid_vals[indices]

    weights = 1.0 / (distances + 1e-12)
    weights = weights / weights.sum(axis=1, keepdims=True)

    weighted_mean = np.sum(sampled_vals * weights, axis=1)
    weighted_var = np.sum(weights * (sampled_vals - weighted_mean[:, None]) ** 2, axis=1)
    weighted_std = np.sqrt(np.maximum(weighted_var, 0.0))

    return weighted_mean, weighted_std


def assign_provinces(ponds: gpd.GeoDataFrame, china_standard: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Spatial join pond centroids to provinces."""
    print(">>> [3/7] 正在执行省级空间区域关联...")

    if china_standard is None:
        ponds["Province"] = "Unknown"
        return ponds

    try:
        possible_name_cols = ["name", "NAME", "fullname", "省", "省名"]
        prov_name_col = None
        for col in possible_name_cols:
            if col in china_standard.columns:
                prov_name_col = col
                break
        if prov_name_col is None:
            prov_name_col = china_standard.columns[0]

        centroids = gpd.GeoDataFrame(
            ponds[["area_m2", "lon", "lat"]].copy(),
            geometry=gpd.GeoSeries(ponds["centroid_geom"], crs=4326),
            crs=4326
        )

        joined = gpd.sjoin(
            centroids,
            china_standard[[prov_name_col, "geometry"]],
            how="left",
            predicate="within"
        )

        ponds["Province"] = joined[prov_name_col].fillna("Other").values
    except Exception as e:
        print(f"      WARNING: 省级关联失败: {e}")
        ponds["Province"] = "Unknown"

    return ponds


def make_lognormal_multiplier(rng: np.random.Generator, n: int, relative_sd: float) -> np.ndarray:
    """Create lognormal multiplier with mean around 1 and specified relative SD."""
    mu_err, sigma_err = 1.0, relative_sd
    s_log = np.sqrt(np.log(1 + (sigma_err / mu_err) ** 2))
    m_log = np.log(mu_err) - (s_log ** 2 / 2)
    return rng.lognormal(mean=m_log, sigma=s_log, size=n)


def monte_carlo_emissions(
    ponds: gpd.GeoDataFrame,
    config: GasMapConfig,
    top_provinces: list[str]
) -> Dict[str, Any]:
    """Chunked Monte Carlo uncertainty propagation."""
    print(f">>> [4/7] 正在运行 {N_SIMULATIONS} 次蒙特卡洛误差传递...")

    rng = np.random.default_rng(RANDOM_SEED)

    means = ponds[config.flux_col].values.astype(np.float64)
    stds = ponds[config.flux_std_col].values.astype(np.float64)
    areas = ponds["area_m2"].values.astype(np.float64)
    province_labels = ponds["Province"].astype(str).values

    n_ponds = len(ponds)

    national_base = np.zeros(N_SIMULATIONS, dtype=np.float64)
    prov_base = {prov: np.zeros(N_SIMULATIONS, dtype=np.float64) for prov in top_provinces}

    for start in range(0, n_ponds, MC_CHUNK_SIZE):
        end = min(start + MC_CHUNK_SIZE, n_ponds)

        means_chunk = means[start:end]
        stds_chunk = stds[start:end]
        areas_chunk = areas[start:end]
        prov_chunk = province_labels[start:end]

        random_fluxes = rng.normal(
            loc=means_chunk,
            scale=stds_chunk,
            size=(N_SIMULATIONS, end - start)
        ).astype(np.float32)

        if config.clip_negative_flux_in_mc:
            random_fluxes = np.maximum(random_fluxes, 0)

        emission_matrix = (
            random_fluxes.astype(np.float64)
            * 24.0
            * 365.0
            * areas_chunk[None, :]
            / config.conversion_divisor
        )

        national_base += emission_matrix.sum(axis=1)

        for prov in top_provinces:
            mask = prov_chunk == prov
            if mask.any():
                prov_base[prov] += emission_matrix[:, mask].sum(axis=1)

    systematic_multiplier = make_lognormal_multiplier(
        rng=rng,
        n=N_SIMULATIONS,
        relative_sd=config.systematic_error
    )

    national_sims = national_base * systematic_multiplier

    national_stats = {
        "mean": float(np.mean(national_sims)),
        "p025": float(np.percentile(national_sims, 2.5)),
        "p975": float(np.percentile(national_sims, 97.5)),
    }

    province_stats = {}
    for prov in top_provinces:
        sims = prov_base[prov] * systematic_multiplier
        mean_val = float(np.mean(sims))
        p025 = float(np.percentile(sims, 2.5))
        p975 = float(np.percentile(sims, 97.5))
        province_stats[prov] = {
            "mean": mean_val,
            "p025": p025,
            "p975": p975,
            "err_low": max(mean_val - p025, 0),
            "err_high": max(p975 - mean_val, 0),
        }

    return {
        "national": national_stats,
        "province": province_stats,
    }


def export_summary(
    config: GasMapConfig,
    top_provinces: list[str],
    mc_stats: Dict[str, Any],
    output_dir: str
) -> pd.DataFrame:
    """Export Top10 provinces and national total CSV."""
    print(">>> [5/7] 正在导出省级 Top10 与国家总量表...")

    rows = []
    for prov in top_provinces:
        st = mc_stats["province"][prov]
        rows.append({
            "Province": prov,
            f"Mean_{config.gas}_Emission_Gg_yr": st["mean"],
            "Lower_CI_2_5th": st["p025"],
            "Upper_CI_97_5th": st["p975"],
        })

    df_export = pd.DataFrame(rows).sort_values(
        by=f"Mean_{config.gas}_Emission_Gg_yr",
        ascending=False
    )

    national = mc_stats["national"]
    total_row = pd.DataFrame({
        "Province": ["National Total"],
        f"Mean_{config.gas}_Emission_Gg_yr": [national["mean"]],
        "Lower_CI_2_5th": [national["p025"]],
        "Upper_CI_97_5th": [national["p975"]],
    })

    df_export = pd.concat([df_export, total_row], ignore_index=True)

    csv_path = os.path.join(output_dir, config.output_csv_name)
    df_export.to_csv(csv_path, index=False, float_format=config.output_float_format, encoding="utf-8-sig")
    print(f"      OK: 表格已保存至: {csv_path}")

    return df_export


def plot_dashboard(
    ponds: gpd.GeoDataFrame,
    china_standard: Optional[gpd.GeoDataFrame],
    config: GasMapConfig,
    top_provinces: list[str],
    mc_stats: Dict[str, Any],
    output_dir: str
) -> None:
    """Plot map + Top10 bar chart dashboard."""
    print(">>> [6/7] 正在绘制联合看板大图...")

    fig = plt.figure(figsize=(19, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.4, 1], wspace=0.12)
    albers_proj = ccrs.AlbersEqualArea(
        central_longitude=105,
        central_latitude=35,
        standard_parallels=(25, 47)
    )

    # --------------------------
    # Map
    # --------------------------
    ax_map = fig.add_subplot(gs[0], projection=albers_proj)
    ax_map.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    gl = ax_map.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.3,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    china_outline = None
    if china_standard is not None:
        china_standard.plot(
            ax=ax_map,
            facecolor="#F5F5F5",
            edgecolor="#DCDCDC",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=1
        )
        china_outline = gpd.GeoSeries([china_standard.geometry.unary_union], crs=china_standard.crs)
        china_outline.plot(
            ax=ax_map,
            facecolor="none",
            edgecolor="#555555",
            linewidth=0.9,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    vals = ponds[config.flux_col]
    vmin_val = vals.quantile(0.01)
    vmax_val = vals.quantile(0.99)

    cmap = "YlOrRd"
    norm = None

    if config.use_diverging_cmap_if_negative and vmin_val < 0 < vmax_val:
        cmap = "RdBu_r"
        vmax_abs = max(abs(float(vmin_val)), abs(float(vmax_val)))
        norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
        vmin_plot, vmax_plot = None, None
    else:
        vmin_plot, vmax_plot = vmin_val, vmax_val

    sc = ax_map.scatter(
        ponds["lon"],
        ponds["lat"],
        c=ponds[config.flux_col],
        s=3.0,
        cmap=cmap,
        norm=norm,
        vmin=vmin_plot,
        vmax=vmax_plot,
        edgecolors="none",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=4,
        rasterized=True
    )

    cbar = plt.colorbar(sc, ax=ax_map, orientation="vertical", shrink=0.45, pad=0.02)
    cbar.set_label(config.flux_unit_label, fontsize=11)

    # Inset map
    ax_inset = ax_map.inset_axes([0.86, 0.01, 0.12, 0.28], projection=albers_proj)
    ax_inset.set_extent([106, 124, 2, 24], crs=ccrs.PlateCarree())

    if china_standard is not None:
        china_standard.plot(
            ax=ax_inset,
            facecolor="#F5F5F5",
            edgecolor="#DCDCDC",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=1
        )
        if china_outline is not None:
            china_outline.plot(
                ax=ax_inset,
                facecolor="none",
                edgecolor="#555555",
                linewidth=0.9,
                transform=ccrs.PlateCarree(),
                zorder=2
            )

    ax_inset.scatter(
        ponds["lon"],
        ponds["lat"],
        c=ponds[config.flux_col],
        s=1.2,
        cmap=cmap,
        norm=norm,
        vmin=vmin_plot,
        vmax=vmax_plot,
        edgecolors="none",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=4,
        rasterized=True
    )

    for spine in ax_inset.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.6)

    ax_map.set_title(config.map_title, fontsize=15, pad=15, weight="bold")

    # --------------------------
    # Bar chart
    # --------------------------
    ax_bar = fig.add_subplot(gs[1])

    # Keep ascending order for horizontal bar plot.
    top_provinces_for_plot = top_provinces
    bar_means = np.array([mc_stats["province"][p]["mean"] for p in top_provinces_for_plot])
    err_low = np.array([mc_stats["province"][p]["err_low"] for p in top_provinces_for_plot])
    err_high = np.array([mc_stats["province"][p]["err_high"] for p in top_provinces_for_plot])
    xerr_array = np.vstack([err_low, err_high])

    if np.nanmax(bar_means) > np.nanmin(bar_means):
        bar_norm = plt.Normalize(vmin=np.nanmin(bar_means), vmax=np.nanmax(bar_means))
        bar_colors = plt.cm.YlOrRd(bar_norm(bar_means))
    else:
        bar_colors = plt.cm.YlOrRd(np.full_like(bar_means, 0.7, dtype=float))

    bars = ax_bar.barh(
        top_provinces_for_plot,
        bar_means,
        xerr=xerr_array,
        color=bar_colors,
        edgecolor="none",
        height=0.55,
        error_kw=dict(ecolor="#333333", lw=0.8, capsize=2, capthick=0.8)
    )

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_linewidth(0.7)
    ax_bar.spines["bottom"].set_linewidth(0.7)
    ax_bar.xaxis.grid(True, linestyle="--", linewidth=0.3, color="gray", alpha=0.4)
    ax_bar.set_axisbelow(True)

    ax_bar.set_xlabel(config.bar_xlabel, fontsize=11, labelpad=8)
    ax_bar.set_title("Top 10 Provinces Contribution\n(with 95% Confidence Interval)", fontsize=11, pad=12, weight="bold")
    ax_bar.tick_params(labelsize=10)

    max_mean = np.nanmax(bar_means)
    max_err = np.nanmax(err_high)
    label_pad = max(abs(max_mean), 1.0) * 0.02

    for idx, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width + err_high[idx] + label_pad
        fmt = config.output_float_format.replace("%", "{:").replace("f", "f}")
        try:
            label_text = fmt.format(width)
        except Exception:
            label_text = f"{width:.2f}"

        ax_bar.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=9,
            color="#333333"
        )

    x_max = (max_mean + max_err) * 1.15 if (max_mean + max_err) > 0 else 1.0
    ax_bar.set_xlim(0, x_max)

    # Save
    print(">>> [7/7] 正在保存 PDF...")
    output_pdf = os.path.join(output_dir, config.output_pdf_name)
    plt.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      OK: 图片已保存至: {output_pdf}")


def run_dashboard(config: GasMapConfig) -> None:
    """Complete pipeline for one gas dashboard."""
    ensure_output_dir(OUTPUT_DIR)

    china_standard = read_china_boundary()
    ponds = prepare_ponds(AQUACULTURE_SHP, china_standard)

    mean_grid = read_period_mean_grid(config.nc_path, config.flux_var)

    flux_mean, flux_std = distance_weighted_grid_sample(
        mean_grid=mean_grid,
        lon=ponds["lon"].values,
        lat=ponds["lat"].values,
        k=K_NEIGHBORS
    )

    ponds[config.flux_col] = flux_mean
    ponds[config.flux_std_col] = flux_std

    # Deterministic annual emission per pond for selecting Top10 provinces.
    ponds[config.emission_col] = (
        ponds[config.flux_col]
        * 24.0
        * 365.0
        * ponds["area_m2"]
        / config.conversion_divisor
    )

    ponds = assign_provinces(ponds, china_standard)

    prov_summary = ponds.groupby("Province")[config.emission_col].sum().reset_index()
    top10_df = prov_summary.sort_values(by=config.emission_col, ascending=True).tail(10)
    top_provinces = list(top10_df["Province"].values)

    mc_stats = monte_carlo_emissions(
        ponds=ponds,
        config=config,
        top_provinces=top_provinces
    )

    export_summary(
        config=config,
        top_provinces=top_provinces,
        mc_stats=mc_stats,
        output_dir=OUTPUT_DIR
    )

    plot_dashboard(
        ponds=ponds,
        china_standard=china_standard,
        config=config,
        top_provinces=top_provinces,
        mc_stats=mc_stats,
        output_dir=OUTPUT_DIR
    )

    print(">>> 全流程完毕！")
