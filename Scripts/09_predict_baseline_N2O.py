# -*- coding: utf-8 -*-
"""
ERA5 现状基准核算系统 (Baseline 2011-2024 全量版)
功能: 强力绕过源文件损坏的时间坐标，精确预测 2011 到 2024 年共 14 年的逐月网格数据
"""

from aquaculture_paths import project_path, external_path

import os
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 核心路径与参数配置 =================
SMEARING_FACTOR = 2.012610
MODEL_PATH = "RF_China_N2O_BayesTuned.pkl"

# 纯英文安全路径
ERA5_FILE_PATH = str(project_path('meta/era5_2011_2024.nc'))

OUTPUT_DIR = "./N2O_China_Projections_Final_pH_median"
PH_MEDIAN_VALUE = 8.04
ELEVATION_FILE = str(project_path("meta/elev.1-deg.nc"))

# 【学术设定】：使用全国中位数 (Median)
BASELINE_NH4N = 0.57
BASELINE_NO3N = 0.93


# ================= 2. 科学计算函数 =================
def calculate_do_graham2025(temp_k, elevation_m):
    Tw = np.maximum(temp_k, 273.15)
    C0, C1, C2 = -139.34411, 1.575701e5, -6.642308e7
    C3, C4, C_elev = 1.243800e10, -8.621949e11, 0.0001148
    exponent = C0 + (C1 / Tw) + (C2 / Tw ** 2) + (C3 / Tw ** 3) + (C4 / Tw ** 4)
    do_sat = np.exp(exponent)
    return np.maximum(do_sat * (1 - (C_elev * elevation_m)), 0)


def calculate_ph_cai2011(do_val):
    return np.full_like(do_val, PH_MEDIAN_VALUE, dtype=float)


def inverse_log_modulus(x):
    return np.sign(x) * (np.power(10, np.abs(x)) - 1)


# ================= 3. 执行逻辑 =================
def run_era5_baseline_all_years():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(">>> 正在加载随机森林模型...")
    rf_model = joblib.load(MODEL_PATH)

    # 1. 准备高程数据并修复潜在的隐藏维度
    print(">>> 正在处理高程数据...")
    ds_elev = xr.open_dataset(ELEVATION_FILE, decode_times=False).squeeze()
    if ds_elev.lon.max() > 180:
        ds_elev.coords['lon'] = (ds_elev.coords['lon'] + 180) % 360 - 180
    ds_elev_china = ds_elev.sortby(['lat', 'lon']).sel(lat=slice(15, 55), lon=slice(70, 140))
    elev_var_name = list(ds_elev_china.data_vars)[0]

    # 2. 读取你的 ERA5 原始文件 (关闭时间解码，防止崩溃)
    print(f">>> 正在解析 ERA5 数据: {os.path.basename(ERA5_FILE_PATH)}")
    ds_era5 = xr.open_dataset(ERA5_FILE_PATH, decode_times=False).squeeze()

    if 'valid_time' in ds_era5.dims:
        ds_era5 = ds_era5.rename({'valid_time': 'time'})
    elif 'valid_time' in ds_era5.coords:
        ds_era5 = ds_era5.rename({'valid_time': 'time'})

    # 适配 ERA5 常见的坐标命名 (latitude/longitude)
    try:
        ds_era5 = ds_era5.sel(latitude=slice(55, 15), longitude=slice(70, 140))
    except:
        ds_era5 = ds_era5.sel(latitude=slice(15, 55), longitude=slice(70, 140))

    # 🔥 核心修改：提取 2011 到 2024 年共 14 年数据 (14*12 = 168个月)
    print(">>> 正在提取 2011-2024 年全量气象数据 (第 0-167 层)...")
    TOTAL_MONTHS = 168
    ds_era5_target = ds_era5.isel(time=slice(0, TOTAL_MONTHS))

    elev_regrid = ds_elev_china[elev_var_name].interp(
        lat=ds_era5_target.latitude, lon=ds_era5_target.longitude, method='linear').values

    if elev_regrid.ndim > 2:
        elev_regrid = np.squeeze(elev_regrid)

    lon_grid, lat_grid = np.meshgrid(ds_era5_target.longitude.values, ds_era5_target.latitude.values)

    pred_list = []

    # 3. 逐月预测循环 (共 168 个月)
    print(f">>> 开始计算全时段精准排放 (共 {TOTAL_MONTHS} 个月)...")
    for i in tqdm(range(TOTAL_MONTHS)):
        t2m_k = ds_era5_target['t2m'].isel(time=i).values
        tp_m = ds_era5_target['tp'].isel(time=i).values
        u10 = ds_era5_target['u10'].isel(time=i).values
        v10 = ds_era5_target['v10'].isel(time=i).values

        mask = ~np.isnan(t2m_k)
        if not mask.any():
            pred_list.append(np.full_like(t2m_k, np.nan));
            continue

        air_c = t2m_k[mask] - 273.15
        pr_mm = tp_m[mask] * 1000 * 30
        wind_val = np.sqrt(u10[mask] ** 2 + v10[mask] ** 2)
        elev_m = elev_regrid[mask]

        water_c = 0.4180 * air_c + 12.9454
        do_val = calculate_do_graham2025(t2m_k[mask], elev_m)
        ph_val = calculate_ph_cai2011(do_val)

        X_input = pd.DataFrame({
            "Watertempreature": water_c,
            "pH": ph_val,
            "DO": do_val,
            "NH4N": BASELINE_NH4N,
            "NO3N": BASELINE_NO3N,
            "MAT_Final": air_c,
            "MAP_Final": pr_mm * 12,
            "Monthly_Precipitation_ERA5": pr_mm,
            "Windspeed_Filled": wind_val,
            "Longitude": lon_grid[mask]
        })

        y_pred_log = rf_model.predict(X_input)
        y_pred_real = inverse_log_modulus(y_pred_log) * SMEARING_FACTOR

        res_frame = np.full_like(t2m_k, np.nan)
        res_frame[mask] = y_pred_real
        pred_list.append(res_frame)

    # 4. 生成最终的现状基准网格
    output_path = os.path.join(OUTPUT_DIR, "N2O_China_ERA5_Baseline_2011_2024.nc")

    # 🔥 核心修改：生成包含 168 个月的干净时间轴
    clean_time_axis = pd.date_range(start='2011-01-01', periods=TOTAL_MONTHS, freq='MS')

    da_res = xr.DataArray(
        np.array(pred_list),
        dims=['time', 'lat', 'lon'],
        coords={
            'time': clean_time_axis,
            'lat': ds_era5_target.latitude.values,
            'lon': ds_era5_target.longitude.values
        },
        name='n2o_flux',
        attrs={
            'units': 'ug m-2 h-1',
            'description': 'Baseline N2O flux from ERA5 (2011-2024)',
            'nh4n_median': BASELINE_NH4N,
            'no3n_median': BASELINE_NO3N
        }
    )

    da_res.to_netcdf(output_path)
    print(f"\n🎉 完美收工！14年全量版基准网格已生成: {output_path}")


if __name__ == "__main__":
    run_era5_baseline_all_years()
