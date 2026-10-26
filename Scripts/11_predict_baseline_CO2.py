# -*- coding: utf-8 -*-
"""
CO2 现状基准核算系统 (Baseline 2011-2024 全量版)
功能: 提取 ERA5 前 168 个月，结合 CO2 的专属特征，气压严格对齐 hPa
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
SMEARING_FACTOR = 1.57
CHLA_STATIC_VALUE = 50.43
MODEL_PATH = "LGBM_China_CO2_Best_Param_v2.pkl"
ERA5_FILE_PATH = str(project_path('meta/era5_2011_2024.nc'))
OUTPUT_DIR = "./CO2_China_Projections_Final_pH_median"
PH_MEDIAN_VALUE = 8.04
ELEVATION_FILE = str(project_path("meta/elev.1-deg.nc"))


# ================= 2. 科学计算函数 =================
def calculate_pressure_hPa(elevation_m):
    """🔥 气压修正: 海拔(m) -> 气压(hPa) (严格对齐你的训练集)"""
    return 101.325 * np.power((1 - 2.25577e-5 * elevation_m), 5.25588) * 10.0


def calculate_solar_radiation_FAO56(lat_array, day_of_year):
    doy = np.array(day_of_year)
    phi = np.deg2rad(lat_array)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    tan_phi_delta = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(tan_phi_delta)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return np.maximum(0, Ra * 0.6 * 11.574)


def calculate_do_graham2025(temp_k, elevation_m):
    Tw = np.maximum(temp_k, 273.15)
    C0, C1, C2 = -139.34411, 1.575701e5, -6.642308e7
    C3, C4, C_elev = 1.243800e10, -8.621949e11, 0.0001148
    exponent = C0 + (C1 / Tw) + (C2 / Tw ** 2) + (C3 / Tw ** 3) + (C4 / Tw ** 4)
    return np.maximum(np.exp(exponent) * (1 - (C_elev * elevation_m)), 0)


def calculate_ph_cai2011(do_val):
    return np.full_like(do_val, PH_MEDIAN_VALUE, dtype=float)


def inverse_log_modulus(x):
    return np.sign(x) * (np.power(10, np.abs(x)) - 1)


# ================= 3. 执行逻辑 =================
def run_era5_co2_baseline_all_years():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(">>> 正在加载 CO2 LGBM 模型...")
    lgbm_model = joblib.load(MODEL_PATH)

    ds_elev = xr.open_dataset(ELEVATION_FILE, decode_times=False).squeeze()
    if ds_elev.lon.max() > 180: ds_elev.coords['lon'] = (ds_elev.coords['lon'] + 180) % 360 - 180
    ds_elev_china = ds_elev.sortby(['lat', 'lon']).sel(lat=slice(15, 55), lon=slice(70, 140))
    elev_var_name = list(ds_elev_china.data_vars)[0]

    print(f">>> 正在解析 ERA5 数据: {os.path.basename(ERA5_FILE_PATH)}")
    ds_era5 = xr.open_dataset(ERA5_FILE_PATH, decode_times=False).squeeze()
    if 'valid_time' in ds_era5.dims:
        ds_era5 = ds_era5.rename({'valid_time': 'time'})
    elif 'valid_time' in ds_era5.coords:
        ds_era5 = ds_era5.rename({'valid_time': 'time'})

    try:
        ds_era5 = ds_era5.sel(latitude=slice(55, 15), longitude=slice(70, 140))
    except:
        ds_era5 = ds_era5.sel(latitude=slice(15, 55), longitude=slice(70, 140))

    TOTAL_MONTHS = 168
    print(f">>> 正在提取 2011-2024 年全量数据 (第 0-{TOTAL_MONTHS - 1} 层)...")
    ds_era5_target = ds_era5.isel(time=slice(0, TOTAL_MONTHS))

    elev_regrid = ds_elev_china[elev_var_name].interp(lat=ds_era5_target.latitude, lon=ds_era5_target.longitude,
                                                      method='linear').values
    if elev_regrid.ndim > 2: elev_regrid = np.squeeze(elev_regrid)

    lon_grid, lat_grid = np.meshgrid(ds_era5_target.longitude.values, ds_era5_target.latitude.values)
    pred_list = []

    clean_time_axis = pd.date_range(start='2011-01-01', periods=TOTAL_MONTHS, freq='MS')
    doy_series = clean_time_axis.dayofyear

    print(f">>> 开始计算 CO2 全时段精准排放...")
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

        chla_val = np.full_like(air_c, CHLA_STATIC_VALUE)
        # 🔥 使用修正后的 hPa 气压公式 🔥
        pressure_hPa = calculate_pressure_hPa(elev_m)
        rad_val = calculate_solar_radiation_FAO56(lat_grid[mask], doy_series[i])

        X_input = pd.DataFrame({
            "DO": do_val, "pH": ph_val, "chla": chla_val,
            "Watertempreature": water_c, "Airtemperature_Filled": air_c,
            "Monthly_Precipitation_ERA5": pr_mm, "Windspeed_Filled": wind_val,
            "ERA5_Radiation": rad_val,
            "Atomsphericpressure_Filled": pressure_hPa,  # 严格对齐 hPa
            "ERA5_SoilTemp": air_c
        })

        y_pred_real = inverse_log_modulus(lgbm_model.predict(X_input)) * SMEARING_FACTOR
        res_frame = np.full_like(t2m_k, np.nan)
        res_frame[mask] = y_pred_real
        pred_list.append(res_frame)

    output_path = os.path.join(OUTPUT_DIR, "CO2_China_ERA5_Baseline_2011_2024.nc")
    da_res = xr.DataArray(
        np.array(pred_list), dims=['time', 'lat', 'lon'],
        coords={'time': clean_time_axis, 'lat': ds_era5_target.latitude.values, 'lon': ds_era5_target.longitude.values},
        name='co2_flux',
        attrs={'units': 'mg m-2 h-1', 'smearing_factor': SMEARING_FACTOR}
    )
    da_res.to_netcdf(output_path)
    print(f"\n🎉 完美收工！CO2 基准网格已生成: {output_path}")


if __name__ == "__main__":
    run_era5_co2_baseline_all_years()
