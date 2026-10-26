# -*- coding: utf-8 -*-
"""
CH4 现状基准核算系统 Baseline 2011–2024 全量版

功能：
1. 读取 ERA5 2011–2024 月尺度气象数据；
2. 读取高程数据；
3. 计算 CH4 模型所需的派生环境因子；
4. 读取 CH4 的 pkl 机器学习模型；
5. 逐月预测中国区域 CH4 通量；
6. 输出 CH4_China_ERA5_Baseline_2011_2024.nc。

说明：
- 本脚本是按照你 CO2 和 N2O 的代码结构改写的 CH4 预测版；
- 输出的是单位面积通量，不是总排放；
- 后续总排放需要再乘以水产养殖面积栅格或矢量斑块面积。
"""

from aquaculture_paths import project_path, external_path

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 1. 核心路径与参数配置
# ============================================================

# ！！！请改成你的 CH4 模型路径
MODEL_PATH = "RF_China_CH4_Best_Param_v2.pkl"

# ERA5 文件路径，沿用你 CO2 和 N2O 的数据
ERA5_FILE_PATH = str(project_path('meta/era5_2011_2024.nc'))

# 高程文件
ELEVATION_FILE = str(project_path("meta/elev.1-deg.nc"))

# 输出文件夹
OUTPUT_DIR = "./CH4_China_Projections_Final_pH_median"

# 输出文件名
OUTPUT_NC_NAME = "CH4_China_ERA5_Baseline_2011_2024.nc"
PH_MEDIAN_VALUE = 8.04

# 历史期月份数量：2011–2024 共 14 年 × 12 月 = 168 个月
TOTAL_MONTHS = 168
START_TIME = "2011-01-01"

# ============================================================
# 2. CH4 模型变换参数
# ============================================================

# 如果你的 CH4 模型训练时使用了 log-modulus 转换，则设置为 True；
# 如果模型已经直接预测真实 CH4 通量，则设置为 False。
APPLY_INVERSE_LOG_MODULUS = True

# 如果你的 CH4 模型训练后做过 smearing correction，请填入对应值；
# 如果没有，则保持 1.0。
SMEARING_FACTOR = 1.304119

# 是否将负值通量截断为 0。
# 水产养殖 CH4 通常作为排放源，建议 True；
# 如果你的观测数据允许 CH4 吸收，则改为 False。
CLIP_NEGATIVE_TO_ZERO = True

# CH4 输出单位。
# 这里按照你原来 CH4 面积加权代码中的色标单位设置为 mg m-2 h-1。
OUTPUT_UNITS = "mg m-2 h-1"


# ============================================================
# 3. 固定背景参数
# ============================================================

# 如果 CH4 模型中用到了 chla，但未来/历史栅格中没有 chla，
# 可以使用训练集或观测数据的中位数/均值。
CHLA_STATIC_VALUE = 50.43

# 如果你的 CH4 模型用到了 NH4N、NO3N、COD、TOC、TN、TP 等变量，
# 可以在这里添加固定背景值。
# 例如：
# EXTRA_STATIC_FEATURES = {
#     "NH4N": 0.57,
#     "NO3N": 0.93,
#     "COD": 20.0
# }
EXTRA_STATIC_FEATURES = {}


# ============================================================
# 4. CH4 模型输入变量顺序
# ============================================================

"""
非常重要：
CH4_FEATURE_COLUMNS 必须和你训练 CH4 模型时的变量名、变量顺序完全一致。

如果你的 pkl 模型中保存了 feature_names_in_，脚本会优先自动读取；
如果无法自动读取，就会使用下面这个列表。

你需要根据自己的 CH4 训练表格修改这个列表。
下面先给出一个和 CO2/N2O 派生变量兼容的常用模板。
"""

CH4_FEATURE_COLUMNS = [
    "Watertempreature",
    "pH",
    "DO",
    "MAT_Final",
    "MAP_Final",
    "Monthly_Precipitation_ERA5",
    "Windspeed_Filled",
    "Longitude"
]

# 如果你的 CH4 模型变量和 CO2 更接近，可以改成类似：
# CH4_FEATURE_COLUMNS = [
#     "DO",
#     "pH",
#     "chla",
#     "Watertempreature",
#     "Airtemperature_Filled",
#     "Monthly_Precipitation_ERA5",
#     "Windspeed_Filled",
#     "ERA5_Radiation",
#     "Atomsphericpressure_Filled",
#     "ERA5_SoilTemp"
# ]


# ============================================================
# 5. 科学计算函数
# ============================================================

def calculate_pressure_hPa(elevation_m):
    """
    海拔 m -> 大气压 hPa
    该公式与你 CO2 脚本保持一致。
    """
    return 101.325 * np.power((1 - 2.25577e-5 * elevation_m), 5.25588) * 10.0


def calculate_solar_radiation_FAO56(lat_array, day_of_year):
    """
    基于 FAO56 公式估算太阳辐射。
    输出单位与 CO2 脚本保持一致。
    """
    doy = np.array(day_of_year)
    phi = np.deg2rad(lat_array)

    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    tan_phi_delta = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)

    ws = np.arccos(tan_phi_delta)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)

    ra = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (
            ws * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(ws)
        )
    )

    return np.maximum(0, ra * 0.6 * 11.574)


def calculate_do_graham2025(temp_k, elevation_m):
    """
    根据温度和海拔估算溶解氧 DO。
    与 CO2 和 N2O 脚本保持一致。
    """
    tw = np.maximum(temp_k, 273.15)

    c0 = -139.34411
    c1 = 1.575701e5
    c2 = -6.642308e7
    c3 = 1.243800e10
    c4 = -8.621949e11
    c_elev = 0.0001148

    exponent = (
        c0
        + (c1 / tw)
        + (c2 / tw ** 2)
        + (c3 / tw ** 3)
        + (c4 / tw ** 4)
    )

    do_sat = np.exp(exponent)
    do_val = do_sat * (1 - (c_elev * elevation_m))

    return np.maximum(do_val, 0)


def calculate_ph_cai2011(do_val):
    """
    根据 DO 估算 pH。
    与 CO2 和 N2O 脚本保持一致。
    """
    return np.full_like(do_val, PH_MEDIAN_VALUE, dtype=float)


def inverse_log_modulus(x):
    """
    log-modulus 反变换：
    y = sign(x) * (10^abs(x) - 1)
    与 CO2 和 N2O 脚本保持一致。
    """
    return np.sign(x) * (np.power(10, np.abs(x)) - 1)


def get_lat_lon_names(ds):
    """
    自动识别 ERA5 或其他 NetCDF 中的经纬度坐标名称。
    """
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return "latitude", "longitude"
    if "lat" in ds.coords and "lon" in ds.coords:
        return "lat", "lon"

    raise ValueError("无法识别经纬度坐标名称。请检查数据中是否包含 latitude/longitude 或 lat/lon。")


def get_model_feature_names(model):
    """
    尝试从 pkl 模型中自动读取训练变量名。
    如果读取失败，则返回 None。
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    if hasattr(model, "booster_"):
        try:
            return list(model.booster_.feature_name())
        except Exception:
            pass

    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            names = get_model_feature_names(step)
            if names is not None:
                return names

    return None


def prepare_era5_dataset(era5_file_path):
    """
    读取并裁剪 ERA5 数据。
    """
    print(f">>> 正在解析 ERA5 数据: {os.path.basename(era5_file_path)}")

    ds_era5 = xr.open_dataset(era5_file_path, decode_times=False).squeeze()

    if "valid_time" in ds_era5.dims:
        ds_era5 = ds_era5.rename({"valid_time": "time"})
    elif "valid_time" in ds_era5.coords:
        ds_era5 = ds_era5.rename({"valid_time": "time"})

    lat_name, lon_name = get_lat_lon_names(ds_era5)

    # 中国范围裁剪：纬度方向可能从北到南，也可能从南到北
    lat_values = ds_era5[lat_name].values

    if lat_values[0] > lat_values[-1]:
        ds_era5 = ds_era5.sel({lat_name: slice(55, 15), lon_name: slice(70, 140)})
    else:
        ds_era5 = ds_era5.sel({lat_name: slice(15, 55), lon_name: slice(70, 140)})

    print(f">>> 正在提取 2011–2024 年全量数据，共 {TOTAL_MONTHS} 个月...")
    ds_era5_target = ds_era5.isel(time=slice(0, TOTAL_MONTHS))

    return ds_era5_target, lat_name, lon_name


def prepare_elevation(elevation_file, target_lat, target_lon):
    """
    读取高程数据并插值到 ERA5 网格。
    """
    print(">>> 正在处理高程数据...")

    ds_elev = xr.open_dataset(elevation_file, decode_times=False).squeeze()

    if ds_elev.lon.max() > 180:
        ds_elev.coords["lon"] = (ds_elev.coords["lon"] + 180) % 360 - 180

    ds_elev_china = ds_elev.sortby(["lat", "lon"]).sel(
        lat=slice(15, 55),
        lon=slice(70, 140)
    )

    elev_var_name = list(ds_elev_china.data_vars)[0]

    elev_regrid = ds_elev_china[elev_var_name].interp(
        lat=target_lat,
        lon=target_lon,
        method="linear"
    ).values

    if elev_regrid.ndim > 2:
        elev_regrid = np.squeeze(elev_regrid)

    return elev_regrid


def build_ch4_input_dataframe(
    t2m_k,
    tp_m,
    u10,
    v10,
    elev_regrid,
    lat_grid,
    lon_grid,
    mask,
    day_of_year
):
    """
    构建 CH4 模型候选输入变量表。
    后续会按照 CH4_FEATURE_COLUMNS 或模型自带变量名筛选。
    """

    air_c = t2m_k[mask] - 273.15

    # 注意：
    # 这里沿用你 CO2 和 N2O 脚本中的降水处理：
    # tp_m * 1000 * 30
    # 如果你的 ERA5 tp 已经是月累计降水，则这里不应再乘 30。
    monthly_precip_mm = tp_m[mask] * 1000 * 30

    wind_val = np.sqrt(u10[mask] ** 2 + v10[mask] ** 2)

    elev_m = elev_regrid[mask]

    water_c = 0.4180 * air_c + 12.9454

    do_val = calculate_do_graham2025(t2m_k[mask], elev_m)

    ph_val = calculate_ph_cai2011(do_val)

    chla_val = np.full_like(air_c, CHLA_STATIC_VALUE, dtype=float)

    pressure_hPa = calculate_pressure_hPa(elev_m)

    rad_val = calculate_solar_radiation_FAO56(lat_grid[mask], day_of_year)

    df = pd.DataFrame({
        # 与 N2O 模型兼容的变量
        "Watertempreature": water_c,
        "pH": ph_val,
        "DO": do_val,
        "MAT_Final": air_c,
        "MAP_Final": monthly_precip_mm * 12,
        "Monthly_Precipitation_ERA5": monthly_precip_mm,
        "Windspeed_Filled": wind_val,
        "Longitude": lon_grid[mask],
        "Latitude": lat_grid[mask],

        # 与 CO2 模型兼容的变量
        "chla": chla_val,
        "Airtemperature_Filled": air_c,
        "ERA5_Radiation": rad_val,
        "Atomsphericpressure_Filled": pressure_hPa,
        "ERA5_SoilTemp": air_c,

        # 其他可能有用的变量
        "Elevation": elev_m
    })

    # 添加用户自定义静态变量
    for key, value in EXTRA_STATIC_FEATURES.items():
        df[key] = value

    return df


def predict_real_flux(model, x_input):
    """
    模型预测并转换为真实通量。
    """
    y_pred = model.predict(x_input)

    if APPLY_INVERSE_LOG_MODULUS:
        y_real = inverse_log_modulus(y_pred) * SMEARING_FACTOR
    else:
        y_real = y_pred * SMEARING_FACTOR

    if CLIP_NEGATIVE_TO_ZERO:
        y_real = np.maximum(y_real, 0)

    return y_real


# ============================================================
# 6. 主执行函数
# ============================================================

def run_era5_ch4_baseline_all_years():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(">>> 正在加载 CH4 机器学习模型...")
    ch4_model = joblib.load(MODEL_PATH)

    model_features = get_model_feature_names(ch4_model)

    if model_features is not None:
        feature_columns = model_features
        print(">>> 已从 pkl 模型中自动读取特征名:")
    else:
        feature_columns = CH4_FEATURE_COLUMNS
        print(">>> 未能从 pkl 模型自动读取特征名，使用脚本中 CH4_FEATURE_COLUMNS:")

    for col in feature_columns:
        print(f"    - {col}")

    ds_era5_target, lat_name, lon_name = prepare_era5_dataset(ERA5_FILE_PATH)

    target_lat = ds_era5_target[lat_name]
    target_lon = ds_era5_target[lon_name]

    elev_regrid = prepare_elevation(
        ELEVATION_FILE,
        target_lat=target_lat,
        target_lon=target_lon
    )

    lon_grid, lat_grid = np.meshgrid(
        ds_era5_target[lon_name].values,
        ds_era5_target[lat_name].values
    )

    clean_time_axis = pd.date_range(
        start=START_TIME,
        periods=TOTAL_MONTHS,
        freq="MS"
    )

    doy_series = clean_time_axis.dayofyear

    pred_list = []

    print(f">>> 开始计算 CH4 全时段逐月通量，共 {TOTAL_MONTHS} 个月...")

    for i in tqdm(range(TOTAL_MONTHS)):
        t2m_k = ds_era5_target["t2m"].isel(time=i).values
        tp_m = ds_era5_target["tp"].isel(time=i).values
        u10 = ds_era5_target["u10"].isel(time=i).values
        v10 = ds_era5_target["v10"].isel(time=i).values

        mask = ~np.isnan(t2m_k)

        if not mask.any():
            pred_list.append(np.full_like(t2m_k, np.nan, dtype=np.float32))
            continue

        candidate_df = build_ch4_input_dataframe(
            t2m_k=t2m_k,
            tp_m=tp_m,
            u10=u10,
            v10=v10,
            elev_regrid=elev_regrid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            mask=mask,
            day_of_year=doy_series[i]
        )

        missing_features = [f for f in feature_columns if f not in candidate_df.columns]

        if missing_features:
            available_cols = list(candidate_df.columns)
            raise ValueError(
                "\nCH4 模型所需变量缺失。\n"
                f"缺失变量: {missing_features}\n\n"
                f"当前脚本可提供变量: {available_cols}\n\n"
                "解决方法：\n"
                "1. 检查 CH4_FEATURE_COLUMNS 是否和训练模型变量名一致；\n"
                "2. 如果模型需要 COD、TN、TP、TOC 等静态变量，请在 EXTRA_STATIC_FEATURES 中添加；\n"
                "3. 如果变量名只是拼写不同，请在 build_ch4_input_dataframe 中增加对应别名。\n"
            )

        x_input = candidate_df[feature_columns]

        y_pred_real = predict_real_flux(ch4_model, x_input)

        res_frame = np.full_like(t2m_k, np.nan, dtype=np.float32)
        res_frame[mask] = y_pred_real.astype(np.float32)

        pred_list.append(res_frame)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NC_NAME)

    da_res = xr.DataArray(
        np.array(pred_list, dtype=np.float32),
        dims=["time", "lat", "lon"],
        coords={
            "time": clean_time_axis,
            "lat": ds_era5_target[lat_name].values,
            "lon": ds_era5_target[lon_name].values
        },
        name="ch4_flux",
        attrs={
            "units": OUTPUT_UNITS,
            "description": "Baseline CH4 flux from ERA5 2011-2024",
            "model_path": MODEL_PATH,
            "apply_inverse_log_modulus": str(APPLY_INVERSE_LOG_MODULUS),
            "smearing_factor": SMEARING_FACTOR,
            "clip_negative_to_zero": str(CLIP_NEGATIVE_TO_ZERO),
            "chla_static_value": CHLA_STATIC_VALUE,
            "feature_columns": ",".join(feature_columns)
        }
    )

    da_res.to_netcdf(output_path)

    print("\n🎉 CH4 基准网格已生成！")
    print(f">>> 输出路径: {output_path}")
    print(f">>> 输出变量: ch4_flux")
    print(f">>> 输出单位: {OUTPUT_UNITS}")


# ============================================================
# 7. 程序入口
# ============================================================

if __name__ == "__main__":
    run_era5_ch4_baseline_all_years()
