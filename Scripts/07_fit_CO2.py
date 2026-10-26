# -*- coding: utf-8 -*-
"""
Step 1 (Final + CF): CO2 LightGBM 冠军模型训练 + 涂抹因子计算
功能:
  1. 使用贝叶斯优化参数训练 LightGBM
  2. 使用 5-Fold CV 获取袋外模拟预测值 (因 LGBM 无 OOB)
  3. 计算段氏涂抹因子 (Duan's Smearing Factor) 以修正 Log 偏差
"""

from aquaculture_paths import project_path, external_path

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
from sklearn.model_selection import cross_val_predict, KFold

# ================= 配置区域 =================
DATA_PATH = "CO2_Subset_Imputed_BRT.csv"
MODEL_OUTPUT_NAME = "LGBM_China_CO2_Best_Param_v2.pkl"
CF_OUTPUT_NAME = "CO2_residual_multiplier_diagnostic_only.txt"

# 🔥 Top 10 核心特征 (训练与预测必须严格保持此顺序)
SPECIFIED_FEATURES = [
    "DO",  # 1
    "pH",  # 2
    "chla",  # 3
    "Watertempreature",  # 4
    "Airtemperature_Filled",  # 5
    "Monthly_Precipitation_ERA5",  # 6
    "Windspeed_Filled",  # 7
    "ERA5_Radiation",  # 8
    "Atomsphericpressure_Filled",  # 9
    "ERA5_SoilTemp"  # 10
]


def log_modulus_transform(x):
    """目标变量对数变换"""
    return np.sign(x) * np.log10(np.abs(x) + 1)


def train_and_save():
    print(">>> [Step 1] 读取并准备 CO2 数据...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # 统一列名
    if 'Chl伪' in df.columns:
        df.rename(columns={'Chl伪': 'chla'}, inplace=True)
    elif 'Chla' in df.columns:
        df.rename(columns={'Chla': 'chla'}, inplace=True)

    # 检查特征
    missing = [f for f in SPECIFIED_FEATURES if f not in df.columns]
    if missing:
        print(f"❌ 缺少特征: {missing}")
        return

    # 目标变量处理
    target_col = 'CO2_flux_trans'
    if target_col not in df.columns:
        df = df.dropna(subset=['CO2_flux'])
        df[target_col] = log_modulus_transform(df['CO2_flux'])

    # 提取训练数据
    work_df = df[SPECIFIED_FEATURES + [target_col]].dropna()
    print(f"   - 训练样本量: {len(work_df)}")

    X = work_df[SPECIFIED_FEATURES]
    y = work_df[target_col]

    # ==========================================
    # 2. 模型训练 (使用新参数)
    # ==========================================
    print("\n>>> [Step 2] 训练 LightGBM 模型 (Best Params)...")

    # 最佳参数配置
    params = {
        'objective': 'regression',
        'n_estimators': 800,
        'learning_rate': 0.0864,
        'num_leaves': 31,
        'min_child_samples': 21,
        'max_depth': 7,
        'subsample': 0.723,
        'subsample_freq': 1,
        'reg_lambda': 5.49,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }

    # A. 全量训练并保存
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    print(f"   ✅ 训练完成! 全量数据 R²: {model.score(X, y):.4f}")

    # ==========================================
    # 3. 计算涂抹因子 (Smearing Factor)
    # ==========================================
    print("\n>>> [Step 2.5] 计算校正因子 (Duan's Smearing Factor)...")

    # 由于 LightGBM 没有 OOB，我们使用 5-Fold CV 获取“袋外”预测值
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # 注意: 这里使用同样的参数进行 CV 预测
    cv_model = lgb.LGBMRegressor(**params)
    y_pred_cv = cross_val_predict(cv_model, X, y, cv=kf, n_jobs=-1)

    # 1. 计算 Log 空间残差
    residuals = y - y_pred_cv

    # 2. 计算涂抹因子 CF = mean(10^residuals)
    # 这里的 10 是因为我们变换用的是 log10
    smearing_factor = np.mean(10 ** residuals)

    print(f"   📊 涂抹因子 (CF): {smearing_factor:.6f}")

    # 保存因子到文本文件
    with open(CF_OUTPUT_NAME, "w") as f:
        f.write(str(smearing_factor))
    print(f"   💾 因子已保存至: {CF_OUTPUT_NAME}")

    # ==========================================
    # 4. 保存模型
    # ==========================================
    print(f"\n>>> [Step 3] 保存模型至: {MODEL_OUTPUT_NAME}")
    joblib.dump(model, MODEL_OUTPUT_NAME)
    print("🎉 全部完成！")
    print("The residual multiplier is diagnostic only. The supplied CO2 prediction workflow uses its separately specified physical-scale multiplier of 1.57.")


if __name__ == "__main__":
    train_and_save()