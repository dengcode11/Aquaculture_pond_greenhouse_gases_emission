# -*- coding: utf-8 -*-
"""
Step 1 (Final + CF): CH4 随机森林冠军模型训练 + 涂抹因子计算
功能:
  1. 使用贝叶斯优化参数训练 RandomForest
  2. 自动计算段氏涂抹因子 (Duan's Smearing Factor)
  3. 输出模型文件 (.pkl) 和 因子文件 (.txt)
"""

from aquaculture_paths import project_path, external_path

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

# ================= 配置区域 =================
# 输入文件
DATA_PATH = "CH4_Environment_Final_Imputed_MissForest.csv"
# 输出模型名称
MODEL_OUTPUT_NAME = "RF_China_CH4_Best_Param_v2.pkl"
# 输出因子记录文件
CF_OUTPUT_NAME = "CH4_Smearing_Factor.txt"

# 核心特征 (10个)
SPECIFIED_FEATURES = [
    "Longitude",  # 1. 经度
    "Watertempreature",  # 2. 水温
    "DO",  # 3. DO
    "MAT_Final",  # 4. MAT (用气温代替)
    "Latitude",  # 5. 纬度
    "Atomsphericpressure_Filled",  # 6. 气压 (hPa)
    "ERA5_SoilTemp",  # 7. 土温
    "Airtemperature_Filled",  # 8. 气温
    "pH",  # 9. pH
    "MAP_Final"  # 10. MAP
]


def log_modulus_transform(x):
    """目标变量对数变换"""
    return np.sign(x) * np.log10(np.abs(x) + 1)


def train_and_save():
    print(">>> [Step 1] 读取并准备 CH4 数据...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # 1. 气压检查 (确认是 hPa)
    if "Atomsphericpressure_Filled" in df.columns:
        mean_pres = df["Atomsphericpressure_Filled"].mean()
        print(f"   ℹ️  气压均值: {mean_pres:.2f} (确认单位为 hPa)")
        if mean_pres < 200:
            print("   ⚠️ 警告: 训练数据气压似乎是 kPa，请检查源数据！")

    # 2. 目标变量处理
    if 'CH4_flux' not in df.columns:
        print("❌ 缺少 CH4_flux 列");
        return

    # 确保数值型
    df['CH4_flux'] = pd.to_numeric(df['CH4_flux'], errors='coerce')
    df = df.dropna(subset=['CH4_flux'])
    df['CH4_flux_trans'] = log_modulus_transform(df['CH4_flux'])

    # 3. 特征筛选
    missing = [f for f in SPECIFIED_FEATURES if f not in df.columns]
    if missing:
        print(f"❌ 缺少特征: {missing}");
        return

    work_df = df[SPECIFIED_FEATURES + ['CH4_flux_trans']].dropna()
    print(f"   - 训练样本量: {len(work_df)}")

    X = work_df[SPECIFIED_FEATURES]
    y = work_df['CH4_flux_trans']

    # ==========================================
    # 3. 模型训练 (贝叶斯最优参数)
    # ==========================================
    print("\n>>> [Step 2] 开始训练 Random Forest (n_estimators=535, max_depth=16)...")

    rf = RandomForestRegressor(
        n_estimators=535,
        max_depth=16,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
        oob_score=True  # 🔥 开启 OOB 计算
    )

    rf.fit(X, y)
    print(f"   ✅ 训练完成! OOB R² Score: {rf.oob_score_:.4f}")

    # ==========================================
    # 🔥 计算段氏涂抹因子 (Smearing Factor)
    # ==========================================
    print("\n>>> [Step 2.5] 计算校正因子 (Duan's Smearing Factor)...")

    # 使用 OOB 预测值计算残差
    y_pred_log = rf.oob_prediction_

    # 1. 计算残差
    residuals = y - y_pred_log

    # 2. 计算因子 CF = mean(10^residuals)
    smearing_factor = np.mean(10 ** residuals)

    print(f"   📊 CH4 涂抹因子 (CF): {smearing_factor:.6f}")

    # 保存因子
    with open(CF_OUTPUT_NAME, "w") as f:
        f.write(str(smearing_factor))
    print(f"   💾 因子已保存至: {CF_OUTPUT_NAME}")

    # ==========================================

    # 4. 保存模型
    print(f"\n>>> [Step 3] 保存模型至: {MODEL_OUTPUT_NAME}")
    joblib.dump(rf, MODEL_OUTPUT_NAME)
    print("🎉 全部完成！")
    print(f"👉 下一步预测时，请在反变换公式中乘以: {smearing_factor:.6f}")
    print("👉 并在 Step 2 中确保气压单位转换为 hPa (kPa * 10)！")


if __name__ == "__main__":
    train_and_save()