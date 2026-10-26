# -*- coding: utf-8 -*-
"""
Step 1 (Final + CF): 训练模型并计算涂抹因子
更新内容：
  1. 训练随机森林 (贝叶斯最优参数)
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
DATA_PATH = "N2O_Subset_Imputed_BRT.csv"
# 输出模型名称
MODEL_OUTPUT_NAME = "RF_China_N2O_BayesTuned.pkl"
# 输出因子记录文件
CF_OUTPUT_NAME = "Smearing_Factor.txt"

# 核心特征 (10个)
SELECTED_FEATURES = [
    "Watertempreature", "pH", "DO", "NH4N", "NO3N",
    "MAT_Final", "MAP_Final", "Monthly_Precipitation_ERA5",
    "Windspeed_Filled", "Longitude"
]


def log_modulus_transform(x):
    """目标变量对数变换"""
    return np.sign(x) * np.log10(np.abs(x) + 1)


def train_and_save():
    print(">>> [Step 1] 读取并准备 N2O 数据...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # 1. 目标变量处理
    target_col = 'N2O_flux_trans'
    if target_col not in df.columns:
        df = df.dropna(subset=['N2O_flux'])
        df[target_col] = log_modulus_transform(df['N2O_flux'])

    # 2. 特征筛选
    work_df = df[SELECTED_FEATURES + [target_col]].dropna()
    print(f"   - 训练样本量: {len(work_df)}")

    X = work_df[SELECTED_FEATURES]
    y = work_df[target_col]

    # 3. 模型训练 (贝叶斯最优参数)
    print(">>> [Step 2] 开始训练 (n_estimators=527, max_depth=22)...")

    rf = RandomForestRegressor(
        n_estimators=527,
        max_depth=22,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
        oob_score=True  # 🔥 开启 OOB 以便计算稳健的残差
    )

    rf.fit(X, y)
    print(f"   ✅ 训练完成! OOB R² Score: {rf.oob_score_:.4f}")

    # ==========================================
    # 🔥 新增: 计算段氏涂抹因子 (Smearing Factor)
    # ==========================================
    print("\n>>> [Step 2.5] 计算校正因子 (Duan's Smearing Factor)...")

    # 使用 OOB 预测值计算残差 (比直接用 predict(X) 更严谨，避免过拟合导致的因子偏小)
    # 如果 OOB 效果不好，也可以改用 rf.predict(X)
    y_pred_log = rf.oob_prediction_

    # 1. 计算 Log 空间残差
    residuals = y - y_pred_log

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
    print(f"\n>>> [Step 3] 保存模型至: {MODEL_OUTPUT_NAME}")
    joblib.dump(rf, MODEL_OUTPUT_NAME)
    print("🎉 全部完成！")
    print(f"👉 下一步预测时，请在反变换公式中乘以: {smearing_factor:.6f}")


if __name__ == "__main__":
    train_and_save()