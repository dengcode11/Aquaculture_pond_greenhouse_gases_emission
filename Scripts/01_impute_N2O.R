source(Sys.getenv("AQUACULTURE_R_RUNTIME"), encoding="UTF-8")
# ==============================================================================
# 0. 环境准备与包加载
# ==============================================================================
# 加载必要的包 (新增 caret 和 gbm)
required_pkgs <- c("tidyverse", "caret", "gbm")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs) > 0) require_installed_packages(new_pkgs)

library(tidyverse)
library(caret)
library(gbm)

# ==============================================================================
# 1. 数据加载与预处理
# ==============================================================================
message(">>> [Step 1] 数据加载中...")
# setwd("Configured through code/run.py
df <- read_csv("环境因子预测_Enriched_Final_v4_Corrected.csv", show_col_types = FALSE)

# 1.1 基础数据清洗 & 筛选 N2O 子集
# 我们只针对 N2O 数据集进行插值，以保证物理规律的纯粹性
df_n2o <- df %>%
  mutate(Location_type = if_else(Row_ID == 2081, "CPS", Location_type)) %>%
  mutate(Season_Clean = tools::toTitleCase(tolower(Season_Clean))) %>%
  filter(Season_Clean != "Annual") %>%
  filter(!is.na(N2O_flux)) %>%   # <--- 锁定 N2O 样本
  filter(!is.na(Month_Clean))

# 1.2 【关键】异常值清洗 (根据之前的分析结果)
message(">>> 执行异常值清洗 (NH4N>10, NO3N>30)...")

# 统一列名 Chla
if("Chla" %in% names(df_n2o)) df_n2o <- df_n2o %>% rename(chla = Chla)
if("Chl伪" %in% names(df_n2o)) df_n2o <- df_n2o %>% rename(chla = `Chl伪`)

# 清洗氨氮 (NH4N)
if("NH4N" %in% names(df_n2o)) {
  outliers_nh4 <- which(df_n2o$NH4N > 10)
  if(length(outliers_nh4) > 0) {
    df_n2o$NH4N[outliers_nh4] <- NA
    message(paste("    已剔除", length(outliers_nh4), "个 NH4N > 10 的异常值"))
  }
}

# 清洗硝氮 (NO3N)
if("NO3N" %in% names(df_n2o)) {
  outliers_no3 <- which(df_n2o$NO3N > 30)
  if(length(outliers_no3) > 0) {
    df_n2o$NO3N[outliers_no3] <- NA
    message(paste("    已剔除", length(outliers_no3), "个 NO3N > 30 的异常值"))
  }
}

# 清洗叶绿素 (Chla)
if("chla" %in% names(df_n2o)) {
  df_n2o$chla[df_n2o$chla > 2000] <- NA
}

# 1.3 定义变量
# 辅助变量 (Predictors)
aux_vars <- c("ERA5_Temp", "ERA5_SoilTemp", "ERA5_Radiation", "ERA5_WindSpeed", "ERA5_Precip",
              "MAT_Final", "MAP_Final", 
              "Latitude", "Longitude", "Month_Clean", 
              "System_type", "Climate_Zone", "Location_type")

# 目标变量 (Targets to impute)
# N2O 组通常关注这 5 个
env_targets <- c("Watertempreature", "pH", "DO", "NH4N", "NO3N")

# 1.4 准备 BRT 输入数据
# 提取相关列
data_full <- df_n2o %>%
  select(all_of(c(aux_vars, env_targets))) %>%
  mutate(across(where(is.character), as.factor)) %>%
  as.data.frame()

# 【重要】BRT 预处理：
# BRT 允许目标变量(Y)有缺失，但作为特征(X)输入时最好是完整的。
# 我们先用中位数快速填充辅助变量的微量缺失，以防报错。
data_brt_input <- data_full %>% 
  mutate(across(all_of(aux_vars), ~ifelse(is.na(.), median(., na.rm=TRUE), .)))

message(paste("待处理 N2O 数据量:", nrow(data_full), "行"))
message("核心因子缺失概况:")
print(colSums(is.na(data_full[env_targets])))

# ==============================================================================
# 2. 运行 BRT (Gradient Boosting) 插值
# ==============================================================================
message("\n>>> [Step 2] 正在运行 BRT 插值 (循环处理 5 个因子)...")

# 复制一份结果容器
df_imputed <- df_n2o

# BRT 参数网格 (已优化)
brt_grid <- expand.grid(interaction.depth = 5, 
                        n.trees = 800, 
                        shrinkage = 0.05, 
                        n.minobsinnode = 10)

# 训练控制 (method="none" 表示不进行交叉验证，直接用全数据训练以节省时间)
fit_control <- trainControl(method = "none")

# --- 循环插值核心逻辑 ---
for(var in env_targets) {
  message(paste("  -> 正在拟合:", var, "..."))
  
  # 1. 拆分训练集 (Y 已知) 和 预测集 (Y 缺失)
  train_data <- data_brt_input[!is.na(data_brt_input[[var]]), ]
  pred_data  <- data_brt_input[is.na(data_brt_input[[var]]), ]
  
  # 2. 如果存在缺失值，则训练并预测
  if(nrow(pred_data) > 0) {
    set.seed(2025)
    
    # 构建公式: Target ~ Aux1 + Aux2 + ...
    form <- as.formula(paste(var, "~", paste(aux_vars, collapse = "+")))
    
    # 训练模型
    # verbose=FALSE 静默运行
    model <- train(form, data = train_data, method = "gbm", 
                   trControl = fit_control, tuneGrid = brt_grid, verbose = FALSE)
    
    # 预测并填补
    preds <- predict(model, newdata = pred_data)
    
    # 将预测值填回结果数据框
    # 注意：这里我们修改的是 df_imputed (保留了原始所有列)
    df_imputed[[var]][is.na(df_imputed[[var]])] <- preds
    
    message(paste("     已填补", length(preds), "个缺失值。"))
  } else {
    message("     无缺失值，跳过。")
  }
}

# ==============================================================================
# 3. 后处理：物理范围约束 (Sanity Check)
# ==============================================================================
message("\n>>> [Step 3] 执行物理范围约束 (Post-processing)...")

# 1. 溶解氧 (DO)
# 范围：0 - 30 (去除极其不可能的负值和超高值)
df_imputed$DO <- pmax(df_imputed$DO, 0)
df_imputed$DO <- pmin(df_imputed$DO, 30)

# 2. pH 值 
# 范围：4 - 11 (一般水产养殖极限)
df_imputed$pH <- pmax(pmin(df_imputed$pH, 11), 4) 

# 3. 营养盐 (必须非负)
if("NH4N" %in% names(df_imputed)) df_imputed$NH4N <- pmax(df_imputed$NH4N, 0)
if("NO3N" %in% names(df_imputed)) df_imputed$NO3N <- pmax(df_imputed$NO3N, 0)

# 4. 水温
df_imputed$Watertempreature <- pmax(pmin(df_imputed$Watertempreature, 45), -5)

# ==============================================================================
# 4. 结果导出
# ==============================================================================
message("\n>>> [Step 4] 导出数据...")

output_filename <- "N2O_Subset_Imputed_BRT.csv"
write_csv(df_imputed, output_filename)

message(paste("BRT 插值完成！文件已保存为:", output_filename))
message("提示：此文件仅包含 N2O 及其相关的环境因子数据。")

