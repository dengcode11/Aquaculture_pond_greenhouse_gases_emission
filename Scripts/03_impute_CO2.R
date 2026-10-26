source(Sys.getenv("AQUACULTURE_R_RUNTIME"), encoding="UTF-8")
# ==============================================================================
# 0. 环境准备
# ==============================================================================
library(tidyverse)
library(caret)
library(gbm)

# ==============================================================================
# 1. 数据读取与预处理
# ==============================================================================
message(">>> [Step 1] 数据加载中...")
# setwd("Configured through code/run.py
df <- read_csv("环境因子预测_Enriched_Final_v4_Corrected.csv", show_col_types = FALSE)

# 1.1 筛选 CO2 子集
# 锁定 CO2_flux 不为空的数据行
df_co2 <- df %>%
  mutate(Location_type = if_else(Row_ID == 2081, "CPS", Location_type)) %>%
  mutate(Season_Clean = tools::toTitleCase(tolower(Season_Clean))) %>%
  filter(Season_Clean != "Annual") %>%
  filter(!is.na(CO2_flux)) %>%   # <--- 关键筛选
  filter(!is.na(Month_Clean))

# 1.2 异常值清洗 (Critical Step)
message(">>> 执行异常值清洗...")

# 统一列名 (防止 Chla / Chl伪 混淆)
if("Chla" %in% names(df_co2)) df_co2 <- df_co2 %>% rename(chla = Chla)
if("Chl伪" %in% names(df_co2)) df_co2 <- df_co2 %>% rename(chla = `Chl伪`)

# 清洗叶绿素 (chla): 剔除 > 2000 的极端异常值 (针对之前发现的 75210 异常值)
if("chla" %in% names(df_co2)) {
  outliers_chla <- which(df_co2$chla > 2000)
  if(length(outliers_chla) > 0) {
    df_co2$chla[outliers_chla] <- NA
    message(paste("    已剔除", length(outliers_chla), "个 chla > 2000 的异常值"))
  }
}

# 1.3 定义变量
# 辅助变量 (Predictors)
aux_vars <- c("ERA5_Temp", "ERA5_SoilTemp", "ERA5_Radiation", "ERA5_WindSpeed", "ERA5_Precip",
              "MAT_Final", "MAP_Final", 
              "Latitude", "Longitude", "Month_Clean", 
              "System_type", "Climate_Zone", "Location_type")

# CO2 组关注的核心环境因子 (Targets)
env_targets <- c("Watertempreature", "pH", "DO", "chla")

# 1.4 准备 BRT 输入数据
data_full <- df_co2 %>%
  select(all_of(c(aux_vars, env_targets))) %>%
  mutate(across(where(is.character), as.factor)) %>%
  as.data.frame()

# 预填补辅助变量 (防止 BRT 运行时报错)
# 对辅助变量的微量缺失使用中位数填充
data_brt_input <- data_full %>% 
  mutate(across(all_of(aux_vars), ~ifelse(is.na(.), median(., na.rm=TRUE), .)))

message(paste("待处理 CO2 数据量:", nrow(data_full), "行"))
message("核心因子缺失概况:")
print(colSums(is.na(data_full[env_targets])))

# ==============================================================================
# 2. 运行 BRT (Gradient Boosting) 插值
# ==============================================================================
message("\n>>> [Step 2] 正在运行 BRT 插值 (循环处理 4 个因子)...")

df_imputed <- df_co2 # 复制副本用于存储结果

# BRT 参数网格
brt_grid <- expand.grid(interaction.depth = 5, 
                        n.trees = 800, 
                        shrinkage = 0.05, 
                        n.minobsinnode = 10)

# 训练控制 (全量训练，不交叉验证以节省时间)
fit_control <- trainControl(method = "none")

# --- 循环插值核心逻辑 ---
for(var in env_targets) {
  message(paste("  -> 正在拟合:", var, "..."))
  
  # 1. 拆分训练集 (Y 已知) 和 预测集 (Y 缺失)
  train_data <- data_brt_input[!is.na(data_brt_input[[var]]), ]
  pred_data  <- data_brt_input[is.na(data_brt_input[[var]]), ]
  
  # 2. 如果存在缺失值，则训练并预测
  if(nrow(pred_data) > 0) {
    set.seed(2025) # 保证结果可复现
    
    # 构建公式: Target ~ Aux1 + Aux2...
    form <- as.formula(paste(var, "~", paste(aux_vars, collapse = "+")))
    
    # 训练模型
    model <- train(form, data = train_data, method = "gbm", 
                   trControl = fit_control, tuneGrid = brt_grid, verbose = FALSE)
    
    # 预测
    preds <- predict(model, newdata = pred_data)
    
    # 填回
    df_imputed[[var]][is.na(df_imputed[[var]])] <- preds
    
    message(paste("     已填补", length(preds), "个缺失值。"))
  } else {
    message("     无缺失值，跳过。")
  }
}

# ==============================================================================
# 3. 后处理：物理范围约束 (Sanity Check)
# ==============================================================================
message("\n>>> [Step 3] 执行物理范围约束...")

# 水温 (-5 ~ 45)
df_imputed$Watertempreature <- pmax(pmin(df_imputed$Watertempreature, 45), -5)

# pH (4 ~ 11)
df_imputed$pH <- pmax(pmin(df_imputed$pH, 11), 4)

# DO (0 ~ 30)
df_imputed$DO <- pmax(pmin(df_imputed$DO, 30), 0)

# Chla (>= 0)
if("chla" %in% names(df_imputed)) df_imputed$chla <- pmax(df_imputed$chla, 0)

# ==============================================================================
# 4. 导出结果
# ==============================================================================
message("\n>>> [Step 4] 导出数据...")

output_filename <- "CO2_Subset_Imputed_BRT.csv"
write_csv(df_imputed, output_filename)

message(paste("成功！CO2 数据集 BRT 插值完成，文件已保存为:", output_filename))

