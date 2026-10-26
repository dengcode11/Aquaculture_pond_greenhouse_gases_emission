source(Sys.getenv("AQUACULTURE_R_RUNTIME"), encoding="UTF-8")
# ==============================================================================
# 0. 环境准备
# ==============================================================================
# 检查并安装必要的包
required_pkgs <- c("tidyverse", "missForest", "ggplot2", "patchwork")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs) > 0) require_installed_packages(new_pkgs)

library(tidyverse)
library(missForest)
library(ggplot2)
library(patchwork)

# --- 【关键】设置绝对路径变量 ---
# 这样修改一次，后面读写都会自动用这个路径
work_dir <- project_path("literature")

# 尝试设置工作目录 (如果文件夹不存在会报错提示)
if(dir.exists(work_dir)) {
  setwd(work_dir)
  message(paste("成功设置工作目录为:", work_dir))
} else {
  stop(paste("错误：找不到文件夹", work_dir, "，请检查路径拼写是否正确！"))
}

# ==============================================================================
# 1. 数据读取与预处理
# ==============================================================================
message(">>> [Step 1] 正在读取并清洗数据...")

# 读取文件 (自动拼接路径)
input_file <- file.path(work_dir, "环境因子预测_Enriched_Final_v4_Corrected.csv")
df <- read_csv(input_file, show_col_types = FALSE)

# 1.1 统一列名
if("Chla" %in% names(df)) df <- df %>% rename(chla = Chla)
if("Chl伪" %in% names(df)) df <- df %>% rename(chla = `Chl伪`)

# 1.2 锁定 CH4 子集
df_ch4 <- df %>%
  filter(!is.na(CH4_flux)) %>%         
  filter(Season_Clean != "Annual") %>% 
  filter(!is.na(Month_Clean))          

message(paste("原始 CH4 子集样本量:", nrow(df_ch4)))

# 1.3 异常值清洗
# DO > 30 (异常)
invalid_do <- which(df_ch4$DO > 30)
if(length(invalid_do) > 0) {
  message(paste("清洗: 发现", length(invalid_do), "个 DO 异常值 (>30)，已重置为 NA。"))
  df_ch4$DO[invalid_do] <- NA
}

# pH > 11 (异常)
invalid_ph <- which(df_ch4$pH > 11)
if(length(invalid_ph) > 0) {
  message(paste("清洗: 发现", length(invalid_ph), "个 pH 异常值 (>11)，已重置为 NA。"))
  df_ch4$pH[invalid_ph] <- NA
}

# ==============================================================================
# 2. 执行 MissForest 插值
# ==============================================================================
message("\n>>> [Step 2] 正在运行 MissForest 插值 (ntree=500)...")
message("提示: 计算量较大，可能需要 1-3 分钟，请耐心等待...")

# 2.1 准备数据
target_vars <- c("Watertempreature", "pH", "DO")
predictors <- c("ERA5_Temp", "ERA5_SoilTemp", "ERA5_Radiation", "ERA5_WindSpeed", 
                "ERA5_Precip", "Windspeed_Filled", "ERA5_Pressure",
                "Latitude", "Longitude", "Year_Clean", "Month_Clean",            
                "System_type", "MAT_Final", "MAP_Final")

data_full <- df_ch4 %>%
  select(all_of(c(target_vars, predictors))) %>%
  mutate(across(c(System_type, Month_Clean, Year_Clean), as.factor)) %>%
  as.data.frame()

# 2.2 运行算法
set.seed(2025)
mf_result <- missForest(data_full, ntree = 500, verbose = TRUE)

# 2.3 提取结果
df_final <- df_ch4
df_final[target_vars] <- mf_result$ximp[target_vars]
df_final$Is_Imputed <- apply(df_ch4[target_vars], 1, function(x) any(is.na(x)))

message(paste("插值完成！OOB 误差 (NRMSE):", round(mf_result$OOBerror[1], 4)))

# ==============================================================================
# 3. 防虚高处理
# ==============================================================================
message("\n>>> [Step 3] 正在执行最终边界修正...")

# DO 修正
df_final$DO <- pmax(df_final$DO, 0)
df_final$DO <- pmin(df_final$DO, 25)
do_limit <- quantile(df_ch4$DO, 0.99, na.rm = TRUE)
imputed_idx <- which(is.na(df_ch4$DO))
df_final$DO[imputed_idx] <- pmin(df_final$DO[imputed_idx], do_limit)

# pH 修正
df_final$pH <- pmax(df_final$pH, 4.0)
df_final$pH <- pmin(df_final$pH, 11.0)
ph_limit <- quantile(df_ch4$pH, 0.99, na.rm = TRUE)
imputed_idx_ph <- which(is.na(df_ch4$pH))
df_final$pH[imputed_idx_ph] <- pmin(df_final$pH[imputed_idx_ph], ph_limit)

message("修正完成。")

# ==============================================================================
# 4. 绘图检查并保存到绝对路径
# ==============================================================================
message("\n>>> [Step 4] 正在保存结果...")

# 4.1 保存图片
plot_list <- list()
for(col in target_vars) {
  orig_vals <- df_ch4[[col]]
  final_vals <- df_final[[col]]
  df_plot <- data.frame(Value = final_vals, Type = ifelse(is.na(orig_vals), "Imputed", "Observed"))
  
  plot_list[[col]] <- ggplot(df_plot, aes(x = Value, fill = Type)) +
    geom_density(alpha = 0.5, linewidth = 0.5) +
    scale_fill_manual(values = c("Imputed" = "#E69F00", "Observed" = "gray40")) +
    labs(title = col, x = "", y = "Density") +
    theme_minimal() + theme(legend.position = "top")
}
final_plot <- wrap_plots(plot_list, ncol = 3) + 
  plot_annotation(title = "CH4 最终数据集插值检查 (MissForest)")

# 图片保存路径
plot_path <- file.path(work_dir, "CH4_Final_MissForest_Check.png")
ggsave(plot_path, final_plot, width = 12, height = 4)
message(paste("图片已保存至:", plot_path))

# 4.2 保存最终 CSV (使用绝对路径)
output_filename <- "CH4_Environment_Final_Imputed_MissForest.csv"
output_path <- file.path(work_dir, output_filename)

write_csv(df_final, output_path)

message("\n========================================================")
message(paste(" 处理成功！最终文件绝对路径如下："))
message(output_path)
message(" (请直接复制上面的路径在文件夹中打开)")
message("========================================================")
