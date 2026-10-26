arguments <- commandArgs(trailingOnly = TRUE)
input_file <- if (length(arguments) >= 1) arguments[[1]] else "Source_Data/Fig_2/flux_plot_values.csv"
output_dir <- if (length(arguments) >= 2) arguments[[2]] else "outputs"
summary_file <- file.path(dirname(input_file), "flux_boxplot_points.csv")
if (!file.exists(input_file) || !file.exists(summary_file)) stop("Supply both Fig. 2 plot-value and box-summary tables.")
packages <- c("ggplot2", "patchwork")
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop(paste("Install R packages:", paste(missing_packages, collapse = ", ")))
library(ggplot2)
library(patchwork)
points <- read.csv(input_file, check.names = FALSE, fileEncoding = "UTF-8-BOM")
summaries <- read.csv(summary_file, check.names = FALSE, fileEncoding = "UTF-8-BOM")
categories <- c("Location_type", "System_type", "Climate_Zone", "Season_Clean")
titles <- c("Location type", "System type", "Climate zone", "Season")
orders <- list(Location_type = c("CAS", "IAS"), System_type = c("Fish", "Shrimp", "Crab", "Algae", "Shellfish", "Mixed"), Climate_Zone = c("WTZ", "STZ", "TZ"), Season_Clean = c("Spring", "Summer", "Autumn", "Winter"))
gases <- c("N2O", "CH4", "CO2")
plots <- list()
for (gas in gases) {
    for (category in categories) {
        values <- points[points$Gas == gas & points$Category == category, ]
        boxes <- summaries[summaries$Gas == gas & summaries$Category == category, ]
        visible <- orders[[category]][orders[[category]] %in% values$Group]
        values$Group <- factor(values$Group, levels = visible)
        boxes$Group <- factor(boxes$Group, levels = visible)
        if (sum(boxes$n) != nrow(values)) stop("A plotted sample count does not match the summary table.")
        panel <- ggplot(values, aes(x = Group, y = Plot_value, fill = Group)) +
            geom_violin(trim = TRUE, scale = "width", alpha = 0.6, color = NA) +
            geom_boxplot(data = boxes, aes(x = Group, ymin = Lower_whisker, lower = Q1, middle = Median, upper = Q3, ymax = Upper_whisker), stat = "identity", inherit.aes = FALSE, width = 0.15, fill = "white", color = "grey30", alpha = 0.9) +
            geom_point(data = boxes, aes(x = Group, y = Mean_plot), inherit.aes = FALSE, shape = 18, size = 1.5, color = "darkred") +
            geom_text(data = boxes, aes(x = Group, y = Max_plot, label = Letter), inherit.aes = FALSE, vjust = -0.6, size = 3.5, color = "black") +
            scale_fill_brewer(palette = "Set3") +
            scale_y_continuous(expand = expansion(mult = c(0.05, 0.13))) +
            labs(title = if (gas == "N2O") titles[match(category, categories)] else NULL, x = NULL, y = if (category == "Location_type") paste0(gas, " flux\n(Log-modulus)") else NULL) +
            theme_minimal(base_size = 10) +
            theme(legend.position = "none", panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.text.x = element_text(angle = 40, hjust = 1), plot.title = element_text(hjust = 0.5), panel.border = element_rect(color = "grey35", fill = NA))
        plots[[length(plots) + 1]] <- panel
    }
}
figure <- wrap_plots(plots, ncol = 4)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(output_dir, "Fig2_flux_distributions.pdf"), figure, width = 12, height = 9)
ggsave(file.path(output_dir, "Fig2_flux_distributions.png"), figure, width = 12, height = 9, dpi = 160)
message("Saved Fig. 2 from plot-scale values and precomputed summaries; no tests or models were rerun.")
