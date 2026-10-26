# Aquaculture_pond_greenhouse_gases_emission
This repository contains source data for main Figures 1–6 and the key analysis
scripts accompanying the study. Data are grouped by figure; scripts are numbered
by analysis stage. Model settings, predictor order and random seeds remain in the
code. Separate parameter exports, optimization logs and duplicate scripts are
omitted.

## Contents

| Location            | Contents                                                     |
| ------------------- | ------------------------------------------------------------ |
| `Source_Data/Fig_1` | Site locations, observed water quality and group annotations |
| `Source_Data/Fig_2` | Observed gas fluxes and grouping variables                   |
| `Source_Data/Fig_3` | Predictor importance and displayed SHAP dependence data      |
| `Source_Data/Fig_4` | Pond fluxes and provincial emission inventories              |
| `Source_Data/Fig_5` | Future emissions, reduction rates and avoided-flux map values |
| `Source_Data/Fig_6` | Management combinations and spatial allocation results       |
| `Scripts/01–03`     | Gas-specific imputation                                      |
| `Scripts/04–08`     | Water-quality comparisons, final models and SHAP export      |
| `Scripts/09–11`     | Baseline flux prediction                                     |
| `Scripts/12–14`     | Future pathways and emission aggregation                     |
| `Scripts/15–16`     | Management combinations and spatial prioritization           |
| `Scripts/17–18`     | Plots from supplied source data                              |

All source-data tables are uncompressed xls files. Rounding rules and plot-scale definitions are given in `README.md`.
`Scripts/_shared` contains path handling and shared inventory, Monte Carlo and
spatial integration functions.

## Software requirements

The Python workflows use the following packages. Version constraints are retained
from the original dependency list:

```text
python -m pip install numpy pandas scikit-learn==1.8.0 lightgbm==4.6.0 shap==0.50.0 scipy matplotlib seaborn joblib statsmodels scikit-posthocs xarray netCDF4 cftime geopandas cartopy shapely pyproj tqdm
```

The R workflows require the following packages:

```r
install.packages(c("caret", "gbm", "missForest", "ggplot2", "ggpubr", "multcompView", "patchwork", "rstatix", "stringr", "tidyverse"))
```

Individual scripts use only a subset of these packages. Redrawing the saved
Python panels requires only numpy, pandas and matplotlib.

## Using the supplied data

From this repository's root directory:

```text
python -m pip install numpy pandas matplotlib
python Scripts/18_plot_saved_main_panels.py
Rscript Scripts/17_plot_flux_distributions.R Source_Data/Fig_2/flux_plot_values.csv outputs
```

These commands redraw Fig. 5a–c and Fig. 6a–b in Python, and Fig. 2 in R. Maps
require separate boundary and line layers; final manuscript layouts are not
reproduced.

N2O flux is in micrograms m-2 h-1; CH4 and CO2 fluxes are in milligrams m-2 h-1.
Negative physical fluxes indicate net uptake. Fig. 1 and Fig. 2 distribution tables contain plotted axis values, not model-training inputs. Water-quality values are observed, not imputed. The original flux dataset for Fig. 2 contains 2,193 records with overlapping gas measurements.
Row_ID and the site-table `number` are not reference identifiers; the original
site table's correspondence to retained studies was not independently rebuilt.
SHAP values are on the log-modulus response scale, not physical fluxes.

Fig. 4 includes Monte Carlo means and 95% interval endpoints. Fig. 5 uses five-GCM
ranges and standard deviations; reduction rates average GCM-specific percentages.
Fig. 6 uses deterministic IPSL-CM6A-LR projections under SSP585. Its random-allocation
band is not GCM uncertainty, and provincial percentages refer to each province's
pond area.

## Running the analysis scripts

```text
python Scripts/run.py list
python Scripts/run.py 05 --workspace /path/to/working_inputs
```

The second example requires `meta/N2O_Subset_Imputed_BRT.csv` in the specified
workspace. Imputation scripts require the original inputs in `literature`;
prediction requires model files, climate fields and spatial inputs. Optional
`--external-config` supplies JSON path mappings, and `--rscript` selects R.
Script 08 requires a full upstream project with `data/analysis` and `models`.

This compact repository does not include the complete extraction database,
imputed training tables, climate/GIS inputs, fitted models or supplementary-figure
data. It supports inspection of main-figure values and the retained methods, not
an end-to-end rerun using these files alone. Please cite the accompanying article.
