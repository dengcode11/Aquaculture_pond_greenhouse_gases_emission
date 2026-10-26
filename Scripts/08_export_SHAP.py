from pathlib import Path
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import shap


SETTINGS = {
    "N2O": {"data": "N2O_Subset_Imputed_BRT.csv", "model": "n2o_random_forest.joblib", "features": ["Watertempreature", "pH", "DO", "NH4N", "NO3N", "MAT_Final", "MAP_Final", "Monthly_Precipitation_ERA5", "Windspeed_Filled", "Longitude"]},
    "CH4": {"data": "CH4_Environment_Final_Imputed_MissForest.csv", "model": "ch4_random_forest.joblib", "features": ["Longitude", "Watertempreature", "DO", "MAT_Final", "Latitude", "Atomsphericpressure_Filled", "ERA5_SoilTemp", "Airtemperature_Filled", "pH", "MAP_Final"]},
    "CO2": {"data": "CO2_Subset_Imputed_BRT.csv", "model": "co2_lightgbm.joblib", "features": ["DO", "pH", "chla", "Watertempreature", "Airtemperature_Filled", "Monthly_Precipitation_ERA5", "Windspeed_Filled", "ERA5_Radiation", "Atomsphericpressure_Filled", "ERA5_SoilTemp"]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gas", choices=["all", *SETTINGS], default="all")
    arguments = parser.parse_args()
    root = arguments.root
    output = root / "outputs/shap"
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for gas in SETTINGS if arguments.gas == "all" else [arguments.gas]:
        settings = SETTINGS[gas]
        data_path = root / "data/analysis" / (gas.lower() + "_imputed.csv")
        model_path = root / "models" / settings["model"]
        data = pd.read_csv(data_path)
        modified_pressure = 0
        if gas == "CH4":
            mask = data.Atomsphericpressure_Filled < 900
            modified_pressure = int(mask.sum())
            data.loc[mask, "Atomsphericpressure_Filled"] *= 10
            data["CH4_flux"] = pd.to_numeric(data.CH4_flux, errors="coerce")
        if gas == "CO2":
            data = data.rename(columns={"Chl伪": "chla", "Chla": "chla"})
        selected = data[settings["features"] + [gas + "_flux"]].dropna()
        predictors = selected[settings["features"]]
        model = joblib.load(model_path)
        if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != settings["features"]:
            raise ValueError(f"Feature order mismatch for {gas}")
        print(f"Exporting {gas} SHAP source data ({len(selected)} records)", flush=True)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(predictors, check_additivity=False if gas == "CH4" else True)
        if isinstance(values, list):
            values = values[0]
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        predictions = np.asarray(model.predict(predictors))
        difference = float(np.max(np.abs(expected + values.sum(axis=1) - predictions)))
        if not np.isfinite(values).all() or difference > 1e-4:
            raise ValueError(f"SHAP sum-to-prediction check failed: {gas}, {difference}")
        exported = data.loc[selected.index, ["Row_ID", "Study_ID"]].copy()
        exported["Study_ID"] = exported.Study_ID
        exported["Reference_ID"] = exported.Study_ID
        exported[gas + "_flux"] = selected[gas + "_flux"]
        exported["prediction_log_modulus"] = predictions
        exported["SHAP_expected_value"] = expected
        for feature_index, feature in enumerate(settings["features"]):
            exported[feature] = predictors[feature]
            exported["SHAP__" + feature] = values[:, feature_index]
        exported.to_csv(output / f"{gas}_SHAP_dependence_source.csv", index=False, encoding="utf-8-sig")
        importance = pd.DataFrame({"Feature": settings["features"], "Mean_absolute_SHAP": np.abs(values).mean(axis=0)}).sort_values("Mean_absolute_SHAP", ascending=False)
        importance.to_csv(output / f"{gas}_SHAP_importance.csv", index=False, encoding="utf-8-sig")
        reports.append({"gas": gas, "records": len(selected), "CH4_pressure_adjusted_records": modified_pressure, "max_SHAP_additivity_error": difference, "top_three": importance.Feature.head(3).tolist(), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "model_parameters": model.get_params(), "shap_version": shap.__version__, "scope": "Recomputed explanation of saved model with original SHAP-script preprocessing; no fitting or future projection rerun"})
    (output / f"SHAP_export_validation_{arguments.gas}.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
