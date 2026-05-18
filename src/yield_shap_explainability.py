"""
SHAP explainability for the XGBoost crop yield model.

Creates:
  - SHAP summary beeswarm plot
  - SHAP summary bar plot
  - SHAP dependence plot for disease_severity

Run from the project root:
    python src/yield_shap_explainability.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
YIELD_DATA = ROOT / "data" / "yield_data" / "yield_df.csv"
OUTPUT_DIR = ROOT / "results" / "shap_yield"

TARGET_CROPS = ["Maize", "Wheat", "Rice, paddy", "Potatoes"]

DISEASE_YIELD_IMPACT = {
    "Corn___Common_Rust": 0.35,
    "Corn___Leaf_Blight": 0.40,
    "Corn___Healthy": 0.00,
    "Potato___Early_Blight": 0.20,
    "Potato___Late_Blight": 0.45,
    "Potato___Healthy": 0.00,
    "Rice___Brown_Spot": 0.25,
    "Rice___Healthy": 0.00,
    "Rice___Hispa": 0.30,
    "Rice___Leaf_Blast": 0.50,
    "Wheat___Brown_Rust": 0.35,
    "Wheat___Healthy": 0.00,
    "Wheat___Yellow_Rust": 0.40,
    "Invalid": 0.00,
}

CROP_TO_DISEASE = {
    "Maize": ["Corn___Common_Rust", "Corn___Leaf_Blight", "Corn___Healthy"],
    "Wheat": ["Wheat___Brown_Rust", "Wheat___Yellow_Rust", "Wheat___Healthy"],
    "Rice, paddy": [
        "Rice___Brown_Spot",
        "Rice___Hispa",
        "Rice___Leaf_Blast",
        "Rice___Healthy",
    ],
    "Potatoes": ["Potato___Early_Blight", "Potato___Late_Blight", "Potato___Healthy"],
}

FEATURES = [
    "crop_encoded",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "disease_severity",
]
TARGET = "hg/ha_yield"


def average_disease_severity(crop: str) -> float:
    diseases = CROP_TO_DISEASE.get(crop, [])
    if not diseases:
        return 0.0
    return float(np.mean([DISEASE_YIELD_IMPACT[disease] for disease in diseases]))


def prepare_yield_data():
    df = pd.read_csv(YIELD_DATA)
    df = df[df["Item"].isin(TARGET_CROPS)].copy()

    label_encoder = LabelEncoder()
    df["crop_encoded"] = label_encoder.fit_transform(df["Item"])
    df["disease_severity"] = df["Item"].apply(average_disease_severity)

    clean = df[FEATURES + [TARGET]].dropna()
    x = clean[FEATURES]
    y = clean[TARGET]

    return train_test_split(x, y, test_size=0.2, random_state=42), label_encoder


def train_xgboost(x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model


def save_shap_plots(model: XGBRegressor, x_sample: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    plt.figure()
    shap.summary_plot(shap_values, x_sample, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, x_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.dependence_plot("disease_severity", shap_values, x_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_dependence_disease_severity.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame(
        {
            "feature": x_sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    return importance


def write_metrics(model: XGBRegressor, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    preds = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    metrics = pd.DataFrame([{"r2": r2, "rmse": rmse, "n_test": len(x_test)}])
    metrics.to_csv(output_dir / "xgboost_shap_model_metrics.csv", index=False)


def run(sample_size: int, output_dir: Path) -> None:
    (x_train, x_test, y_train, y_test), label_encoder = prepare_yield_data()
    model = train_xgboost(x_train, y_train)

    if sample_size and len(x_test) > sample_size:
        x_sample = x_test.sample(n=sample_size, random_state=42)
    else:
        x_sample = x_test.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "crop": label_encoder.classes_,
            "crop_encoded": label_encoder.transform(label_encoder.classes_),
            "disease_severity": [average_disease_severity(crop) for crop in label_encoder.classes_],
        }
    ).to_csv(output_dir / "crop_encoding_and_severity.csv", index=False)

    write_metrics(model, x_test, y_test, output_dir)
    importance = save_shap_plots(model, x_sample, output_dir)

    print(f"Saved SHAP outputs to {output_dir}")
    print("Top SHAP features:")
    print(importance.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SHAP plots for XGBoost yield model.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Rows sampled from test set for SHAP plots.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(sample_size=args.sample_size, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
