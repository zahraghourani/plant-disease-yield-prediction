"""
Ablation study for XGBoost crop yield prediction.

Experiments:
  1. Climate only
  2. Climate + crop type
  3. Climate + disease severity
  4. Full model: climate + crop type + disease severity

Run from the project root:
    python src/yield_ablation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
YIELD_DATA = ROOT / "data" / "yield_data" / "yield_df.csv"
OUTPUT_DIR = ROOT / "results" / "yield_ablation"

TARGET_CROPS = ["Maize", "Wheat", "Rice, paddy", "Potatoes"]
CLIMATE_FEATURES = [
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
TARGET = "hg/ha_yield"

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


def average_disease_severity(crop: str) -> float:
    diseases = CROP_TO_DISEASE.get(crop, [])
    if not diseases:
        return 0.0
    return float(np.mean([DISEASE_YIELD_IMPACT[disease] for disease in diseases]))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(YIELD_DATA)
    df = df[df["Item"].isin(TARGET_CROPS)].copy()

    encoder = LabelEncoder()
    df["crop_encoded"] = encoder.fit_transform(df["Item"])
    df["disease_severity"] = df["Item"].apply(average_disease_severity)
    return df


def evaluate_feature_set(df: pd.DataFrame, experiment: str, features: list[str]) -> dict:
    clean = df[features + [TARGET]].dropna()
    x_train, x_test, y_train, y_test = train_test_split(
        clean[features],
        clean[TARGET],
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return {
        "experiment": experiment,
        "features": ", ".join(features),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "r2": r2_score(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }


def save_plot(results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(results["experiment"], results["r2"], color="#2d6cdf")
    axes[0].set_title("Yield Ablation: R2")
    axes[0].set_ylabel("R2")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylim(0, max(1.0, results["r2"].max() + 0.02))

    axes[1].bar(results["experiment"], results["rmse"], color="#157a58")
    axes[1].set_title("Yield Ablation: RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_dir / "yield_ablation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run() -> pd.DataFrame:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data()

    experiments = [
        ("Climate only", CLIMATE_FEATURES),
        ("Climate + crop type", CLIMATE_FEATURES + ["crop_encoded"]),
        ("Climate + disease severity", CLIMATE_FEATURES + ["disease_severity"]),
        ("Full model", CLIMATE_FEATURES + ["crop_encoded", "disease_severity"]),
    ]

    results = pd.DataFrame(
        [evaluate_feature_set(df, name, features) for name, features in experiments]
    )
    results.to_csv(output_dir / "yield_ablation_results.csv", index=False)
    save_plot(results, output_dir)

    print(results[["experiment", "r2", "rmse"]].to_string(index=False))
    print(f"Saved ablation outputs to {output_dir}")
    return results


if __name__ == "__main__":
    run()
