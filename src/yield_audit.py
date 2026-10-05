"""
yield_audit.py  --  audit of the Stage-3 yield models

Answers four questions with real numbers from data/yield_data/yield_df.csv:

 1. Does 'disease_severity' carry ANY information beyond crop identity?
    (In training it is a fixed function of crop: Maize/Wheat 0.25, Rice 0.2625,
     Potatoes 0.2167.)
 2. Under the current random 80/20 row split, how many test rows have a
    (country, crop) pair that is also in the training set?
    (country-level leakage, analogous to the duplicate-image leakage)
 3. How do XGBoost AND Random Forest score under a COUNTRY-HELD-OUT
    5-fold GroupKFold, compared with the random split used in the paper?
 4. Does the severity ablation (Climate / +crop / +severity / full) survive the
    grouped evaluation, for both models?

Run from the project root:
    python src/yield_audit.py
Outputs: results/yield_audit/yield_audit_results.csv  (+ console summary)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
YIELD_DATA = ROOT / "data" / "yield_data" / "yield_df.csv"
OUT_DIR = ROOT / "results" / "yield_audit"

TARGET_CROPS = ["Maize", "Wheat", "Rice, paddy", "Potatoes"]
CLIMATE = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
TARGET = "hg/ha_yield"

# Same literature values and averaging as yield_ablation.py / yield_prediction.py
IMPACT = {
    "Corn___Common_Rust": 0.35, "Corn___Leaf_Blight": 0.40, "Corn___Healthy": 0.00,
    "Potato___Early_Blight": 0.20, "Potato___Late_Blight": 0.45, "Potato___Healthy": 0.00,
    "Rice___Brown_Spot": 0.25, "Rice___Healthy": 0.00, "Rice___Hispa": 0.30, "Rice___Leaf_Blast": 0.50,
    "Wheat___Brown_Rust": 0.35, "Wheat___Healthy": 0.00, "Wheat___Yellow_Rust": 0.40,
}
CROP_TO_DISEASE = {
    "Maize": ["Corn___Common_Rust", "Corn___Leaf_Blight", "Corn___Healthy"],
    "Wheat": ["Wheat___Brown_Rust", "Wheat___Yellow_Rust", "Wheat___Healthy"],
    "Rice, paddy": ["Rice___Brown_Spot", "Rice___Hispa", "Rice___Leaf_Blast", "Rice___Healthy"],
    "Potatoes": ["Potato___Early_Blight", "Potato___Late_Blight", "Potato___Healthy"],
}


def severity(crop):
    return float(np.mean([IMPACT[d] for d in CROP_TO_DISEASE[crop]]))


def load():
    if not YIELD_DATA.exists():
        sys.exit(f"Not found: {YIELD_DATA}")
    df = pd.read_csv(YIELD_DATA)
    if "Area" not in df.columns:
        sys.exit(f"No 'Area' (country) column; columns are {df.columns.tolist()}")
    df = df[df["Item"].isin(TARGET_CROPS)].copy()
    df["crop_encoded"] = LabelEncoder().fit_transform(df["Item"])
    df["disease_severity"] = df["Item"].map(severity)
    return df.dropna(subset=CLIMATE + [TARGET]).reset_index(drop=True)


def make(name):
    if name == "XGBoost":
        return XGBRegressor(n_estimators=100, random_state=42)
    if name == "Linear Regression":
        return LinearRegression()
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"rows: {len(df)}   countries: {df['Area'].nunique()}   crops: {df['Item'].nunique()}")

    # ---------------- 1. severity vs crop identity ----------------
    print("\n=== 1. IS 'disease_severity' JUST A CROP LOOKUP? ===")
    ct = df.groupby("Item")["disease_severity"].agg(["nunique", "min", "max"])
    print(ct.to_string())
    print(f"distinct severity values overall: {df['disease_severity'].nunique()} "
          f"(vs {df['Item'].nunique()} crops)")
    print("-> every row of a given crop has the SAME severity; it cannot encode disease.")

    # ---------------- 2. country leakage under the random split ----------------
    print("\n=== 2. COUNTRY OVERLAP UNDER THE PAPER'S RANDOM 80/20 ROW SPLIT ===")
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    train_pairs = set(zip(tr["Area"], tr["Item"]))
    share_pair = np.mean([(a, i) in train_pairs for a, i in zip(te["Area"], te["Item"])])
    share_area = te["Area"].isin(set(tr["Area"])).mean()
    print(f"test rows whose country appears in train:            {share_area:.1%}")
    print(f"test rows whose (country, crop) pair is in train:    {share_pair:.1%}")
    rain_nuniq = df.groupby("Area")["average_rain_fall_mm_per_year"].nunique()
    print(f"countries with a single rainfall value across all years: "
          f"{(rain_nuniq == 1).mean():.1%}  (rainfall then works as a country fingerprint)")

    # ---------------- 3 & 4. random split vs country-held-out, ablation ----------------
    experiments = [
        ("Climate only", CLIMATE),
        ("Climate + crop type", CLIMATE + ["crop_encoded"]),
        ("Climate + severity", CLIMATE + ["disease_severity"]),
        ("Full (climate+crop+severity)", CLIMATE + ["crop_encoded", "disease_severity"]),
    ]
    rows = []
    gkf = GroupKFold(n_splits=5)
    for mname in ["XGBoost", "Random Forest", "Linear Regression"]:
        for ename, feats in experiments:
            # random split exactly as in the paper
            m = make(mname).fit(tr[feats], tr[TARGET])
            p = m.predict(te[feats])
            r2_rand, rmse_rand = r2_score(te[TARGET], p), rmse(te[TARGET], p)
            # country-held-out CV
            r2s, rms = [], []
            for tri, tei in gkf.split(df, groups=df["Area"]):
                a, b = df.iloc[tri], df.iloc[tei]
                mm = make(mname).fit(a[feats], a[TARGET])
                pp = mm.predict(b[feats])
                r2s.append(r2_score(b[TARGET], pp)); rms.append(rmse(b[TARGET], pp))
            rows.append(dict(model=mname, experiment=ename,
                             r2_random_split=r2_rand, rmse_random_split=rmse_rand,
                             r2_country_heldout_mean=np.mean(r2s), r2_country_heldout_std=np.std(r2s),
                             rmse_country_heldout_mean=np.mean(rms)))
    # Baseline with NO model: predict each crop's mean training yield.
    def crop_mean_pred(a, b):
        means = a.groupby("Item")[TARGET].mean()
        return b["Item"].map(means).values
    p = crop_mean_pred(tr, te)
    r2s, rms = [], []
    for tri, tei in gkf.split(df, groups=df["Area"]):
        a, b = df.iloc[tri], df.iloc[tei]
        pp = crop_mean_pred(a, b)
        r2s.append(r2_score(b[TARGET], pp)); rms.append(rmse(b[TARGET], pp))
    rows.append(dict(model="Crop-mean baseline", experiment="(no features, no model)",
                     r2_random_split=r2_score(te[TARGET], p), rmse_random_split=rmse(te[TARGET], p),
                     r2_country_heldout_mean=np.mean(r2s), r2_country_heldout_std=np.std(r2s),
                     rmse_country_heldout_mean=np.mean(rms)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "yield_audit_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== 3/4. RANDOM SPLIT (paper) vs COUNTRY-HELD-OUT 5-FOLD ===")
    show = res.copy()
    for c in show.columns[2:]:
        show[c] = show[c].round(4) if "r2" in c else show[c].round(0)
    print(show.to_string(index=False))

    print("\n=== HOW TO READ THIS ===")
    full = res[res.experiment.str.startswith("Full")].set_index("model")
    crop = res[res.experiment == "Climate + crop type"].set_index("model")
    for mname in ["XGBoost", "Random Forest", "Linear Regression"]:
        d_rand = full.loc[mname, "r2_random_split"] - crop.loc[mname, "r2_random_split"]
        d_grp = full.loc[mname, "r2_country_heldout_mean"] - crop.loc[mname, "r2_country_heldout_mean"]
        print(f"{mname:14s} gain from adding severity on top of crop: random split {d_rand:+.4f} | "
              f"country-held-out {d_grp:+.4f}")
    print("A big drop from the random-split R2 to the country-held-out R2 means the paper's R2 is "
          "inflated by country overlap.\nA gain near zero (or inconsistent in sign) when severity "
          "is added to crop means severity adds nothing beyond crop identity.")
    print(f"\nsaved: {OUT_DIR / 'yield_audit_results.csv'}")


if __name__ == "__main__":
    main()