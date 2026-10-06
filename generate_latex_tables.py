"""
generate_latex_tables.py

Builds the two data-dependent LaTeX tables from your real result files, so
no number is ever typed by hand:

    tables/table_full38.tex     (Table: full 38-model benchmark)
    tables/table_perclass.tex   (Table: ConvNeXtLarge per-class results)

Run from the PROJECT ROOT (the folder that contains table4_full_38_model_stats.csv,
test_predictions_all_38.csv and results/), then copy the generated tables/ folder
into the LaTeX project, next to main.tex:

    python generate_latex_tables.py

It also prints sanity checks and a few numbers worth quoting in the text.
"""
import os
import re
import sys
import numpy as np
import pandas as pd

STATS_CSV = "table4_full_38_model_stats.csv"
PRED_CSV = "test_predictions_all_38.csv"
PERSON_CSVS = ["results/zahra_final_results.csv",
               "results/sireen_final_results.csv",
               "results/tala_final_results.csv"]
FEATURED = "ConvNeXtLarge"
OUT_DIR = "tables"

# Alphabetical order = the class_indices order used in training.
CLASS_NAMES = [
    "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight",
    "Invalid", "Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight",
    "Rice___Brown_Spot", "Rice___Healthy", "Rice___Hispa", "Rice___Leaf_Blast",
    "Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust",
]


def pretty(name):
    return name.replace("___", " ").replace("_", " ")


def fmt_p(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def parse_list(s, expected=15):
    s = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", str(s))   # tolerate np.float64(0.98) style
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    vals = [float(x) for x in nums]
    if len(vals) != expected:
        sys.exit(f"Could not parse {expected} per-class values from: {str(s)[:80]}... (got {len(vals)})")
    return vals


# ───────────────────────── full 38-model table ─────────────────────────
def build_full38():
    df = pd.read_csv(STATS_CSV).sort_values("accuracy", ascending=False).reset_index(drop=True)
    assert len(df) == 38, f"expected 38 models, found {len(df)}"
    n_test = None
    if os.path.exists(PRED_CSV):
        n_test = len(pd.read_csv(PRED_CSV, usecols=["y_true"]))
    n_txt = f"{n_test:,}".replace(",", "{,}") if n_test else "4{,}792"

    lines = []
    lines.append("{\\tiny")
    lines.append("\\begin{longtable}{L{1.9cm}ccccccccc}")
    lines.append(
        "\\caption{Full 38-model benchmark on the corrected, duplicate-leakage-free test set "
        f"($n = {n_txt}$; all 15 output classes). Acc = accuracy; subscript m = macro, w = weighted; "
        "MCC = Matthews CC; AUC = macro ROC-AUC (one-vs-rest); CI = Wilson 95\\% interval on accuracy; "
        f"$p$ = McNemar's test against {FEATURED} (uncorrected for multiple comparisons). "
        "\\textbf{Bold} = best per metric.}")
    lines.append("\\label{tab:full}\\\\")
    hdr = ("\\textbf{Model}&\\textbf{Acc}&\\textbf{Prec\\textsubscript{m}}&\\textbf{Rec\\textsubscript{m}}&"
           "\\textbf{F\\textsubscript{1m}}&\\textbf{F\\textsubscript{1w}}&\\textbf{MCC}&\\textbf{AUC}&"
           "\\textbf{95\\% CI}&\\textbf{$p$}\\\\")
    lines.append("\\toprule")
    lines.append(hdr)
    lines.append("\\midrule\\endfirsthead")
    lines.append("\\multicolumn{10}{c}{\\tablename~\\thetable{} (continued)}\\\\\\toprule")
    lines.append(hdr)
    lines.append("\\midrule\\endhead")
    lines.append("\\midrule\\multicolumn{10}{r}{\\textit{Continued\\ldots}}\\\\\\endfoot")
    lines.append("\\bottomrule\\endlastfoot")

    best = {c: df[c].max() for c in ["accuracy", "precision_macro", "recall_macro",
                                      "f1_macro", "f1_weighted", "mcc", "roc_auc_ovr"]}

    def cell(col, row):
        v = row[col]
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if abs(v - best[col]) < 1e-12 else s

    for _, r in df.iterrows():
        ci = f"[{r['ci_95_low']:.3f},{r['ci_95_high']:.3f}]"
        lines.append(
            f"{r['model_name']}&{cell('accuracy', r)}&{cell('precision_macro', r)}&"
            f"{cell('recall_macro', r)}&{cell('f1_macro', r)}&{cell('f1_weighted', r)}&"
            f"{cell('mcc', r)}&{cell('roc_auc_ovr', r)}&{ci}&{fmt_p(r['p_value_vs_top'])}\\\\")
    lines.append("\\end{longtable}")
    lines.append("}")
    return "\n".join(lines) + "\n", df


# ───────────────────────── per-class table ─────────────────────────
def build_perclass():
    row = None
    for path in PERSON_CSVS:
        if os.path.exists(path):
            d = pd.read_csv(path)
            m = d[d["model_name"] == FEATURED]
            if len(m):
                row = m.iloc[-1]
                src = path
    if row is None:
        sys.exit(f"{FEATURED} not found in any of {PERSON_CSVS}")
    prec = parse_list(row["precision_per_class"])
    rec = parse_list(row["recall_per_class"])
    f1 = parse_list(row["f1_per_class"])

    support = None
    if os.path.exists(PRED_CSV):
        y = pd.read_csv(PRED_CSV, usecols=["y_true"])["y_true"].values
        support = [int((y == i).sum()) for i in range(15)]

    keep = [i for i, c in enumerate(CLASS_NAMES) if c != "Invalid"]
    macro15_f1 = float(np.mean(f1))

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Per-class performance of {FEATURED} on the corrected, duplicate-leakage-free "
        "test set (14 disease and healthy classes; the Invalid class is excluded from this table). "
        "Support = number of test images. The macro average is taken over the 14 classes shown.}")
    lines.append("\\label{tab:perclass}")
    lines.append("{\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & "
                 "\\textbf{F\\textsubscript{1}} & \\textbf{Support}\\\\")
    lines.append("\\midrule")
    for i in keep:
        sup = str(support[i]) if support else "--"
        lines.append(f"{pretty(CLASS_NAMES[i])} & {prec[i]:.3f} & {rec[i]:.3f} & {f1[i]:.3f} & {sup}\\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Macro avg (14 classes)}} & \\textbf{{{np.mean([prec[i] for i in keep]):.3f}}} & "
                 f"\\textbf{{{np.mean([rec[i] for i in keep]):.3f}}} & "
                 f"\\textbf{{{np.mean([f1[i] for i in keep]):.3f}}} & "
                 f"{sum(support[i] for i in keep) if support else '--'}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    hardest = sorted(keep, key=lambda i: f1[i])[:3]
    n_ge = sum(1 for i in keep if f1[i] >= 0.87)
    info = dict(src=src, macro15_f1=macro15_f1, hardest=[(pretty(CLASS_NAMES[i]), f1[i]) for i in hardest],
                n_ge_087=n_ge, acc_in_csv=float(row.get("accuracy", np.nan)))
    return "\n".join(lines) + "\n", info


def epoch_stats():
    frames = [pd.read_csv(p) for p in PERSON_CSVS if os.path.exists(p)]
    if not frames:
        return None
    d = pd.concat(frames)
    if "epochs_trained" not in d.columns:
        return None
    e = d["epochs_trained"].astype(float)
    return float(e.mean()), float(e.min()), float(e.max()), len(e)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    full_tex, stats_df = build_full38()
    open(os.path.join(OUT_DIR, "table_full38.tex"), "w", encoding="utf-8").write(full_tex)
    print(f"wrote {OUT_DIR}/table_full38.tex  ({len(stats_df)} models)")

    pc_tex, info = build_perclass()
    open(os.path.join(OUT_DIR, "table_perclass.tex"), "w", encoding="utf-8").write(pc_tex)
    print(f"wrote {OUT_DIR}/table_perclass.tex  (from {info['src']})")

    print("\n=== SANITY CHECKS ===")
    top = stats_df.iloc[0]
    print(f"top model in stats file : {top['model_name']}  acc={top['accuracy']:.4f}   (expected ConvNeXtLarge 0.9610)")
    print(f"{FEATURED} accuracy in per-person CSV: {info['acc_in_csv']:.4f}")
    mf1 = stats_df.loc[stats_df.model_name == FEATURED, "f1_macro"].values[0]
    print(f"15-class macro F1 from per-class values: {info['macro15_f1']:.4f}  vs  f1_macro in stats: {mf1:.4f}"
          f"   -> {'OK' if abs(info['macro15_f1'] - mf1) < 0.002 else 'MISMATCH - check the CSV is from the final run'}")

    print("\n=== NUMBERS YOU MAY WANT TO QUOTE IN THE TEXT ===")
    print("three lowest-F1 classes (excl. Invalid):",
          "; ".join(f"{n} (F1={v:.3f})" for n, v in info["hardest"]))
    print(f"classes with F1 >= 0.87 (of 14): {info['n_ge_087']}")
    es = epoch_stats()
    if es:
        print(f"epochs actually trained across {es[3]} models: mean {es[0]:.1f}, min {es[1]:.0f}, max {es[2]:.0f}")
    print("\nNow copy the 'tables' folder next to main.tex and recompile.")


if __name__ == "__main__":
    main()