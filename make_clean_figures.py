"""
make_clean_figures.py

Regenerates the three classification figures for the DE-DUPLICATED evaluation (n = 1,774):
  Figure 4  figures/all_models_accuracy.png          (uses the numbers below, no CSV needed)
  Figure 7  figures/accuracy_vs_speed.png            (needs ALL_38_MODELS_COMBINED.csv for inference times)
  Figure 8  figures/ConvNeXtLarge_confusion_matrix.png
            needs EITHER clean_confusion_ConvNeXtLarge.csv (15x15 counts)
            OR clean_test_files.csv + test_predictions_all_38.csv (it recomputes the matrix).
Run from the project root (the folder with the CSVs):
    python make_clean_figures.py
It checks its own output: Figure 8 must give n = 1,774 and ConvNeXtLarge accuracy 0.9369.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
N_CLEAN = 1774
N_NOT_SIG = 18          # ranks 1-18 are not significantly different from the best (Holm)

# model, accuracy, Wilson 95% CI low, high  (de-duplicated subset, from clean_scores_all38.csv)
CLEAN = [
    ("EfficientNetB7", 0.9380, 0.926, 0.948),   # rank 1
    ("ConvNeXtLarge", 0.9369, 0.925, 0.947),   # rank 2
    ("EfficientNetB4", 0.9346, 0.922, 0.945),   # rank 3
    ("ConvNeXtXLarge", 0.9340, 0.921, 0.945),   # rank 4
    ("EfficientNetB5", 0.9340, 0.921, 0.945),   # rank 5
    ("EfficientNetV2M", 0.9335, 0.921, 0.944),   # rank 6
    ("EfficientNetV2S", 0.9335, 0.921, 0.944),   # rank 7
    ("ConvNeXtBase", 0.9267, 0.914, 0.938),   # rank 8
    ("EfficientNetV2L", 0.9250, 0.912, 0.936),   # rank 9
    ("DenseNet201", 0.9245, 0.911, 0.936),   # rank 10
    ("ResNet101", 0.9239, 0.911, 0.935),   # rank 11
    ("EfficientNetB6", 0.9239, 0.911, 0.935),   # rank 12
    ("EfficientNetV2B0", 0.9239, 0.911, 0.935),   # rank 13
    ("EfficientNetB2", 0.9233, 0.910, 0.935),   # rank 14
    ("EfficientNetB3", 0.9233, 0.910, 0.935),   # rank 15
    ("ResNet152", 0.9228, 0.909, 0.934),   # rank 16
    ("DenseNet121", 0.9222, 0.909, 0.934),   # rank 17
    ("EfficientNetB0", 0.9205, 0.907, 0.932),   # rank 18
    ("EfficientNetV2B2", 0.9205, 0.907, 0.932),   # rank 19
    ("ConvNeXtSmall", 0.9194, 0.906, 0.931),   # rank 20
    ("EfficientNetV2B1", 0.9188, 0.905, 0.931),   # rank 21
    ("ResNet50", 0.9183, 0.905, 0.930),   # rank 22
    ("EfficientNetV2B3", 0.9183, 0.905, 0.930),   # rank 23
    ("ConvNeXtTiny", 0.9171, 0.903, 0.929),   # rank 24
    ("EfficientNetB1", 0.9166, 0.903, 0.929),   # rank 25
    ("DenseNet169", 0.9160, 0.902, 0.928),   # rank 26
    ("InceptionV3", 0.9132, 0.899, 0.925),   # rank 27
    ("InceptionResNetV2", 0.9126, 0.899, 0.925),   # rank 28
    ("ResNet101V2", 0.9109, 0.897, 0.923),   # rank 29
    ("ResNet152V2", 0.9076, 0.893, 0.920),   # rank 30
    ("ResNet50V2", 0.9076, 0.893, 0.920),   # rank 31
    ("Xception", 0.9042, 0.890, 0.917),   # rank 32
    ("MobileNetV2", 0.9025, 0.888, 0.915),   # rank 33
    ("NASNetLarge", 0.9008, 0.886, 0.914),   # rank 34
    ("VGG16", 0.8968, 0.882, 0.910),   # rank 35
    ("MobileNet", 0.8946, 0.879, 0.908),   # rank 36
    ("VGG19", 0.8822, 0.866, 0.896),   # rank 37
    ("NASNetMobile", 0.8777, 0.862, 0.892),   # rank 38
]

CLASSES = ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight", "Invalid",
           "Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight", "Rice___Brown_Spot",
           "Rice___Healthy", "Rice___Hispa", "Rice___Leaf_Blast", "Wheat___Brown_Rust", "Wheat___Healthy",
           "Wheat___Yellow_Rust"]

def family(m):
    for p, f in (("ConvNeXt", "ConvNeXt"), ("EfficientNetV2", "EfficientNetV2"), ("EfficientNet", "EfficientNet"),
                 ("DenseNet", "DenseNet"), ("ResNet", "ResNet"), ("Inception", "Inception"), ("Xception", "Inception"),
                 ("VGG", "VGG"), ("MobileNet", "MobileNet"), ("NASNet", "NASNet")):
        if m.startswith(p):
            return f
    return "Other"

COLORS = {"ConvNeXt": "#1f77b4", "EfficientNet": "#ff7f0e", "EfficientNetV2": "#2ca02c", "DenseNet": "#d62728",
          "ResNet": "#9467bd", "Inception": "#8c564b", "VGG": "#e377c2", "MobileNet": "#7f7f7f",
          "NASNet": "#bcbd22", "Other": "#17becf"}


def fig4():
    models = [c[0] for c in CLEAN]; acc = np.array([c[1] for c in CLEAN])
    lo = np.array([c[2] for c in CLEAN]); hi = np.array([c[3] for c in CLEAN])
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(38), acc, color=[COLORS[family(m)] for m in models],
           yerr=[acc - lo, hi - acc], error_kw=dict(ecolor="#444444", lw=0.8, capsize=2))
    ax.axvline(N_NOT_SIG - 0.5, color="black", ls="--", lw=1)
    ax.text(N_NOT_SIG / 2 - 0.5, 0.962, "not significantly different from the best model (Holm-adjusted)",
            ha="center", fontsize=9)
    ax.text(N_NOT_SIG + (38 - N_NOT_SIG) / 2 - 0.5, 0.962, "significantly worse", ha="center", fontsize=9)
    ax.set_xticks(range(38)); ax.set_xticklabels(models, rotation=90, fontsize=8)
    ax.set_ylim(0.85, 0.97); ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy of all 38 CNN architectures on the de-duplicated test subset (n = {N_CLEAN:,})")
    fams = list(dict.fromkeys(family(m) for m in models))
    ax.legend(handles=[mpatches.Patch(color=COLORS[f], label=f) for f in fams], loc="upper right",
              bbox_to_anchor=(0.995, 0.90), ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/all_models_accuracy.png", dpi=300); plt.close()
    print("wrote Figure 4 ->", f"{OUT}/all_models_accuracy.png")


def fig7():
    csv = "ALL_38_MODELS_COMBINED.csv"
    if not os.path.exists(csv):
        print("SKIP Figure 7: need", csv); return
    t = pd.read_csv(csv).set_index("model_name")["inference_time_ms"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for m, a, _, _ in CLEAN:
        if m not in t.index:
            print("  no inference time for", m); continue
        ax.scatter(t[m], a, color=COLORS[family(m)], s=60, alpha=0.85)
    for m, a, _, _ in CLEAN[:7]:
        if m in t.index:
            ax.annotate(m, (t[m], a), textcoords="offset points", xytext=(5, 4), fontsize=8)
    fams = list(dict.fromkeys(family(m) for m, *_ in CLEAN))
    ax.legend(handles=[mpatches.Patch(color=COLORS[f], label=f) for f in fams], loc="lower right", fontsize=8, ncol=2)
    ax.set_xlabel("Batched per-image inference time (ms), lower is better")
    ax.set_ylabel(f"Accuracy on the de-duplicated test subset (n = {N_CLEAN:,})")
    ax.set_title("Accuracy vs speed (all 38 models)"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/accuracy_vs_speed.png", dpi=300); plt.close()
    print("wrote Figure 7 ->", f"{OUT}/accuracy_vs_speed.png")


def load_confusion():
    f = "clean_confusion_ConvNeXtLarge.csv"
    if os.path.exists(f):
        for kw in (dict(index_col=0), dict(header=None)):
            try:
                m = pd.read_csv(f, **kw).values.astype(float)
                if m.shape == (15, 15):
                    return m.astype(int), "from " + f
            except Exception:
                pass
        print("  could not read", f, "as a 15x15 matrix; recomputing instead")
    if os.path.exists("clean_test_files.csv") and os.path.exists("test_predictions_all_38.csv"):
        keep = pd.read_csv("clean_test_files.csv")
        col = next((c for c in keep.columns if any(k in c.lower() for k in ("file", "path", "name"))), keep.columns[0])
        names = set(os.path.basename(str(x)) for x in keep[col])
        pr = pd.read_csv("test_predictions_all_38.csv")
        sel = pr[pr["filename"].map(lambda x: os.path.basename(str(x)) in names)]
        m = np.zeros((15, 15), int)
        for y, p in zip(sel["y_true"], sel["ConvNeXtLarge"]):
            m[int(y), int(p)] += 1
        return m, "recomputed from clean_test_files.csv + test_predictions_all_38.csv"
    return None, None


def fig8():
    m, how = load_confusion()
    if m is None:
        print("SKIP Figure 8: need clean_confusion_ConvNeXtLarge.csv or clean_test_files.csv + test_predictions_all_38.csv"); return
    n, acc = int(m.sum()), np.trace(m) / m.sum()
    print(f"  confusion matrix {how}: n = {n}, accuracy = {acc:.4f}")
    if n != N_CLEAN or abs(acc - 0.9369) > 0.0006:
        print(f"  !! CHECK FAILED: expected n = {N_CLEAN} and accuracy 0.9369 -- do not use this figure until resolved")
    pretty = [c.replace("___", " ").replace("_", " ") for c in CLASSES]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(15)); ax.set_yticks(range(15))
    ax.set_xticklabels(pretty, rotation=90, fontsize=8); ax.set_yticklabels(pretty, fontsize=8)
    for i in range(15):
        for j in range(15):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > m.max() * 0.5 else "black")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"ConvNeXtLarge, de-duplicated test subset (n = {n:,})")
    plt.colorbar(im, fraction=0.046); plt.tight_layout()
    plt.savefig(f"{OUT}/ConvNeXtLarge_confusion_matrix.png", dpi=300); plt.close()
    print("wrote Figure 8 ->", f"{OUT}/ConvNeXtLarge_confusion_matrix.png")


if __name__ == "__main__":
    fig4(); fig7(); fig8()