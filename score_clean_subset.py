"""
score_clean_subset.py -- re-score every model on the CLEAN, de-duplicated test subset.
No retraining and no GPU needed: it only reads existing predictions.

Needs, in the project folder:
   test_predictions_all_38.csv   (columns: filename, y_true, <one column per model>)
   clean_test_files.csv          (made by clean_test_subset.py)

Run:
  & "C:\\Users\\HPZ4-03-Adm01\\miniconda3\\envs\\tf-gpu\\python.exe" score_clean_subset.py

Writes:
   clean_scores_all38.csv           one row per model (accuracy, CI, macro/weighted F1, MCC, McNemar vs best, Holm p, ...)
   clean_per_class_<MODEL>.csv      per-class precision / recall / F1 / support for the top model and key models
   clean_confusion_<MODEL>.csv      confusion matrix
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)

KEY_MODELS = ['ConvNeXtLarge', 'EfficientNetV2S', 'EfficientNetB3']


def norm(p):
    return os.path.normcase(os.path.normpath(str(p)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_p(b, c):
    """b = best right/other wrong, c = best wrong/other right."""
    if b + c == 0:
        return 1.0
    if b + c < 25:                                   # exact binomial for small counts
        return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)           # continuity-corrected chi-square
    return float(stats.chi2.sf(chi2, 1))


def holm(pvals):
    p = np.array(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='test_predictions_all_38.csv')
    ap.add_argument('--clean', default='clean_test_files.csv')
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    clean = pd.read_csv(args.clean)
    models = [c for c in pred.columns if c not in ('filename', 'y_true')]
    pred['_k'] = pred['filename'].map(norm)
    clean['_k'] = clean['filepath'].map(norm)
    sub = pred[pred['_k'].isin(set(clean['_k']))].reset_index(drop=True)
    print(f"Predictions for {len(pred)} original test images; clean subset matched: {len(sub)} "
          f"(clean file lists {len(clean)})")
    if len(sub) != len(clean):
        print("WARNING: some clean files were not found in the predictions file (path mismatch?).")
    if len(sub) == 0:
        print("No matches - check that both files use the same path format.")
        return 1

    y = sub['y_true'].values
    labels = sorted(np.unique(y))
    n = len(y)
    names = {}
    lab_map = clean.set_index('_k')['label'].to_dict()
    for yy, k in zip(sub['y_true'], sub['_k']):
        names[int(yy)] = lab_map.get(k, str(yy))

    correct = np.stack([(sub[m].values == y) for m in models], axis=1)     # (n, M)
    acc_clean = correct.mean(axis=0)
    acc_orig = np.array([(pred[m].values == pred['y_true'].values).mean() for m in models])
    best = int(acc_clean.argmax())
    best_name = models[best]

    rows = []
    pvals = []
    for j, m in enumerate(models):
        yp = sub[m].values
        k = int(correct[:, j].sum())
        lo, hi = wilson(k, n)
        pm, rm, fm, _ = precision_recall_fscore_support(y, yp, labels=labels, average='macro', zero_division=0)
        fw = f1_score(y, yp, labels=labels, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y, yp)
        b_ = int((correct[:, best] & ~correct[:, j]).sum())
        c_ = int((~correct[:, best] & correct[:, j]).sum())
        p = np.nan if j == best else mcnemar_p(b_, c_)
        pvals.append(p)
        rows.append(dict(model=m, acc_clean=acc_clean[j], ci_lo=lo, ci_hi=hi, precision_macro=pm,
                         recall_macro=rm, f1_macro=fm, f1_weighted=fw, mcc=mcc,
                         acc_original_test=acc_orig[j], change_vs_original=acc_clean[j] - acc_orig[j],
                         p_vs_best=p))
    res = pd.DataFrame(rows)
    mask = res['p_vs_best'].notna().values
    res['p_holm'] = np.nan
    res.loc[mask, 'p_holm'] = holm(res.loc[mask, 'p_vs_best'].values)

    # bootstrap: how often is each model the top one?
    rng = np.random.default_rng(42)
    top_count = np.zeros(len(models))
    for _ in range(args.boot):
        idx = rng.integers(0, n, n)
        a = correct[idx].mean(axis=0)
        winners = np.flatnonzero(a == a.max())
        top_count[winners] += 1.0 / len(winners)
    res['bootstrap_top_freq'] = top_count / args.boot

    res = res.sort_values('acc_clean', ascending=False).reset_index(drop=True)
    res['rank'] = np.arange(1, len(res) + 1)
    res.to_csv('clean_scores_all38.csv', index=False, float_format='%.4f')

    pd.set_option('display.width', 200)
    print(f"\nCLEAN test subset: n = {n}   (classes present: {len(labels)})")
    print(f"Best model on the clean subset: {best_name}\n")
    show = res[['rank', 'model', 'acc_clean', 'ci_lo', 'ci_hi', 'f1_macro', 'mcc',
                'acc_original_test', 'p_vs_best', 'p_holm', 'bootstrap_top_freq']].head(12)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for m in KEY_MODELS:
        if m in set(res['model']):
            r = res[res['model'] == m].iloc[0]
            print(f"\n{m}: rank {int(r['rank'])}, acc {r['acc_clean']:.4f} "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] (was {r['acc_original_test']:.4f} on the original test set)")
    n_sig = int((res['p_holm'] < 0.05).sum())
    print(f"\nModels significantly different from the best after Holm correction: {n_sig} of {len(res) - 1}")
    print(f"Models NOT distinguishable from the best (Holm p >= 0.05): {len(res) - 1 - n_sig}")
    print(f"Accuracy range over all models: {res['acc_clean'].min():.4f} - {res['acc_clean'].max():.4f}")

    # per-class tables + confusion matrices
    for m in dict.fromkeys([best_name] + [k for k in KEY_MODELS if k in models]):
        yp = sub[m].values
        p, r, f, s = precision_recall_fscore_support(y, yp, labels=labels, zero_division=0)
        pc = pd.DataFrame({'class': [names[l] for l in labels], 'precision': p, 'recall': r, 'f1': f, 'support': s})
        pc.loc[len(pc)] = ['macro avg (all classes shown)', p.mean(), r.mean(), f.mean(), int(s.sum())]
        pc.to_csv(f'clean_per_class_{m}.csv', index=False, float_format='%.4f')
        cm = confusion_matrix(y, yp, labels=labels)
        pd.DataFrame(cm, index=[names[l] for l in labels], columns=[names[l] for l in labels]).to_csv(
            f'clean_confusion_{m}.csv')
    print("\nSaved: clean_scores_all38.csv, clean_per_class_<model>.csv, clean_confusion_<model>.csv")
    small = [names[l] for l, c in zip(labels, [int((y == l).sum()) for l in labels]) if c < 30]
    if small:
        print("Classes with fewer than 30 clean test images (do NOT interpret their per-class numbers):", small)
    return 0


if __name__ == '__main__':
    sys.exit(main())