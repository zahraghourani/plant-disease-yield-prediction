"""
clean_test_subset.py -- pick the part of your EXISTING test set that is genuinely clean.

No retraining needed. The existing 38 models are re-scored on this subset only.

Rules for a test image to be kept:
  * none of its duplicates / near-duplicates (exact bytes, same filename base, or
    perceptual-hash match incl. flips & rotations, within the same class) is in TRAIN or VAL
  * only ONE image per duplicate group is kept (duplicates inside test are not independent samples)

Run from the project folder (needs the ORIGINAL splits.csv, hash_cache.npz, hash_check.py,
build_grouped_split.py in the same folder):

  & "C:\\Users\\HPZ4-03-Adm01\\miniconda3\\envs\\tf-gpu\\python.exe" clean_test_subset.py

Output: clean_test_files.csv   (filepath, label)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import hashlib

from hash_check import compute_hashes, POP


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='splits.csv')
    ap.add_argument('--cache', default='hash_cache.npz')
    ap.add_argument('--out', default='clean_test_files.csv')
    ap.add_argument('--thr', type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.splits)
    paths = df['filepath'].tolist()
    N = len(df)
    if 'base_id' not in df.columns:
        df['base_id'] = df['label'] + '::' + df['filepath'].map(lambda p: os.path.splitext(os.path.basename(p))[0])

    H = None
    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        if list(z['paths']) == paths:
            H, ok = z['H'], z['ok']
            print("Loaded cached hashes")
    if H is None:
        print("Hashing images (several minutes)...")
        H, ok = compute_hashes(paths, True, 'dataset')
        np.savez(args.cache, paths=np.array(paths, dtype=object), H=H, ok=ok)

    uf = UF(N)
    print("Grouping: exact copies (MD5)...")
    seen = {}
    for i, p in enumerate(paths):
        m = md5(p)
        if m in seen:
            uf.union(i, seen[m])
        else:
            seen[m] = i
    first = {}
    for i, b in enumerate(df['base_id'].tolist()):
        if b in first:
            uf.union(i, first[b])
        else:
            first[b] = i

    print("Grouping: near-identical pictures (perceptual hash)...")
    for lab in sorted(df['label'].unique()):
        idx = np.where((df['label'].values == lab) & ok)[0]
        n = len(idx)
        if n < 2:
            continue
        Hc = H[idx]
        flat = Hc.reshape(n * 6, 32)
        q = Hc[:, 0, :]
        for s in range(0, n, 50):
            x = np.bitwise_xor(q[s:s + 50, None, :], flat[None, :, :])
            d = POP[x].sum(axis=2, dtype=np.int32)
            qi, rj = np.nonzero(d <= args.thr)
            for a, b in zip(qi, rj // 6):
                ia, ib = idx[s + a], idx[b]
                if ia != ib and uf.find(ia) != uf.find(ib):
                    uf.union(ia, ib)
        print(f"  {lab} done")

    df['gid'] = [uf.find(i) for i in range(N)]
    splits_in_group = df.groupby('gid')['split'].agg(lambda s: frozenset(s))
    df['clean_group'] = df['gid'].map(lambda g: splits_in_group[g] == frozenset(['test']))

    test = df[df['split'] == 'test'].copy()
    kept = test[test['clean_group']].sort_values('filepath').drop_duplicates('gid')

    print(f"\nOriginal test images:                         {len(test)}")
    print(f"  ...with a duplicate in train/val (removed): {int((~test['clean_group']).sum())}")
    print(f"  ...clean but duplicated inside test:        {int(test['clean_group'].sum() - len(kept))}")
    print(f"Clean, de-duplicated test images kept:        {len(kept)}")
    tab = pd.concat([test['label'].value_counts().rename('original_test'),
                     kept['label'].value_counts().rename('clean_test')], axis=1).fillna(0).astype(int)
    print("\nPer class:")
    print(tab.to_string())
    kept[['filepath', 'label']].to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    small = tab[tab['clean_test'] < 30]
    if len(small):
        print("NOTE: classes with fewer than 30 clean test images (per-class numbers will be noisy):",
              list(small.index))


if __name__ == '__main__':
    sys.exit(main())