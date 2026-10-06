"""
hash_check.py  --  independent duplicate / contamination check (does NOT use filenames)

What it does
  1) LEAKAGE: for every TEST image in splits.csv, finds the most similar TRAIN image
     (also trying flipped / rotated versions of the train images) using a 256-bit
     difference hash, and reports how many test images have a near-duplicate in train.
  2) CONTAMINATION (optional): checks whether images from an external dataset
     (e.g. PlantVillage) also appear in your Bangladeshi dataset.

Run from the project folder (PowerShell), using the full tf-gpu python path:

  & "C:\\Users\\HPZ4-03-Adm01\\miniconda3\\envs\\tf-gpu\\python.exe" hash_check.py --splits splits.csv --external "C:\\path\\to\\PlantVillage"

Leave out --external to run only the leakage check.
If Pillow is missing:  python -m pip install pillow
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

EXT = ('.jpg', '.jpeg', '.png')
T = getattr(Image, 'Transpose', Image)
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def dhash(img, size=16):
    """256-bit difference hash returned as 32 uint8 bytes."""
    g = img.convert('L').resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits)


def variants(img):
    return [img,
            img.transpose(T.FLIP_LEFT_RIGHT),
            img.transpose(T.FLIP_TOP_BOTTOM),
            img.transpose(T.ROTATE_90),
            img.transpose(T.ROTATE_180),
            img.transpose(T.ROTATE_270)]


def compute_hashes(paths, all_variants=True, label=''):
    V = 6 if all_variants else 1
    H = np.zeros((len(paths), V, 32), dtype=np.uint8)
    ok = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im.load()
                vs = variants(im) if all_variants else [im]
                for k, v in enumerate(vs):
                    H[i, k] = dhash(v)
            ok[i] = True
        except Exception as e:  # corrupt / unreadable file
            print(f"  could not read {p}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  {label} hashed {i + 1}/{len(paths)}")
    return H, ok


def min_dist(query, ref_variants, chunk=25):
    """query (Q,32); ref_variants (R,V,32). Returns best distance and best ref index per query."""
    Q = query.shape[0]
    R, V, B = ref_variants.shape
    ref_flat = ref_variants.reshape(R * V, B)
    best = np.full(Q, 9999, dtype=np.int32)
    best_idx = np.full(Q, -1, dtype=np.int64)
    for s in range(0, Q, chunk):
        q = query[s:s + chunk]
        x = np.bitwise_xor(q[:, None, :], ref_flat[None, :, :])
        d = POP[x].sum(axis=2, dtype=np.int32)
        j = d.argmin(axis=1)
        best[s:s + chunk] = d[np.arange(len(q)), j]
        best_idx[s:s + chunk] = j // V
        if (s // chunk) % 40 == 0:
            print(f"  compared {min(s + chunk, Q)}/{Q}")
    return best, best_idx


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='splits.csv')
    ap.add_argument('--external', default=None, help='folder of external (e.g. PlantVillage) images')
    ap.add_argument('--cache', default='hash_cache.npz')
    args = ap.parse_args()

    df = pd.read_csv(args.splits)
    paths = df['filepath'].tolist()

    # ---- hash the main dataset (cached) ----
    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        if list(z['paths']) == paths:
            H, ok = z['H'], z['ok']
            print(f"Loaded cached hashes from {args.cache}")
        else:
            H = None
    else:
        H = None
    if H is None:
        print(f"Hashing {len(paths)} dataset images (several minutes)...")
        H, ok = compute_hashes(paths, True, 'dataset')
        np.savez(args.cache, paths=np.array(paths, dtype=object), H=H, ok=ok)
    print(f"Readable images: {ok.sum()} of {len(ok)}")

    rows = []
    # ---- 1) LEAKAGE: test vs train ----
    tr = (df['split'] == 'train').values & ok
    te = (df['split'] == 'test').values & ok
    print(f"\n[1] Leakage check: {te.sum()} test images vs {tr.sum()} train images")
    d, idx = min_dist(H[te][:, 0, :], H[tr])
    te_df = df[te].reset_index(drop=True)
    tr_df = df[tr].reset_index(drop=True)
    same_label = (te_df['label'].values == tr_df['label'].values[idx])
    print("\nTest images with a near-duplicate in TRAIN (Hamming distance out of 256 bits):")
    print("  threshold | any class | same class (the real worry)")
    for thr in (0, 5, 10, 20):
        n_any = int((d <= thr).sum())
        n_same = int(((d <= thr) & same_label).sum())
        print(f"  <= {thr:>2}     | {n_any:>5} ({100 * n_any / len(d):.1f}%) | {n_same:>5} ({100 * n_same / len(d):.1f}%)")
    out = pd.DataFrame({'test_file': te_df['filepath'], 'test_label': te_df['label'],
                        'train_match': tr_df['filepath'].values[idx],
                        'train_label': tr_df['label'].values[idx], 'distance': d})
    out.sort_values('distance').to_csv('hash_check_leakage_matches.csv', index=False)
    print("Closest matches saved to hash_check_leakage_matches.csv (open the top rows and LOOK at them).")

    # ---- 2) CONTAMINATION: external dataset vs everything ----
    if args.external:
        ext_paths = [os.path.join(r, f) for r, _, fs in os.walk(args.external)
                     for f in fs if f.lower().endswith(EXT)]
        print(f"\n[2] Contamination check: {len(ext_paths)} external images vs {ok.sum()} dataset images")
        if not ext_paths:
            print("  no images found in that folder - check the path")
            return
        He, oke = compute_hashes(ext_paths, False, 'external')
        ext_ok = np.where(oke)[0]
        all_ok = np.where(ok)[0]
        d2, idx2 = min_dist(He[ext_ok][:, 0, :], H[all_ok])
        match_paths = np.array(paths, dtype=object)[all_ok][idx2]
        match_lab = df['label'].values[all_ok][idx2]
        print("\nExternal images that look like a copy of a dataset image:")
        for thr in (0, 5, 10, 20):
            n = int((d2 <= thr).sum())
            print(f"  <= {thr:>2}: {n:>5} of {len(d2)} ({100 * n / len(d2):.1f}%)")
        close = d2 <= 10
        if close.any():
            print("\nWhich dataset classes those matches fall in (distance <= 10):")
            print(pd.Series(match_lab[close]).value_counts().to_string())
        # exact byte-identical files
        print("\nChecking exact (MD5) duplicates...")
        ext_md5 = {md5(ext_paths[i]): ext_paths[i] for i in ext_ok}
        exact = sum(1 for p in paths if os.path.exists(p) and md5(p) in ext_md5)
        print(f"  byte-identical files shared between the two datasets: {exact}")
        pd.DataFrame({'external_file': np.array(ext_paths, dtype=object)[ext_ok],
                      'dataset_match': match_paths, 'dataset_label': match_lab,
                      'distance': d2}).sort_values('distance').to_csv(
            'hash_check_external_matches.csv', index=False)
        print("Saved hash_check_external_matches.csv")

    print("\nDone. Send me the printed summary (and tell me what the top rows of the CSVs look like).")


if __name__ == '__main__':
    sys.exit(main())