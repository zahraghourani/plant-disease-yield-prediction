"""
check_duplicate_leakage.py

Checks whether augmented duplicates of the same source photo (e.g.
"RS_Rust 2743.JPG", "RS_Rust 2743_flipLR.JPG", "RS_Rust 2743_flipLR(1).JPG")
end up split across train/val/test. If so, the model may have seen a
near-identical twin of a "held-out" test image during training, which
would inflate test accuracy independent of any split-code correctness.

Run with:
    python check_duplicate_leakage.py
"""
import re
import pandas as pd

SPLITS_CSV = './splits.csv'

def get_base_id(filepath):
    """
    Strip augmentation suffixes to recover a canonical 'base photo id'.
    """
    fname = filepath.split('\\')[-1].split('/')[-1]
    fname_noext = re.sub(r'\.(jpg|jpeg|png)$', '', fname, flags=re.IGNORECASE)

    # "image (N)" is NOT a duplicate marker in this dataset — these are
    # distinct, unique photos with generic sequential naming, not
    # augmented copies of a shared source. Keep each one unique.
    if re.match(r'^image\s*\(\d+\)$', fname_noext, flags=re.IGNORECASE):
        return fname_noext.strip()

    # Otherwise, these suffixes DO indicate a real augmented duplicate.
    stripped = fname_noext
    stripped = re.sub(r'\(\d+\)$', '', stripped)
    stripped = re.sub(r'_?(flipLR|flipTB|\d{2,3}deg)', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'_new\w*', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'\s*copy\s*\d*$', '', stripped, flags=re.IGNORECASE)
    return stripped.strip()


def main():
    df = pd.read_csv(SPLITS_CSV)
    # splits.csv already has the correct, class-namespaced base_id
    # written by data_preprocessing.py — don't recompute it here.
    # (df['base_id'] already exists as a column from the CSV.)

    print(f"Total files: {len(df)}")
    print(f"Unique base ids: {df['base_id'].nunique()}")
    print(f"=> average copies per base photo: {len(df) / df['base_id'].nunique():.2f}\n")

    # For each base_id, which splits does it appear in?
    split_sets = df.groupby('base_id')['split'].apply(lambda s: frozenset(s))
    leaked = split_sets[split_sets.apply(lambda s: len(s) > 1)]

    print(f"Base photos with copies spread across MULTIPLE splits: {len(leaked)}")
    print(f"  (out of {df['base_id'].nunique()} total base photos)")
    print(f"  => {100 * len(leaked) / df['base_id'].nunique():.1f}% of source photos are split-leaked\n")

    # How many actual test images are affected (have a sibling in train)?
    test_base_ids = set(df[df['split'] == 'test']['base_id'])
    train_base_ids = set(df[df['split'] == 'train']['base_id'])
    overlap = test_base_ids & train_base_ids

    test_images_affected = df[(df['split'] == 'test') & (df['base_id'].isin(overlap))]
    print(f"Test base photos that ALSO have a copy in train: {len(overlap)}")
    print(f"Actual test IMAGE FILES affected: {len(test_images_affected)} "
          f"out of {len(df[df['split']=='test'])} total test images "
          f"({100*len(test_images_affected)/len(df[df['split']=='test']):.1f}%)")

    if len(overlap) > 0:
        print("\nExample leaked base photos:")
        for bid in list(overlap)[:5]:
            rows = df[df['base_id'] == bid][['filepath', 'split']]
            print(f"\n  base_id: {bid}")
            print(rows.to_string(index=False))


if __name__ == "__main__":
    main()