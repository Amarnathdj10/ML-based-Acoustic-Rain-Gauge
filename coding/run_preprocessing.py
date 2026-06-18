"""
run_preprocessing.py — Batch preprocessing runner for all Group A datasets
===========================================================================
Edit the DATASETS list below to match where you've downloaded each
Kaggle dataset. Then run:

    python run_preprocessing.py

This will produce one .npz per month in OUTPUT_DIR/group_a/
then merge them into train/val/test splits in OUTPUT_DIR/master/

NOTE: Download datasets from Kaggle first:
    kaggle datasets download -d amarnathdj/dec-2023-rainfall-data
    kaggle datasets download -d amarnathdj/nov-2023-rainfall-data
    ... etc.
"""

import subprocess
import sys
import os

# ─── CONFIGURE THESE PATHS ────────────────────────────────────────────────────
# Base directory where you extract each Kaggle dataset
BASE_KAGGLE = r"D:\ARG_dataset"

# Output directory for preprocessed .npz files
OUTPUT_DIR = r"D:\ARG_dataset\preprocessed"

# Path to dl_preprocess.py
SCRIPT = os.path.join(os.path.dirname(__file__), "dl_preprocess.py")

# ─── GROUP A DATASETS ─────────────────────────────────────────────────────────
# Each entry: (label, wav_dir, csv_path)
# Edit wav_dir and csv_path to match your actual extracted folder structure.
DATASETS_GROUP_A = [
    (
        "dec_2023",
        rf"{BASE_KAGGLE}\dec_2023\audio",
        rf"{BASE_KAGGLE}\dec_2023\mech_data.csv",
    ),
    (
        "nov_2023",
        rf"{BASE_KAGGLE}\nov_2023\audio",
        rf"{BASE_KAGGLE}\nov_2023\mech_data.csv",
    ),
    (
        "jan_2024",
        rf"{BASE_KAGGLE}\jan_2024\audio",
        rf"{BASE_KAGGLE}\jan_2024\mech_data.csv",
    ),
    (
        "apr_2024",
        rf"{BASE_KAGGLE}\apr_2024\audio",
        rf"{BASE_KAGGLE}\apr_2024\mech_data.csv",
    ),
    (
        "may_2024",
        rf"{BASE_KAGGLE}\may_2024\audio",
        rf"{BASE_KAGGLE}\may_2024\mech_data.csv",
    ),
    (
        "jul_2024",
        rf"{BASE_KAGGLE}\jul_2024\audio",
        rf"{BASE_KAGGLE}\jul_2024\mech_data.csv",
    ),
    (
        "sep_2024",
        rf"{BASE_KAGGLE}\sep_2024\audio",
        rf"{BASE_KAGGLE}\sep_2024\mech_data.csv",
    ),
]

# ─── OPTIONAL: Group B+C paired datasets ─────────────────────────────────────
DATASETS_GROUP_BC = [
    (
        "dec_2024",
        rf"{BASE_KAGGLE}\dec_2024_audio\audio",
        rf"{BASE_KAGGLE}\dec_2024_mech\mech_data.csv",
    ),
    (
        "oct_nov_2024",
        rf"{BASE_KAGGLE}\nov_2024_audio\audio",
        rf"{BASE_KAGGLE}\oct_nov_2024_mech\mech_data.csv",
    ),
]
# ─────────────────────────────────────────────────────────────────────────────


def run_preprocess(label, wav_dir, csv_path, group="group_a"):
    """Run dl_preprocess.py preprocess for one dataset."""
    out_path = os.path.join(OUTPUT_DIR, group, f"{label}.npz")

    if os.path.exists(out_path):
        print(f"  [SKIP] {label}.npz already exists. Delete to reprocess.")
        return True

    if not os.path.isdir(wav_dir):
        print(f"  [WARN] WAV dir not found: {wav_dir}")
        print(f"         Skipping {label}. Download and extract the dataset first.")
        return False

    if not os.path.isfile(csv_path):
        print(f"  [WARN] CSV not found: {csv_path}")
        print(f"         Skipping {label}. Check csv_path in run_preprocessing.py.")
        return False

    cmd = [
        sys.executable, SCRIPT, "preprocess",
        "--wav_dir",  wav_dir,
        "--csv_path", csv_path,
        "--output",   out_path,
        "--label",    label,
    ]
    print(f"\n[RUNNING] {label}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_merge(npz_dir, out_dir):
    """Run dl_preprocess.py merge."""
    cmd = [
        sys.executable, SCRIPT, "merge",
        "--npz_dir",    npz_dir,
        "--out_dir",    out_dir,
        "--train_frac", "0.70",
        "--val_frac",   "0.15",
    ]
    print(f"\n[MERGING] {npz_dir} → {out_dir}")
    subprocess.run(cmd)


if __name__ == "__main__":
    print("=" * 60)
    print("  Acoustic Rain Gauge — DL Preprocessing Batch Runner")
    print("=" * 60)

    # ── Step 1: Preprocess Group A (labeled, immediately usable) ──────────
    print("\n[STEP 1] Preprocessing Group A datasets (7 months)...")
    group_a_dir = os.path.join(OUTPUT_DIR, "group_a")
    for label, wav_dir, csv_path in DATASETS_GROUP_A:
        run_preprocess(label, wav_dir, csv_path, group="group_a")

    # ── Step 2: Preprocess Group B+C (paired unlabeled + mech CSVs) ───────
    print("\n[STEP 2] Preprocessing Group B+C (paired) datasets...")
    for label, wav_dir, csv_path in DATASETS_GROUP_BC:
        run_preprocess(label, wav_dir, csv_path, group="group_bc")

    # ── Step 3: Merge Group A into master train/val/test splits ───────────
    print("\n[STEP 3] Merging Group A into master splits...")
    run_merge(group_a_dir, os.path.join(OUTPUT_DIR, "master"))

    print("\n" + "=" * 60)
    print("  Done! Preprocessed data is in:")
    print(f"    {OUTPUT_DIR}\\group_a\\      — per-month .npz")
    print(f"    {OUTPUT_DIR}\\master\\       — train/val/test.npz + norm_stats.json")
    print("=" * 60)
