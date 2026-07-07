# Batch Feature Extraction Workflow

## Overview

Your `classifier_training_batch.ipynb` notebook supports **safe incremental processing** across multiple Kaggle runs. You can extract features for 4–5 datasets, then swap them out and extract more, without ever losing or overwriting previous work.

---

## How It Works

### Key fact: `/kaggle/working/` persists across notebook runs

When you run a Kaggle notebook, the `/kaggle/working/` directory is preserved between runs. This means:
- **Batch 1**: Extract A, B, C → writes `features_A.parquet`, `features_B.parquet`, `features_C.parquet`
- **Batch 2**: Extract D, E → writes `features_D.parquet`, `features_E.parquet`
- **Run the combine cell again** → it reads *all* 5 files from disk and creates `features_all_combined.parquet` with A–E

Each monthly feature file is written once and never overwritten. The combined file is always rebuilt from scratch by reading all monthly files on disk.

---

## Step-by-Step Workflow

### Batch 1: Initial datasets (e.g., 4–5 months)

1. **Attach datasets** to your Kaggle notebook:
   - Go to your notebook's **Add Data** panel
   - Add the aligned pickle files for months A, B, C, D (e.g., `september_2024_aligned_dataset.pkl`, `december_2024_aligned_dataset.pkl`, etc.)
   - Ensure `master_df_clean.parquet` is also attached (the audit report output)

2. **Run all cells** in `classifier_training_batch.ipynb` top to bottom
   - The extraction loop will auto-detect which datasets are present and process them
   - For each month, it writes one `features_<month_name>.parquet` to `/kaggle/working/`
   - The **Processing Status** cell shows you what's been extracted so far
   - The **combine** cell reads all monthly files and writes `features_all_combined.parquet`

3. **Output files saved** in `/kaggle/working/`:
   - `features_september_2024.parquet` (28 features × ~266 samples)
   - `features_december_2024.parquet`
   - `features_january_2025.parquet`
   - ... etc.
   - `features_all_combined.parquet` (combine of all above)

---

### Batch 2: New datasets (remove old ones, add new ones)

1. **Update notebook's attached datasets**:
   - Remove datasets A, B, C, D from the **Add Data** panel
   - Add new datasets E, F, G, H (or however many you're ready to process next)
   - Keep `master_df_clean.parquet` attached (don't remove it)

2. **Run all cells again**
   - The extraction loop auto-detects the *new* datasets and processes them
   - Writes `features_E.parquet`, `features_F.parquet`, `features_G.parquet`, `features_H.parquet`
   - The combine cell reads all 8 monthly files (A–H) from `/kaggle/working/` and rebuilds `features_all_combined.parquet`
   - **Old files A–D still exist on disk** (they're in `/kaggle/working/`, not in the input), so they're automatically included

3. **Output files saved** in `/kaggle/working/`:
   - All previous files (A–D) still there
   - Plus new files (E–H)
   - `features_all_combined.parquet` now combines A–H

---

### Batch 3, 4, ... repeat as needed

Just keep swapping datasets and re-running the notebook. The combined file will always include everything extracted so far.

---

## Troubleshooting

### I want to check what's been extracted so far

Run the **Processing Status** cell (cell 13 in the notebook). It prints:
```
✓ 8 monthly feature files on disk:
  september_2024       266 windows (64.3% rain)
  december_2024      7946 windows (0.4% rain)
  ...
  
Total: 28657 windows across all months
```

### The combine cell is reading old files I don't want

You can manually delete old `.parquet` files from `/kaggle/working/`:
```python
import os
os.remove("/kaggle/working/features_september_2024.parquet")
```
Then re-run the combine cell. It will only include the remaining files.

### Can I append to the existing combined file instead of rebuilding it?

Yes, if you prefer. Modify the last cell to:
```python
new_feature_files = [f for f in all_feature_files if f not in already_combined]
new_df = pd.concat([pd.read_parquet(f) for f in new_feature_files], ignore_index=True)
combined_df = pd.read_parquet(combined_out_path)  # Load existing
combined_df = pd.concat([combined_df, new_df], ignore_index=True)
combined_df.to_parquet(combined_out_path, index=False)
```
But rebuilding from scratch (the default) is simpler and idempotent — no bookkeeping needed.

---

## What to Do After All Batches Are Done

Once you've extracted features for all datasets you plan to use, you have two options:

### Option A: Continue with training in the same notebook
- Add new cells below the combine step to:
  - Load `features_all_combined.parquet`
  - Apply any final preprocessing (scaling, stratification, etc.)
  - Split into train/val/test (use time-based splits!)
  - Train your XGBoost binary classifier (rain/no-rain)

### Option B: Download and train locally
- Download the monthly `.parquet` files individually (or the combined one)
- Train locally with full control over hyperparameters, logging, etc.

---

## Feature Columns in Output

Each `features_<month>.parquet` has:

**Per-clip features aggregated** (mean + std across clips in the 3-minute window):
- `rms_mean_mean`, `rms_mean_std`, `rms_std_mean`, `rms_std_std`, `rms_max_mean`, `rms_max_std`
- `centroid_mean_mean`, `centroid_mean_std`, `centroid_std_mean`, `centroid_std_std`
- `bandwidth_mean_mean`, `bandwidth_mean_std`
- `rolloff_mean_mean`, `rolloff_mean_std`
- `flatness_mean_mean`, `flatness_mean_std`
- `contrast_1_mean_mean` through `contrast_7_mean_std` (7 spectral bands)
- `chroma_mean_mean`, `chroma_mean_std`
- `mfcc_1_mean_mean` through `mfcc_13_std_std` (13 MFCCs × 2 aggregations × 2 = 52 features)
- `mfcc_delta_1_mean_mean` through `mfcc_delta_13_mean_std` (13 delta-MFCCs × 2 = 26 features)

**Metadata** (one value per window, not aggregated):
- `rainfall_mm` (target)
- `timestamp`
- `wav_count` (number of clips in this window)
- `rain` (binary: 1 if rainfall_mm > 0, else 0)
- `n_clips_used` (how many clips successfully processed for this window)

**Total: ~130 features per window** (vs. ~46 in the original single-month notebook)

---

## Quick Reference: Run Order

Every Kaggle run, in this order:

1. ✅ Attach datasets (change only the input datasets, keep `master_df_clean.parquet`)
2. ✅ Cell 0–11: Feature extraction (auto-loops over whatever is attached)
3. ✅ Cell 13: Check status (optional, but recommended — confirms extraction worked)
4. ✅ Cell 16: Combine (reads all monthly files, writes `features_all_combined.parquet`)
5. 🔄 Repeat for next batch

Done!
