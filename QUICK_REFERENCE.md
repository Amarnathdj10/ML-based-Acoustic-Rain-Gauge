# Quick Reference Cheat Sheet

## Your Workflow in 3 Steps

```
Step 1: Attach Datasets
  - Add Data → attach 4–5 aligned pickle files (e.g., september_2024_aligned_dataset.pkl)
  - KEEP master_df_clean.parquet attached throughout

Step 2: Run Notebook
  - Run all cells in classifier_training_batch.ipynb top to bottom
  - Extraction auto-detects which datasets are attached

Step 3: Check Output
  - Run "Processing Status" cell (cell 13)
  - See how many windows extracted, how many have rain
  - Look for errors in extraction loop (watch for "skipped windows")

Step 4: Combine (At the end)
  - Run the "Combine all monthly parquet files" cell (cell 16)
  - Creates features_all_combined.parquet with everything extracted so far
```

---

## Files You Get

| File | Size | Rows | What It Is |
|------|------|------|-----------|
| `features_september_2024.parquet` | ~50 MB | 266 | Features for Sept 2024 (266 windows × ~135 columns) |
| `features_december_2024.parquet` | ~1.5 GB | 7,946 | Features for Dec 2024 |
| ... | ... | ... | ... (one per month) |
| `features_all_combined.parquet` | ~50 MB (total) | 28,657 | All months combined (your final training dataset) |

---

## What Changed from Your Original Notebook

| Aspect | Original | New | Why |
|--------|----------|-----|-----|
| **Scope** | Single month hardcoded | Loops all attached datasets | Batch processing |
| **Features per clip** | ~46 | ~114 | Better signal for skewed rainfall |
| **Aggregation** | Mean only across clips | Mean + Std across clips | Captures within-window variability |
| **Output format** | One manual `.parquet` | Auto-named `features_<month>.parquet` | Avoids naming confusion |
| **Combine step** | Manual | Auto-combine from all monthly files | Safe incremental processing |

---

## Key Commands

### Check what's extracted
```python
import glob
import pandas as pd

files = glob.glob("/kaggle/working/features_*.parquet")
files = [f for f in files if "combined" not in f]

for f in sorted(files):
    df = pd.read_parquet(f)
    print(f"{f}: {len(df)} rows, {df['rain'].sum()} rain windows")
```

### Load combined features for training
```python
feature_df = pd.read_parquet("/kaggle/working/features_all_combined.parquet")
X = feature_df.drop(columns=["rainfall_mm", "timestamp", "rain", "wav_count", "n_clips_used"])
y = feature_df["rain"]
```

### Train XGBoost classifier
```python
import xgboost as xgb

# Time-based split (not random!)
split_date = feature_df["timestamp"].quantile(0.7)
train_mask = feature_df["timestamp"] <= split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Train
model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=10)

# Evaluate
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
```

---

## Batch Processing Schedule (Example)

```
Week 1: Batch 1
  ├─ Attach: Sept 2024, Dec 2024, Jan 2025, May 2024, May 2025
  ├─ Run notebook → 5 monthly files + combined
  └─ Check: ~12,000 windows extracted

Week 2: Batch 2
  ├─ Swap data: Add June 2025, Aug 2025, Oct 2025 (remove others)
  ├─ Run notebook → 3 new monthly files + combined (rebuilt from 8)
  └─ Check: ~20,000 windows total

Week 3: Batch 3
  ├─ Swap data: Add remaining months (Feb-Mar 2026, June 2026, etc.)
  ├─ Run notebook → final monthly files + combined (all 16 months)
  └─ Check: ~28,600 windows total

Week 4: Model Training
  ├─ Load features_all_combined.parquet
  ├─ Train binary classifier (rain vs. no-rain)
  ├─ Train intensity regression (on rain-only samples)
  └─ Evaluate with time-based cross-validation
```

---

## Red Flags (What to Watch For)

| Issue | Sign | Fix |
|-------|------|-----|
| **Extraction failed** | "Processing Status" shows 0 rows for a month | Check that pickle file is in Add Data |
| **Incomplete extraction** | Feature file has way fewer rows than label file | Look for exceptions in extraction loop output |
| **Combine doesn't include all months** | `features_all_combined` row count is suspiciously low | Delete old combined file, re-run combine cell |
| **NaN features** | `.dropna()` removes many rows from combined file | Check clip quality, especially for short/corrupt .wav files |
| **Temporal leakage in validation** | Model seems perfect on test set (>99% accuracy) | Use time-based splits, not random shuffling |

---

## Feature List (TL;DR)

Your ~130 features per window include:

- **Time domain**: zero-crossing rate, RMS energy (mean/std/max)
- **Spectral shape**: centroid, bandwidth, rolloff, **flatness** ⭐, **contrast (7 bands)** ⭐
- **Pitch**: **chroma** ⭐
- **Timbre**: MFCCs (13 coefficients) + **delta-MFCCs** ⭐
- **Metadata**: rainfall_mm, timestamp, wav_count, rain (binary), n_clips_used

⭐ = Added in this version (not in your original notebook)

---

## Final Checklist

Before each batch run:

- [ ] `master_df_clean.parquet` is attached (don't remove it between batches!)
- [ ] 4–5 new pickle files are attached (or swapped from previous batch)
- [ ] Notebook is `classifier_training_batch.ipynb` (not the original single-month version)
- [ ] Plan to run all cells top-to-bottom (don't skip steps)

After each batch run:

- [ ] Check "Processing Status" cell output for any missing months or 0-row files
- [ ] Download monthly `.parquet` files (or keep in `/kaggle/working/`)
- [ ] Note the total row count in combined file (should be monotonically increasing)

---

## One-Liner: Load & Train

```python
import xgboost as xgb
import pandas as pd

df = pd.read_parquet("/kaggle/working/features_all_combined.parquet").dropna()
X = df[[c for c in df.columns if c not in ["rainfall_mm", "timestamp", "rain", "wav_count", "n_clips_used"]]]
y = df["rain"]

split_idx = int(0.7 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=0)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
```

---

## Emergency: I Messed Something Up

### Accidentally deleted a monthly feature file

You need to re-extract that month:
1. Attach only that month's pickle file + `master_df_clean.parquet`
2. Run cells 1–11 (skip the combine step)
3. It will only extract that one month and overwrite the `.parquet` you deleted

### Combined file has duplicates or is corrupted

1. Delete `/kaggle/working/features_all_combined.parquet`
2. Run only the combine cell (cell 16)
3. It rebuilds from scratch from all monthly files

### Extraction loop ran but wrote 0-row files

1. Check the output for exceptions
2. Verify the pickle files are actually in `/kaggle/input/`
3. Verify `master_df_clean.parquet` contains that month (check `source_pickle` values)
4. Re-attach and re-run

---

## Performance Expectations

**Extraction time**: 
- ~5 min for a small month (Sept 2024: 266 windows)
- ~1 hour for a large month (May 2024: 7,946 windows)
- **Total for all 16 months: ~4–6 hours** on Kaggle (CPU-dependent)

**Output file size**:
- Each monthly file: ~30–100 MB (depending on window count)
- Combined file: ~400 MB (all 28,657 windows × ~135 float32 columns)

**Training time** (XGBoost binary classifier):
- With `features_all_combined.parquet`: **5–15 minutes** on Kaggle CPU
- With GPU: <2 minutes

---

Good luck with your batches! 🌧️
