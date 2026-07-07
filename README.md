# Batch Acoustic Feature Extraction — Complete Setup

You now have everything needed to incrementally extract acoustic features from 4–5 datasets at a time, accumulate them safely, and build your full training dataset.

---

## 📋 Files Included

### 1. **`classifier_training_batch.ipynb`** ⭐ MAIN NOTEBOOK

The refactored extraction pipeline. Key features:
- **Loops automatically** over every dataset attached to your Kaggle notebook
- **Expands feature set** from ~46 to ~130 features per audio clip (richer signal for a skewed target)
- **Writes one `features_<month_name>.parquet` per dataset** to `/kaggle/working/`
- **Combines all monthly files** into `features_all_combined.parquet` at the end
- **Safe for incremental runs**: Re-attaching different datasets in subsequent runs doesn't overwrite previous work

**How to use**:
1. Attach 4–5 aligned pickle files + `master_df_clean.parquet` to your Kaggle notebook
2. Run all cells top to bottom
3. Check the **Processing Status** cell (cell 13) to see what's been extracted
4. Download outputs from `/kaggle/working/` or re-run in a new notebook

### 2. **`BATCH_WORKFLOW.md`** ⭐ WORKFLOW GUIDE

Step-by-step instructions for running multiple batches without data loss.

**Key takeaway**: `/kaggle/working/` persists across runs. You can swap out input datasets, re-run the notebook, and the combine cell automatically includes everything extracted so far.

**Read this if**: You're unsure how to structure your batches or worried about overwriting files.

### 3. **`FEATURE_ENGINEERING_NOTES.md`** ⭐ FEATURE DEEP-DIVE

Detailed explanation of every new feature and *why* it matters for your dataset.

**New features added**:
- **Spectral Flatness**: Distinguishes noise-like rain from tonal background sounds
- **Spectral Contrast** (7 bands): Captures frequency-specific peaks (rain leaves different "bumps" in the spectrum than machinery)
- **Chroma**: Detects unwanted tonal content (voices, music, mechanical hum)
- **Delta-MFCCs**: Captures *how fast* the sound's timbre is changing (rain is bursty; background is steady)
- **RMS + Centroid variability**: Distinguishes steady drizzle from bursty downpours

Feature count: ~46 → ~130 per window (no padding; all signal).

**Read this if**: You want to understand what each feature captures or need to justify the choice to your advisors.

### 4. **`DATA_FLOW_DIAGRAM.md`** ⭐ TECHNICAL DEEP-DIVE

Visual explanation of how incremental batching actually works under the hood.

Includes concrete examples:
- Batch 1: Extract A, B, C → creates 3 feature files + 1 combined
- Batch 2: Remove A, B, C; add D, E, F → creates 3 new files, combine reads all 6 (A–F)
- Batch 3, 4, ... repeat

**Read this if**: You want to understand why combining is safe to re-run, or you want to debug a specific batch run.

---

## 🚀 Quick Start

### Step 1: Prepare Your Kaggle Notebook

1. Create a **new Kaggle notebook** (or use an existing one)
2. Upload `classifier_training_batch.ipynb` as a new notebook (or copy its cells into an existing one)
3. Go to **Add Data** and attach:
   - Your first 4–5 aligned pickle files (e.g., `september_2024_aligned_dataset.pkl`, ...)
   - **`master_df_clean.parquet`** (from the audit report output — this is essential!)

### Step 2: Run the Extraction

1. Run all cells from top to bottom
2. The extraction loop auto-detects which datasets are attached and processes only those
3. Output files are written to `/kaggle/working/`

### Step 3: Check Results

Run the **Processing Status** cell to see:
```
✓ 4 monthly feature files on disk:
  september_2024       266 windows (64.3% rain)
  december_2024      7946 windows (0.4% rain)
  january_2025       3024 windows (6.3% rain)
  may_2024            410 windows (91.2% rain)

Total: 11646 windows across all months
```

### Step 4: Download (Optional)

Download the feature files:
- Individual monthly files (e.g., `features_september_2024.parquet`)
- Combined file (e.g., `features_all_combined.parquet`)

Or keep them in Kaggle and continue training in the same notebook.

### Step 5: Next Batch

When ready to extract more datasets:

1. Update **Add Data**: remove old datasets, attach new ones (keep `master_df_clean.parquet`)
2. Run all cells again
3. New monthly files are created; old ones remain in `/kaggle/working/`
4. The combine cell rebuilds `features_all_combined.parquet` with everything extracted so far

---

## 🎯 Next Phase: Model Training

Once you have `features_all_combined.parquet` with all datasets:

### Stage 1: Binary Rain/No-Rain Classifier

```python
# Load combined features
feature_df = pd.read_parquet("features_all_combined.parquet")

# Drop any rows with NaN features
feature_df = feature_df.dropna()

# Create train/test split (TIME-BASED, not random!)
# Use timestamps to ensure no temporal leakage
train_end_date = feature_df["timestamp"].quantile(0.7)
train_df = feature_df[feature_df["timestamp"] <= train_end_date]
test_df = feature_df[feature_df["timestamp"] > train_end_date]

# Train XGBoost
import xgboost as xgb
X_train = train_df.drop(columns=["rainfall_mm", "timestamp", "rain", ...])
y_train = train_df["rain"]

model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
```

### Stage 2: Rainfall Intensity Regression

Once the classifier works, train a regression model on rain-only samples:

```python
# Filter to rain windows only
rain_df = feature_df[feature_df["rain"] == 1].copy()

# Log-transform rainfall (heavy right-skew)
rain_df["log_rainfall"] = np.log(rain_df["rainfall_mm"] + 1e-6)

# Train regression model
reg_model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1)
reg_model.fit(X_rain_train, rain_df["log_rainfall"])

# Predictions on test set
preds_log = reg_model.predict(X_rain_test)
preds_mm = np.exp(preds_log)  # Convert back to millimeters
```

---

## 📊 Expected Output Shape

Each `features_<month>.parquet` contains:

| Column | Type | Notes |
|--------|------|-------|
| `zcr_mean`, `zcr_std` | float | Zero-crossing rate (2) |
| `rms_mean_mean`, `rms_mean_std`, ... | float | RMS energy (6) |
| `centroid_*`, `bandwidth_*`, ... | float | Spectral shape (12) |
| `flatness_*` | float | Noise-likeness (2) ⭐ NEW |
| `contrast_*` | float | Spectral peaks per frequency band (14) ⭐ NEW |
| `chroma_*` | float | Pitch class distribution (2) ⭐ NEW |
| `mfcc_*` | float | Timbre coefficients (52) |
| `mfcc_delta_*` | float | Timbre rate of change (26) ⭐ NEW |
| `rainfall_mm` | float | Ground truth label (target) |
| `timestamp` | datetime | Window timestamp |
| `wav_count` | int | Number of clips in window |
| `rain` | int | Binary label (1 if rainfall_mm > 0, else 0) |
| `n_clips_used` | int | How many clips were successfully processed |

**Total**: ~135 columns, one row per 3-minute rainfall window.

Example shape for September 2024: `(266, 135)`

---

## ⚠️ Important Notes

### 1. Always Attach `master_df_clean.parquet`

The extraction loop uses this to iterate over all rows and check which datasets are available. Without it, the notebook won't know which month each row belongs to, and extraction will fail.

### 2. Use Time-Based Validation (Not Random Splits)

Your data spans 2.5 years across 136 distinct calendar days. A random train/test split will cause **temporal leakage** — the model will train and test on overlapping time periods and will not generalize to new data.

Always:
- Sort by `timestamp`
- Split into train (e.g., first 70% of unique dates) and test (last 30%)
- No randomization in the split

### 3. Stratify by Rainfall Intensity

Since 78% of rain windows are < 0.5 mm, make sure your train/test split preserves this distribution:

```python
# Create rainfall intensity bins
rain_df = feature_df[feature_df["rain"] == 1].copy()
rain_df["rainfall_bin"] = pd.cut(rain_df["rainfall_mm"], 
                                  bins=[0, 0.5, 1, 2, 5, 100],
                                  labels=["<0.5", "0.5-1", "1-2", "2-5", ">5"])

# Stratify by bin when splitting
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    rain_df, test_size=0.3, stratify=rain_df["rainfall_bin"], random_state=42
)
```

### 4. Monitor for NaN Features

If `extract_clip_features()` fails on all clips in a window, that window is skipped. Check:

```python
print(f"Original label rows: {len(master_df)}")
print(f"Extracted feature rows: {len(feature_df)}")
```

If the gap is large (>1%), investigate:

```python
# Which sources have missing features?
missing_sources = [
    s for s in master_df["source_pickle"].unique()
    if s not in feature_df["source_pickle"].unique()
]
print(f"Missing: {missing_sources}")
```

---

## 🔧 Troubleshooting

### Notebook runs but produces no output files

**Check**: Is `master_df_clean.parquet` attached? The loop will silently skip all rows if it can't find the source pickles in the input.

**Solution**: Re-attach your datasets to the **Add Data** panel.

### Some datasets extract successfully, others fail

**Check**: Run the **Processing Status** cell. If a month shows 0 rows, all its clips failed to process.

**Likely cause**: Filepath issues or corrupted .wav files. Check the notebook output for exceptions in the extraction loop.

### `features_all_combined.parquet` is smaller than expected

**Check**: Count rows in all monthly files and compare:

```python
import glob
import pandas as pd

feature_files = glob.glob("/kaggle/working/features_*.parquet")
feature_files = [f for f in feature_files if "combined" not in f]

total = sum([len(pd.read_parquet(f)) for f in feature_files])
combined = len(pd.read_parquet("/kaggle/working/features_all_combined.parquet"))

print(f"Monthly total: {total}")
print(f"Combined: {combined}")
```

If they don't match, some rows may have been dropped. Check for NaN values.

---

## 📚 Citation / References

- **Librosa documentation**: https://librosa.org/ (feature extraction functions)
- **XGBoost guide**: https://xgboost.readthedocs.io/ (training and hyperparameter tuning)
- **Kaggle time-series best practices**: Cross-validate on time, not random splits
- **Your audit report**: `/mnt/project/audit_report_v1.md` (dataset overview and known data quality issues)

---

## 🎓 What to Do Next

1. **Run Batch 1**: Extract your first 4–5 datasets using `classifier_training_batch.ipynb`
2. **Check the output**: Run the Processing Status cell, verify the numbers look right
3. **Repeat Batches**: Follow the workflow in `BATCH_WORKFLOW.md` to process remaining datasets
4. **Train your classifier**: Use `features_all_combined.parquet` to train a binary rain/no-rain XGBoost model
5. **Evaluate**: Time-based cross-validation to avoid leakage
6. **Train Stage 2 (optional)**: Regression on rain-only samples for intensity estimation

Good luck! 🌧️

---

## 📞 Questions?

- **How do I safely process batches?** → See `BATCH_WORKFLOW.md`
- **Why were features added?** → See `FEATURE_ENGINEERING_NOTES.md`
- **How does incremental combining work?** → See `DATA_FLOW_DIAGRAM.md`
- **How do I train models?** → See "Next Phase: Model Training" above
