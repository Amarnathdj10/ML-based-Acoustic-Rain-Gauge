# Data Flow & Incremental Combining

## The Key Insight

**`/kaggle/working/` is persistent across notebook runs.**

When you run a Kaggle notebook, `/kaggle/working/` stays around between execution sessions. The input datasets (`/kaggle/input/`) are temporary and replaced each time you change attachments. But your outputs persist.

---

## Visual: How Batching Works

### Batch 1: Extract from datasets A, B, C (September 2024, December 2024, January 2025)

```
Kaggle Inputs (temporary, changes per run)
│
├── master_df_clean.parquet                [stays throughout all runs]
├── september_2024_aligned_dataset.pkl     [attached for Batch 1]
├── december_2024_aligned_dataset.pkl      [attached for Batch 1]
└── january_2025_aligned_dataset.pkl       [attached for Batch 1]


Extraction Process (master_df_clean.parquet loop):
│
├── Row 1: source_pickle = "september_2024_aligned_dataset.pkl"  ✓ in inputs → extract
├── Row 2: source_pickle = "september_2024_aligned_dataset.pkl"  ✓ in inputs → extract
├── ... (266 rows for Sept)
├── Row 267: source_pickle = "december_2024_aligned_dataset.pkl" ✓ in inputs → extract
├── ... (7946 rows for Dec)
├── Row X: source_pickle = "january_2025_aligned_dataset.pkl"    ✓ in inputs → extract
└── ... (3024 rows for Jan)


Outputs written to /kaggle/working/ (persistent!)
│
├── features_september_2024.parquet   [266 rows × 130 cols]   ← stays after Batch 1
├── features_december_2024.parquet    [7946 rows × 130 cols]  ← stays after Batch 1
├── features_january_2025.parquet     [3024 rows × 130 cols]  ← stays after Batch 1
│
└── Combine cell runs:
    └── reads all `features_*.parquet` from /kaggle/working/
    └── concatenates: 266 + 7946 + 3024 = 11,236 rows
    └── writes: features_all_combined.parquet [11,236 rows × 130 cols]
```

✅ Batch 1 complete. `/kaggle/working/` now contains 4 parquet files.

---

### Batch 2: Swap A, B, C out. Attach D, E, F (May 2024, May 2025, May 2026)

```
Kaggle Inputs (temporary, completely replaced)
│
├── master_df_clean.parquet                [stays]
├── september_2024_aligned_dataset.pkl     [❌ REMOVED]
├── december_2024_aligned_dataset.pkl      [❌ REMOVED]
├── january_2025_aligned_dataset.pkl       [❌ REMOVED]
│
├── may_2024_aligned_dataset.pkl           [✓ NEW for Batch 2]
├── may_2025_aligned_dataset.pkl           [✓ NEW for Batch 2]
└── may_2026_aligned_dataset.pkl           [✓ NEW for Batch 2]


Extraction Process (master_df_clean.parquet loop):
│
  (master_df_clean.parquet still has all 28,657 rows!)
│
├── Row N: source_pickle = "september_2024_aligned_dataset.pkl"  ❌ NOT in inputs → skip
│
├── Row M: source_pickle = "may_2024_aligned_dataset.pkl"        ✓ in inputs → extract
├── ... (410 rows for May 2024)
│
├── Row P: source_pickle = "may_2025_aligned_dataset.pkl"        ✓ in inputs → extract
├── ... (915 rows for May 2025)
│
└── Row Q: source_pickle = "may_2026_aligned_dataset.pkl"        ✓ in inputs → extract
   └── ... (1647 rows for May 2026)


Outputs written to /kaggle/working/
│
├── features_september_2024.parquet   [266 rows]  ← OLD, unaffected
├── features_december_2024.parquet    [7946 rows] ← OLD, unaffected
├── features_january_2025.parquet     [3024 rows] ← OLD, unaffected
│
├── features_may_2024.parquet         [410 rows]  ← NEW
├── features_may_2025.parquet         [915 rows]  ← NEW
├── features_may_2026.parquet         [1647 rows] ← NEW
│
└── Combine cell runs:
    └── reads ALL 6 `features_*.parquet` from /kaggle/working/
        (doesn't care that A-C are no longer in /kaggle/input/)
    └── concatenates: 266 + 7946 + 3024 + 410 + 915 + 1647 = 14,208 rows
    └── overwrites: features_all_combined.parquet [14,208 rows × 130 cols]
```

✅ Batch 2 complete. `/kaggle/working/` now contains 7 parquet files (6 monthly + 1 combined).

---

### Batch 3, 4, ... repeat

Each time you:
1. **Swap the input datasets** (remove old, add new in **Add Data** panel)
2. **Keep `master_df_clean.parquet`** attached (don't remove it!)
3. **Run all cells**
4. The extraction loop auto-detects which new months are available and extracts only those
5. The combine cell reads *all* feature files from `/kaggle/working/` (old + new) and rebuilds the combined file

Result: **monotonically growing combined dataset, no data loss**.

---

## Why This Works: The Magic of `master_df_clean.parquet`

`master_df_clean.parquet` contains every row from every month, indexed by `source_pickle`. When the extraction loop runs:

```python
for source in master_df["source_pickle"].unique():
    df_subset = master_df[master_df["source_pickle"] == source]
    # Try to extract features for this source...
```

If the source's pickle file is **not in the input**, the loop will simply find 0 audio files at the paths listed and skip that source (or write an empty parquet). If the source **is in the input**, extraction proceeds.

So the same script, run on different input datasets, automatically processes only what's available. **You never have to manually edit the notebook.**

---

## Detailed Trace: What Happens When a .wav Path Can't Be Found

From the notebook:

```python
for path in row["wav_files"]:
    try:
        feat = extract_clip_features(path)
        clip_features.append(feat)
    except Exception:
        continue  # Skip this clip, continue to next
```

If a clip path doesn't exist (because the input dataset wasn't attached), the `except` catches it and moves on. At the window level:

```python
if len(clip_features) == 0:
    continue  # Skip this entire window, move to next
```

So if all clips for a window are missing (because the dataset wasn't attached), that window is skipped. This is fine — you'll re-process it in a later batch when the dataset is attached.

---

## Checking What's Been Done So Far

Run the **Processing Status** cell (cell 13) anytime to see:

```python
feature_files = sorted(glob.glob(f"{OUTPUT_DIR}/features_*.parquet"))
feature_files = [f for f in feature_files if 'combined' not in f]

print(f"✓ {len(feature_files)} monthly feature files on disk:")
for f in feature_files:
    df = pd.read_parquet(f)
    month = os.path.basename(f).replace('features_', '').replace('.parquet', '')
    rain_pct = 100 * df['rain'].sum() / len(df)
    print(f"  {month:20s} {len(df):5d} windows ({rain_pct:5.1f}% rain)")
```

Output might look like:

```
✓ 6 monthly feature files on disk:
  december_2024            7946 windows (0.4% rain)
  january_2025             3024 windows (6.3% rain)
  may_2024                  410 windows (91.2% rain)
  may_2025                  915 windows (25.4% rain)
  may_2026                 1647 windows (3.8% rain)
  september_2024            266 windows (64.3% rain)

Total: 14208 windows across all months
```

This is your ground truth of what's been extracted. Compare to what you *expect* to be done, and you'll know exactly where you are in the pipeline.

---

## Combining: Safe to Re-Run Many Times

The combine cell just does:

```python
all_feature_files = sorted(glob.glob(f"{OUTPUT_DIR}/features_*.parquet"))
all_feature_files = [f for f in all_feature_files if "combined" not in f]  # Exclude the combined file itself

combined_df = pd.concat(
    [pd.read_parquet(f) for f in all_feature_files],
    ignore_index=True
)

combined_df.to_parquet(f"{OUTPUT_DIR}/features_all_combined.parquet", index=False)
```

**It's idempotent**: running it twice gives the same result. It doesn't append to an existing file; it reads all monthly files and rebuilds from scratch. So there's no risk of duplicates or corruption.

---

## Example: 4 Full Batches

```
Batch 1 (4 datasets):    Sept, Dec 2024 + Jan, May 2024
  → /kaggle/working/: 4 monthly + 1 combined

Batch 2 (3 datasets):    May 2025 + June 2025 + Aug 2025
  → /kaggle/working/: 7 monthly + 1 combined (rebuilt from 7)

Batch 3 (4 datasets):    Oct 2025 + Jan 2026 + Feb-Mar 2026 + June 2026
  → /kaggle/working/: 11 monthly + 1 combined (rebuilt from 11)

Batch 4 (Remaining):     Any other months not yet done
  → /kaggle/working/: 16 monthly + 1 combined (rebuilt from all)
```

Once Batch 4 is done, you have `features_all_combined.parquet` with all 28,657 rows ready for model training. No data loss, no overwrites.

---

## Edge Case: Manually Cleaning Up Old Files

If you ever want to *exclude* a month (e.g., it had data quality issues), just delete its parquet file before running the combine cell:

```python
import os
os.remove("/kaggle/working/features_september_2024.parquet")
```

Then re-run the combine cell. It will rebuild the combined file without September 2024.

---

## Summary Table

| Action | Effect | Persistent? |
|--------|--------|---|
| Attach new input dataset | Tells the extraction loop which .pkl files to read | Only for this run |
| Extract features from a month | Writes `features_<month>.parquet` to `/kaggle/working/` | Yes ✅ |
| Run combine cell | Reads all `features_*.parquet` files, writes `features_all_combined.parquet` | Yes ✅ |
| Detach old input dataset | Extraction loop skips that month (finds 0 .wav files) | — |
| Run notebook again with new inputs | Old `features_*.parquet` files are still there; new ones are created alongside | Yes ✅ |

---

## One Last Safety Check

Before you start batch processing, verify that `master_df_clean.parquet` covers all the months you plan to extract:

```python
print(master_df["source_pickle"].unique())
```

This should list all 16 months from the audit report. If a month is missing from `master_df_clean`, you'll need to add it to your Kaggle dataset first. But assuming your dataset is up to date, you're good to go.

🎉 You're ready to batch!
