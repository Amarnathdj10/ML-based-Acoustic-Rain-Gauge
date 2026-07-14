# ─────────────────────────────────────────────────────────────
# Cell 1 — Install + Imports
# Enable internet in Kaggle settings BEFORE running this cell
# (downloads CNN14 checkpoint ~300 MB on first run, cached after)
# ─────────────────────────────────────────────────────────────
!pip install panns-inference -q

import os, gc
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import xgboost as xgb
import optuna
import joblib

from panns_inference import AudioTagging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, classification_report,
    roc_auc_score, accuracy_score, precision_score, recall_score
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR  = "/kaggle/working"
MASTER_PATH = "/kaggle/input/datasets/amarnathdj/audit-report-v1/master_df_clean.parquet"
SR_PANNS    = 32000        # CNN14 requires 32 kHz — always resample to this
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = {0: "No Rain", 1: "Rain"}

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# ─────────────────────────────────────────────────────────────
# Cell 2 — Load PANNs CNN14
# AudioSet-pretrained: knows 527 sound classes including Rain,
# Raindrop, Rain on surface, Heavy rain, Drizzle.
# ─────────────────────────────────────────────────────────────
print("Loading CNN14 (downloads checkpoint on first run)...")
at = AudioTagging(checkpoint_path=None, device=DEVICE)
print("CNN14 ready.")


# ─────────────────────────────────────────────────────────────
# Cell 3 — Embedding helpers
# ─────────────────────────────────────────────────────────────
def load_clip(path, sr=SR_PANNS):
    """Load WAV, convert to mono float32, resample to 32 kHz."""
    try:
        y, orig_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)              # stereo → mono
        y = y.astype(np.float32)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        min_len = sr // 5                   # at least 0.2 s
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))
        return y
    except Exception:
        return None


def window_embedding(wav_paths, batch_size=16):
    """
    Given 17-18 WAV paths in one 3-min window:
      1. Load + resample each clip to 32 kHz
      2. Batch through CNN14 → (n_clips, 2048) embeddings
      3. Return concat(mean, std) → (4096,)

    Returns None if no clips are loadable.
    """
    clips = []
    for p in wav_paths:
        y = load_clip(p)
        if y is not None:
            clips.append(y)

    if not clips:
        return None

    # Pad all clips to same length (needed for batching)
    max_len = max(len(c) for c in clips)
    padded  = np.stack([
        np.pad(c, (0, max_len - len(c))) for c in clips
    ])                                      # shape: (n_clips, max_len)

    # Run through CNN14 in batches
    all_embs = []
    for i in range(0, len(padded), batch_size):
        batch = padded[i : i + batch_size]
        with torch.no_grad():
            _, emb = at.inference(batch)    # emb: (B, 2048) numpy array
        all_embs.append(emb)

    emb_all = np.concatenate(all_embs, axis=0)   # (n_clips, 2048)

    # Aggregate across clips: mean captures average acoustic state,
    # std captures variability (intermittent rain vs steady rain vs silence)
    return np.concatenate([
        emb_all.mean(axis=0),               # (2048,)
        emb_all.std(axis=0)                 # (2048,)
    ])                                      # → (4096,) total


# ─────────────────────────────────────────────────────────────
# Cell 4 — Incremental extraction (run once per audio batch)
#
# Same batch workflow as feature extraction:
#   Run 1: attach audio datasets A-D → saves panns_batch_01.parquet
#   Run 2: attach audio datasets E-H → saves panns_batch_02.parquet
#   ...
#
# Change BATCH_TAG each run. Already-done windows are skipped.
# ─────────────────────────────────────────────────────────────
BATCH_TAG = "01"                            # ← INCREMENT each run
BATCH_OUT = f"{OUTPUT_DIR}/panns_batch_{BATCH_TAG}.parquet"

# Collect timestamps already extracted in previous batches
done_files = sorted([
    f for f in os.listdir(OUTPUT_DIR)
    if f.startswith("panns_batch_") and f.endswith(".parquet")
])
done_ts = set()
for f in done_files:
    prev = pd.read_parquet(os.path.join(OUTPUT_DIR, f), columns=["timestamp"])
    done_ts.update(prev["timestamp"].astype(str).tolist())
print(f"Already extracted : {len(done_ts)} windows (from {len(done_files)} batches)")

# Load master labels
master_df = pd.read_parquet(MASTER_PATH)
master_df["timestamp"] = pd.to_datetime(master_df["timestamp"])

# Keep only rows where at least one audio file is on disk right now
def has_audio(wav_list):
    return any(os.path.exists(p) for p in wav_list)

todo = master_df[
    ~master_df["timestamp"].astype(str).isin(done_ts) &
    master_df["wav_files"].apply(has_audio)
].copy()

print(f"Windows to process this run : {len(todo)}")

rows = []
for _, row in tqdm(todo.iterrows(), total=len(todo)):
    existing = [p for p in row["wav_files"] if os.path.exists(p)]
    emb = window_embedding(existing)
    if emb is None:
        continue

    entry = {
        "timestamp"   : row["timestamp"],
        "rainfall_mm" : row["rainfall_mm"],
        "rain"        : int(row["rainfall_mm"] > 0),
        "source"      : row["source_pickle"],
    }
    entry.update({f"p{i}": float(emb[i]) for i in range(4096)})
    rows.append(entry)

    if len(rows) % 500 == 0:
        gc.collect()

if rows:
    batch_df = pd.DataFrame(rows)
    batch_df.to_parquet(BATCH_OUT, index=False)
    print(f"Saved {len(batch_df)} windows → {BATCH_OUT}")
else:
    print("No new windows extracted this run.")


# ─────────────────────────────────────────────────────────────
# Cell 5 — Combine all batch parquets
# Run once after all audio batches are done
# ─────────────────────────────────────────────────────────────
batch_files = sorted([
    os.path.join(OUTPUT_DIR, f)
    for f in os.listdir(OUTPUT_DIR)
    if f.startswith("panns_batch_") and f.endswith(".parquet")
])

print(f"Found {len(batch_files)} batch files:")
for f in batch_files:
    n = len(pd.read_parquet(f, columns=["timestamp"]))
    print(f"  {os.path.basename(f)} : {n} windows")

df_panns = pd.concat(
    [pd.read_parquet(f) for f in batch_files],
    ignore_index=True
)
df_panns["timestamp"] = pd.to_datetime(df_panns["timestamp"])
df_panns = (
    df_panns
    .drop_duplicates(subset="timestamp")
    .sort_values("timestamp")
    .reset_index(drop=True)
)

PANNS_COMBINED = f"{OUTPUT_DIR}/panns_all_combined.parquet"
df_panns.to_parquet(PANNS_COMBINED, index=False)

print(f"\nCombined shape : {df_panns.shape}")
print(f"Date range     : {df_panns['timestamp'].min().date()} → {df_panns['timestamp'].max().date()}")
vc = df_panns["rain"].value_counts().sort_index()
for cls, cnt in vc.items():
    print(f"  {CLASS_NAMES[cls]:12s}: {cnt:6d} ({cnt/len(df_panns)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# Cell 5b — OPTIONAL: merge PANNs (4096) + hand-crafted (112)
#
# PANNs embeddings capture deep AudioSet-trained representations.
# Hand-crafted features (MFCC, ZCR, contrast) capture domain physics.
# Together they outperform either alone.
# ─────────────────────────────────────────────────────────────
COMBINED_FEAT_PATH = f"{OUTPUT_DIR}/features_all_combined.parquet"

if os.path.exists(COMBINED_FEAT_PATH):
    feat_df = pd.read_parquet(COMBINED_FEAT_PATH)
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])

    # Drop label/meta cols from hand-crafted side before merge
    hc_meta = {"timestamp", "rainfall_mm", "wav_count",
                "n_clips_used", "0_mean", "0_std", "rain"}
    hc_cols = [c for c in feat_df.columns if c not in hc_meta]

    df = pd.merge(
        df_panns,
        feat_df[["timestamp"] + hc_cols],
        on="timestamp",
        how="inner"
    )
    print(f"Merged dataset shape   : {df.shape}")
    print(f"  PANNs features       : 4096")
    print(f"  Hand-crafted features: {len(hc_cols)}")
    print(f"  Total features       : {4096 + len(hc_cols)}")
else:
    print("No hand-crafted features found — using PANNs-only.")
    df = df_panns.copy()


# ─────────────────────────────────────────────────────────────
# Cell 6 — Feature prep + scale
# ─────────────────────────────────────────────────────────────
TARGET    = "rain"
META_COLS = {"timestamp", "rainfall_mm", "rain", "source",
             "wav_count", "n_clips_used", "0_mean", "0_std"}
FEAT_COLS = [c for c in df.columns if c not in META_COLS]

df = (
    df.sort_values("timestamp")
      .dropna(subset=FEAT_COLS)
      .reset_index(drop=True)
)

X = df[FEAT_COLS].values
y = df[TARGET].values

n_neg = int((y == 0).sum())
n_pos = int((y == 1).sum())
SPW   = n_neg / n_pos

print(f"Feature columns  : {len(FEAT_COLS)}")
print(f"Total samples    : {len(df)}")
print(f"Date range       : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"\nNo Rain          : {n_neg}  ({n_neg/len(y)*100:.1f}%)")
print(f"Rain             : {n_pos}  ({n_pos/len(y)*100:.1f}%)")
print(f"scale_pos_weight : {SPW:.4f}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_panns.pkl")
print("\nScaler saved → scaler_panns.pkl")


# ─────────────────────────────────────────────────────────────
# Cell 7 — Threshold helper
# ─────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob):
    """Sweep 0.10→0.90, return threshold that maximises Rain F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (y_prob >= t).astype(int)
        f1    = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────
# Cell 8 — Optuna search
# ─────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int  ("n_estimators",     100, 800),
        max_depth        = trial.suggest_int  ("max_depth",          3,   8),
        learning_rate    = trial.suggest_float("learning_rate",   0.005, 0.2, log=True),
        subsample        = trial.suggest_float("subsample",        0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int  ("min_child_weight",   1,  20),
        gamma            = trial.suggest_float("gamma",            0.0,  1.0),
        reg_alpha        = trial.suggest_float("reg_alpha",        0.0,  5.0),
        reg_lambda       = trial.suggest_float("reg_lambda",       0.5,  5.0),
        scale_pos_weight = SPW,
        eval_metric      = "aucpr",
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    fold_f1s = []
    for tr_idx, val_idx in tscv.split(X_scaled):
        m = xgb.XGBClassifier(**params)
        m.fit(X_scaled[tr_idx], y[tr_idx], verbose=False)
        y_prob = m.predict_proba(X_scaled[val_idx])[:, 1]
        _, f1  = find_best_threshold(y[val_idx], y_prob)
        fold_f1s.append(f1)
    return np.mean(fold_f1s)

print("Running Optuna (80 trials) on PANNs + XGBoost...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=80, show_progress_bar=True)

best_params = {
    **study.best_params,
    "scale_pos_weight" : SPW,
    "eval_metric"      : "aucpr",
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbosity"        : 0,
}

print(f"\nBest CV Rain-F1 : {study.best_value:.4f}")
print("Best params:")
for k, v in best_params.items():
    print(f"  {k:25s}: {v}")


# ─────────────────────────────────────────────────────────────
# Cell 9 — CV evaluation
# ─────────────────────────────────────────────────────────────
all_probs = np.full(len(y), -1.0)
all_preds = np.full(len(y), -1, dtype=int)
cv_mask   = np.zeros(len(y), dtype=bool)
fold_ts   = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled), start=1):
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_scaled[tr_idx], y[tr_idx], verbose=False)

    y_prob        = model.predict_proba(X_scaled[val_idx])[:, 1]
    best_t, f1_t  = find_best_threshold(y[val_idx], y_prob)
    fold_ts.append(best_t)

    all_probs[val_idx] = y_prob
    all_preds[val_idx] = (y_prob >= best_t).astype(int)
    cv_mask[val_idx]   = True

    ba = balanced_accuracy_score(y[val_idx], (y_prob >= best_t).astype(int))
    print(
        f"Fold {fold} | thr={best_t:.3f} | Rain-F1={f1_t:.4f} | "
        f"BalAcc={ba:.4f} | val={len(val_idx)} | "
        f"rain%={(y[val_idx].mean()*100):.1f}%"
    )

# OOF global threshold
y_cv      = y[cv_mask]
probs_cv  = all_probs[cv_mask]
preds_cv  = all_preds[cv_mask]        # per-fold thresholds
global_t, global_f1 = find_best_threshold(y_cv, probs_cv)
global_preds = (probs_cv >= global_t).astype(int)

print(f"\nMean fold threshold  : {np.mean(fold_ts):.3f}")
print(f"Global OOF threshold : {global_t:.3f}  (Rain-F1={global_f1:.4f})")

print("\n" + "=" * 60)
print("CV REPORT — PANNs embeddings + XGBoost")
print("=" * 60)
print(classification_report(
    y_cv, global_preds,
    target_names=["No Rain", "Rain"],
    digits=4, zero_division=0
))
print(f"Balanced Accuracy : {balanced_accuracy_score(y_cv, global_preds):.4f}")
print(f"ROC-AUC           : {roc_auc_score(y_cv, probs_cv):.4f}")
print(f"Rain Precision    : {precision_score(y_cv, global_preds, zero_division=0):.4f}")
print(f"Rain Recall       : {recall_score(y_cv, global_preds, zero_division=0):.4f}")
print(f"Rain F1           : {f1_score(y_cv, global_preds, zero_division=0):.4f}")

joblib.dump(global_t, f"{OUTPUT_DIR}/threshold_panns.pkl")
print(f"\nThreshold saved → threshold_panns.pkl  ({global_t:.3f})")
print("\nInference usage:")
print("  proba = model.predict_proba(X_scaled)[:, 1]")
print(f"  pred  = (proba >= {global_t:.3f}).astype(int)")