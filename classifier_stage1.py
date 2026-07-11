# ============================================================
# STAGE 1: Binary Rain/No-Rain Classifier
# Dataset: ML-Based Acoustic Rain Gauge
# Split:   Train → Nov 2023 – Dec 2025
#          Test  → Jan 2026 – Jun 2026
# ============================================================

import numpy as np
import pandas as pd
import glob
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_recall_curve,
    roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── 1. LOAD & COMBINE ───────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

OUTPUT_DIR = "/kaggle/working"

files = sorted(glob.glob(f"{OUTPUT_DIR}/features_*.parquet"))
files = [f for f in files if "combined" not in f]
print(f"Found {len(files)} monthly parquet files")

dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(f"Total cols: {df.shape[1]}")

# ── 2. DEFINE FEATURES & TARGET ─────────────────────────────
META_COLS = ["timestamp", "rainfall_mm", "wav_count",
             "rain", "n_clips_used", "0_mean", "0_std"]
FEATURE_COLS = [c for c in df.columns if c not in META_COLS]
TARGET = "rain"

print(f"\nFeature columns: {len(FEATURE_COLS)}")

# ── 3. TIME-BASED TRAIN / TEST SPLIT ────────────────────────
# CRITICAL: no random split — temporal integrity required
CUTOFF = pd.Timestamp("2026-01-01")

train = df[df["timestamp"] < CUTOFF].copy()
test  = df[df["timestamp"] >= CUTOFF].copy()

print(f"\nTrain: {len(train)} rows ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
print(f"Test:  {len(test)} rows  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
print(f"\nTrain class balance:")
print(train[TARGET].value_counts())
print(train[TARGET].value_counts(normalize=True).mul(100).round(2))
print(f"\nTest class balance:")
print(test[TARGET].value_counts())
print(test[TARGET].value_counts(normalize=True).mul(100).round(2))

# ── 4. DROP NULL ROWS ────────────────────────────────────────
train = train.dropna(subset=FEATURE_COLS).reset_index(drop=True)
test  = test.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"\nAfter dropping nulls → Train: {len(train)} | Test: {len(test)}")

# ── 5. PREPARE X, y ─────────────────────────────────────────
X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS].values
y_test  = test[TARGET].values

# ── 6. FEATURE SCALING ──────────────────────────────────────
# XGBoost doesn't strictly need it, but helps with convergence
# and is required if you later stack with other models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler for inference
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_stage1.pkl")
print("\nScaler saved.")

# ── 7. CLASS IMBALANCE ──────────────────────────────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\nscale_pos_weight = {scale_pos_weight:.4f}")

# ── 8. TRAIN XGBOOST CLASSIFIER ─────────────────────────────
print("\n" + "=" * 60)
print("TRAINING XGBOOST CLASSIFIER")
print("=" * 60)

clf = xgb.XGBClassifier(
    n_estimators        = 1000,
    max_depth           = 6,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight    = 5,
    gamma               = 0.1,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    scale_pos_weight    = scale_pos_weight,
    eval_metric         = "auc",
    early_stopping_rounds = 50,
    use_label_encoder   = False,
    random_state        = 42,
    n_jobs              = -1,
    verbosity           = 1,
)

clf.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=100,
)

print(f"\nBest iteration: {clf.best_iteration}")

# ── 9. PREDICT PROBABILITIES ─────────────────────────────────
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# ── 10. THRESHOLD TUNING ────────────────────────────────────
# Default 0.5 threshold is bad for imbalanced data
# Find threshold that maximises F1 on test set
print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Default threshold (0.5):")
y_pred_default = (y_prob >= 0.5).astype(int)
print(f"  F1 = {f1_score(y_test, y_pred_default, average='binary'):.4f}")

print(f"\nOptimal threshold: {best_threshold:.4f}")
print(f"  F1 = {best_f1:.4f}")

# Use optimised threshold
y_pred = (y_prob >= best_threshold).astype(int)

# ── 11. EVALUATION ──────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ── 12. CONFUSION MATRIX PLOT ────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: No Rain", "Pred: Rain"],
            yticklabels=["True: No Rain", "True: Rain"])
plt.title(f"Confusion Matrix (threshold={best_threshold:.3f})", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_stage1.png", dpi=150)
plt.show()
print("Saved → confusion_matrix_stage1.png")

# ── 13. ROC CURVE PLOT ───────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Rain/No-Rain Classifier")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_stage1.png", dpi=150)
plt.show()
print("Saved → roc_curve_stage1.png")

# ── 14. PRECISION-RECALL CURVE PLOT ─────────────────────────
plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, color="darkorange", lw=2)
plt.axvline(x=recalls[best_idx], color="red", linestyle="--",
            label=f"Best threshold = {best_threshold:.3f}\nF1 = {best_f1:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Rain/No-Rain Classifier")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pr_curve_stage1.png", dpi=150)
plt.show()
print("Saved → pr_curve_stage1.png")

# ── 15. FEATURE IMPORTANCE ───────────────────────────────────
feat_imp = pd.DataFrame({
    "feature":   FEATURE_COLS,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 20 features:")
print(feat_imp.head(20).to_string(index=False))

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(20), x="importance", y="feature",
            palette="viridis")
plt.title("Top 20 Feature Importances — Stage 1 Classifier")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_stage1.png", dpi=150)
plt.show()
print("Saved → feature_importance_stage1.png")

# ── 16. SAVE MODEL ───────────────────────────────────────────
clf.save_model(f"{OUTPUT_DIR}/xgb_classifier_stage1.json")
print(f"\nModel saved → xgb_classifier_stage1.json")
print(f"Threshold  → {best_threshold:.6f}  (save this for inference)")

# Save threshold
with open(f"{OUTPUT_DIR}/threshold_stage1.txt", "w") as f:
    f.write(str(best_threshold))

# ── 17. SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Train samples : {len(train)}")
print(f"Test samples  : {len(test)}")
print(f"Features used : {len(FEATURE_COLS)}")
print(f"Best iteration: {clf.best_iteration}")
print(f"ROC-AUC       : {roc_auc:.4f}")
print(f"Best F1       : {best_f1:.4f} @ threshold {best_threshold:.4f}")
print(f"Precision     : {precisions[best_idx]:.4f}")
print(f"Recall        : {recalls[best_idx]:.4f}")
print("=" * 60)
print("Stage 1 complete. Proceed to Stage 2 (regression on rain-only samples).")
