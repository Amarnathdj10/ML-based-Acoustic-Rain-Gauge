
-------------------------------------------------------

Files in dataset:
  features_september_2025.parquet
  features_july_2024.parquet
  features_jan_2025.parquet
  features_may_2025.parquet
  features_september_2024.parquet
  features_december_2024.parquet
  features_feb_to_march_2026.parquet
  features_may_2024.parquet
  features_december_2023.parquet
  features_october_2025.parquet
  features_august_2025.parquet
  features_june_2026.parquet
  features_may_2026.parquet
  features_november_2024.parquet
  features_november_2023.parquet
  features_april_2024.parquet
  features_january_2024.parquet
  features_june_2025.parquet


Total rows : 30595
Total cols : 117
Date range : 2023-11-22     → 2026-06-17

Class balance:
rain
0    23529
1     7066
Name: count, dtype: int64
rain
0    76.9
1    23.1
Name: proportion, dtype: float64

---------------------------------------------------------

Feature columns: 112

Train: 20921 rows  (2023-11-22 → 2025-10-30)

Test:  9674 rows   (2026-02-27 → 2026-06-17)

Train class balance:
rain
0    17243
1     3678
Name: count, dtype: int64
rain
0    82.42
1    17.58
Name: proportion, dtype: float64

Test class balance:
rain
0    6286
1    3388
Name: count, dtype: int64
rain
0    64.98
1    35.02
Name: proportion, dtype: float64

------------------------------------------------------

Scaler saved → scaler_stage1.pkl
Neg (no rain): 17243
Pos (rain)   : 3678
scale_pos_weight: 4.6881

------------------------------------------------------

[0]	validation_0-auc:0.48297
[52]	validation_0-auc:0.47297

------------------------------------------------------

Default threshold (0.5):
  F1 = 0.3629

Optimal threshold: 0.4624
  F1       = 0.5313
  Precision= 0.3867
  Recall   = 0.8486

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

     No Rain       0.77      0.27      0.40      6286
        Rain       0.39      0.85      0.53      3388

    accuracy                           0.48      9674
   macro avg       0.58      0.56      0.47      9674
weighted avg       0.64      0.48      0.45      9674

ROC-AUC Score: 0.5344

============================================================
FINAL SUMMARY
============================================================
Train samples  : 20921
Test samples   : 9674
Features used  : 112
Best iteration : 2
ROC-AUC        : 0.5344
Best F1        : 0.5313 @ threshold 0.4624
Precision      : 0.3867
Recall         : 0.8486
============================================================