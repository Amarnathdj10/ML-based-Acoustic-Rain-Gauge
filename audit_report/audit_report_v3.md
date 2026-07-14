
===================================================

Feature columns: 112

Train: 23989 rows  (2023-11-22 → 2026-03-07)

Test:  6606 rows   (2026-05-19 → 2026-06-17)

Train class balance:
rain
0    17245
1     6744
Name: count, dtype: int64
rain
0    71.89
1    28.11
Name: proportion, dtype: float64

Test class balance:
rain
0    6284
1     322
Name: count, dtype: int64
rain
0    95.13
1     4.87
Name: proportion, dtype: float64

====================================================

Scaler saved → scaler_stage1.pkl
Neg (no rain): 17245
Pos (rain)   : 6744
scale_pos_weight: 2.5571

====================================================

[0]	validation_0-auc:0.49597
[57]	validation_0-auc:0.56915

Best iteration: 7

====================================================

Default threshold (0.5):
  F1 = 0.0969

Optimal threshold: 0.5824
  F1       = 0.1285
  Precision= 0.0735
  Recall   = 0.5124

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

     No Rain       0.96      0.67      0.79      6284
        Rain       0.07      0.51      0.13       322

    accuracy                           0.66      6606
   macro avg       0.52      0.59      0.46      6606
weighted avg       0.92      0.66      0.76      6606

ROC-AUC Score: 0.5870

============================================================
FINAL SUMMARY
============================================================
Train samples  : 23989
Test samples   : 6606
Features used  : 112
Best iteration : 7
ROC-AUC        : 0.5870
Best F1        : 0.1285 @ threshold 0.5824
Precision      : 0.0735
Recall         : 0.5124
============================================================