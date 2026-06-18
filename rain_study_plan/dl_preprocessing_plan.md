# 🌧️ DL Preprocessing Pipeline — Acoustic Rain Gauge
**Project:** ML-based Acoustic Rain Gauge | **Target:** Deep Learning (CNN / CNN-LSTM)

---

## 📌 Current State vs. Target State

| Aspect | Current (ML) | Target (DL) |
|---|---|---|
| Input | 264-dim hand-crafted feature vector | Raw Mel spectrogram image (64×T) or raw waveform |
| Model | Random Forest / SVR / MLP | CNN, CNN-LSTM, or 1D WaveNet |
| Correlation | ~50% Pearson r | Target: >75% Pearson r |
| Data used | Group A only (~14.4 GB) | Group A + paired Group B+C (~75 GB) |
| Window | 3-min aggregated feature | Per-window spectrogram (3-min or sliding) |

---

## 🗂️ Dataset Inventory (Your 20 Kaggle Datasets)

### ✅ Group A — Immediately Usable (Audio + Mechanical CSV)

| Dataset | Kaggle URL | Size | Period |
|---|---|---|---|
| Dec 2023 | [link](https://www.kaggle.com/datasets/amarnathdj/dec-2023-rainfall-data) | 748 MB | Dec 2023 |
| Nov 2023 | [link](https://www.kaggle.com/datasets/amarnathdj/nov-2023-rainfall-data) | 320 MB | Nov 2023 |
| Jan 2024 | [link](https://www.kaggle.com/datasets/amarnathdj/jan-2024-rainfall-data) | 297 MB | Jan 2024 |
| Apr 2024 | [link](https://www.kaggle.com/datasets/amarnathdj/april-2024-rainfall-data) | 4.3 GB | Apr 2024 |
| May 2024 | [link](https://www.kaggle.com/datasets/amarnathdj/may-2024-rainfall-data) | 3.2 GB | May 2024 |
| Jul 2024 | [link](https://www.kaggle.com/datasets/amarnathdj/july-2024-rainfall-data) | 3.8 GB | Jul 2024 |
| Sep 2024 | [link](https://www.kaggle.com/datasets/amarnathdj/sept-2024-rainfall-data) | 1.7 GB | Sep 2024 |

**~14.4 GB total → Primary supervised training corpus**

### 🔄 Group B+C — Pair to Unlock More Labels

| Audio Dataset | Mechanical CSV | Combined Size |
|---|---|---|
| Dec 2024 audio (51.3 GB) | Dec 2024 mech CSV (38 KB) | **51.3 GB unlockable** |
| Oct/Nov 2024 audio (9.5 GB) | Oct-Nov 2024 mech CSV (51 KB) | **9.5 GB unlockable** |

---

## 🔬 What Your Mechanical CSV Looks Like

```
time,Device Name,Measurement,Rainfall (mm)
2024-12-01 00:01:44,rainpi_3,device_frmpayload_data_rain,0.0
2024-12-01 00:04:52,rainpi_3,device_frmpayload_data_rain,0.0
...
2024-12-01 08:17:33,rainpi_3,device_frmpayload_data_rain,0.2
```

- **Interval:** ~3 minutes (not exact — variable ~3.07 min)
- **Resolution:** 0.2 mm per tip (Davis AeroCone tipping bucket)
- **Key column:** `Rainfall (mm)` — this is your regression **target (y)**
- **Key issue:** Timestamps are irregular. Must use bisect-based window alignment (already in your `model_pipeline.py`).

---

## 🏗️ DL Preprocessing Pipeline (Step-by-Step)

```
[WAV Files]          [Mechanical CSV]
     │                      │
     ▼                      ▼
1. Parse timestamps    1. Parse timestamps
2. Sort chronologically  2. Clean rainfall values
     │                      │
     └──────────┬────────────┘
                ▼
        3. Align: Group WAVs into
           3-minute windows (bisect)
                │
                ▼
        4. Concatenate audio per window
           (~18 WAV clips × 3.12s = ~56s audio)
                │
                ▼
        5. Normalize audio amplitude
           (per-clip peak normalization)
                │
                ▼
        6. Compute Mel Spectrogram
           (n_mels=64, sr=8000, hop=256)
           → shape: (64, T) per window
                │
                ▼
        7. Resize/Pad to fixed (64, 128)
           for CNN input
                │
                ▼
        8. Apply log compression
           librosa.power_to_db()
                │
                ▼
        9. Global Normalization
           mean=0, std=1 across dataset
                │
                ▼
       10. Label assignment
           y = rainfall_mm for window
           log1p transform for heavy-tail
                │
                ▼
       11. Save as .npz (X, y, timestamps)
           per month → combine for training
```

---

## 📐 DL Input Format

### Option A: CNN on Mel Spectrograms (Recommended)
```
Input shape:  (batch, 1, 64, 128)   # (B, C, Mel_bins, Time_frames)
              or (batch, 64, 128, 1) # TensorFlow channels-last
Output:       (batch, 1)             # Predicted rainfall in mm
```

### Option B: CNN-LSTM (temporal sequences)
```
Input shape:  (batch, seq_len, 64, 128)  # sliding window of spectrograms
Output:       (batch, seq_len, 1)         # rainfall per step
```

### Option C: 1D CNN on raw waveform
```
Input shape:  (batch, 1, N_samples)  # N = 8000 * 56 = 448,000
Output:       (batch, 1)
```

> [!NOTE]
> Option A (CNN on Mel spectrograms) is the easiest starting point and likely best for your dataset size.

---

## ⚙️ Key Preprocessing Parameters

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 8000 Hz | Already downsampled in your dataset |
| n_mels | 64 | Matches your existing feature extraction |
| hop_length | 256 | ~32ms time resolution at 8kHz |
| n_fft | 1024 | Frequency resolution |
| Target spec shape | (64, 128) | Fixed CNN input; ~32s of audio |
| Label transform | `log1p(y)` | Rainfall is highly right-skewed |
| Train/Val/Test split | 70/15/15 | Chronological (no data leakage) |
| Normalization | Global mean/std | Computed on train set only |

---

## 🚨 Critical Preprocessing Issues to Handle

> [!WARNING]
> **Class imbalance**: ~70-80% of 3-min windows have 0.0 mm rainfall. If you train naively, the model will always predict 0. Solutions:
> - Oversample rainy windows (repeat or augment)
> - Use a weighted loss: `MSE * (1 + α * y_true)`
> - Two-stage model: classifier (rain/no-rain) → regressor (rate)

> [!WARNING]
> **Label noise**: The mechanical gauge reports in 0.2mm tips. Many windows near 0 are ambiguous. Consider: treat labels < 0.2mm as 0.0 to clean noise.

> [!CAUTION]
> **Timestamp drift**: Audio WAV filename timestamps may drift slightly from the mechanical gauge clock. Your current bisect approach handles this well, but add a ±30s tolerance buffer for the DL pipeline.

> [!TIP]
> **Data augmentation** for DL: On Mel spectrograms apply:
> - Time masking (SpecAugment): mask random time columns
> - Frequency masking: mask random mel bands
> - Gaussian noise addition on the spectrogram
> These will significantly reduce overfitting.

---

## 📊 Expected Dataset Statistics (Group A)

```
Estimated 3-min windows:     ~2,000 – 5,000 labeled samples
Expected non-zero rain windows: ~20-30% (monsoon months)
Expected zero-rain windows:  ~70-80%
Spectrogram size per sample: 64 × 128 × 4 bytes = 32 KB
Total preprocessed dataset:  ~64–160 MB (compact .npz)
```

---

## 🗓️ Recommended Execution Order

| Step | Action | Script |
|---|---|---|
| 1 | Download all Group A datasets from Kaggle | Manual / kaggle CLI |
| 2 | Run **`dl_preprocess.py`** on each month | `dl_preprocess.py` |
| 3 | Merge all monthly `.npz` files | `dl_preprocess.py --merge` |
| 4 | Inspect class balance and spectrogram quality | EDA notebook |
| 5 | Pair Dec 2024 audio + mech CSV → unlock 51 GB | `dl_preprocess.py` |
| 6 | Train CNN baseline on merged dataset | (next step) |

---

## 📁 Output Directory Structure

```
D:\ARG_dataset\
├── preprocessed\
│   ├── group_a\
│   │   ├── dec_2023.npz       # X=(N,64,128), y=(N,), t=(N,)
│   │   ├── nov_2023.npz
│   │   ├── jan_2024.npz
│   │   ├── apr_2024.npz
│   │   ├── may_2024.npz
│   │   ├── jul_2024.npz
│   │   └── sep_2024.npz
│   ├── group_bc\
│   │   ├── dec_2024.npz       # After pairing Group B+C
│   │   └── oct_nov_2024.npz
│   └── master\
│       ├── train.npz           # 70% chronological
│       ├── val.npz             # 15%
│       ├── test.npz            # 15%
│       └── norm_stats.json     # mean, std for denormalization
```
