# 🌧️ ML-Based Acoustic Rain Gauge

> Predicting rainfall intensity from environmental audio using signal processing and machine learning — no optical or mechanical sensors required.

Developed as part of a research internship at **ICFOSS, Thiruvananthapuram**.

---

## 📌 Overview

Traditional rain gauges require mechanical tipping buckets or optical sensors. This project explores a low-cost alternative: using a **microphone and machine learning** to classify rainfall intensity directly from the acoustic signature of rain.

A Raspberry Pi captures 10-second WAV clips every 3 minutes. Audio features are extracted using Librosa, then fed into an XGBoost classifier trained on ground-truth labels from a co-located mechanical tipping bucket gauge.

---

## 🎯 Problem Statement

Can we reliably estimate rainfall intensity — light, moderate, heavy, or no rain — using only environmental audio? This project answers that using a real-world dataset collected over 2.5 years across multiple monsoon and dry seasons in Kerala, India.

---

## 🗂️ Dataset

| Property | Details |
|----------|---------|
| Audio clips | ~30,000 labeled 10-sec WAV files |
| Sample rate | 8 kHz (downsampled from 48 kHz) |
| Capture interval | Every 3 minutes |
| Label alignment | ±90-second window matched to tipping bucket timestamps |
| Date range | 2023 – 2025 (136 distinct calendar days) |
| Classes | No Rain / Light / Moderate / Heavy (WMO-standard hourly thresholds) |

Ground-truth rainfall labels are sourced from a mechanical tipping bucket rain gauge and aligned to audio timestamps using a custom synchronisation pipeline.

---

## 🧠 Methodology

```
Raw WAV Clips (10s, 8kHz)
        ↓
Feature Extraction (Librosa)
        ↓
Hourly Aggregation + WMO Label Assignment
        ↓
XGBoost 4-Class Classifier
        ↓
Rainfall Intensity Prediction
```

### Feature Extraction (~130 features per window)

| Feature Group | Features | Why It Matters |
|---|---|---|
| MFCCs (13 coefficients) | Mean + Std (52 total) | Captures rain's unique timbral texture |
| Delta-MFCCs | Mean + Std (26 total) | Rate of timbre change — rain is bursty, background is steady |
| Zero Crossing Rate | Mean + Std | Distinguishes noise-like rain from tonal sounds |
| RMS Energy | Mean + Std + variability | Drizzle vs downpour amplitude differences |
| Spectral Centroid & Bandwidth | Mean + Std | Frequency distribution of rain impact sounds |
| Spectral Flatness | Mean + Std | Noise-likeness — rain is more noise-like than speech/machinery |
| Spectral Contrast (7 bands) | Mean + Std | Frequency-specific peaks unique to rain |
| Chroma | Mean + Std | Flags tonal interference (voices, mechanical hum) |

### Classification

- **Model:** XGBoost (`XGBClassifier`)
- **Validation:** Time-based split (no random shuffling to prevent temporal leakage)
- **Class imbalance:** Addressed via `scale_pos_weight` and SMOTE on training folds
- **Label scheme:** WMO standard hourly thresholds → 4 classes

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Weighted F1 | 0.56 |
| Macro F1 | 0.42 |
| Training samples | ~1,924 hourly windows |

> **Note:** Macro F1 is lower due to class imbalance — heavy rain events are rare in the dataset. Ongoing work focuses on improving minority class recall.

---

## 🗃️ Repository Structure

```
ML-based-Acoustic-Rain-Gauge/
│
├── audio_processing.ipynb          # EDA and spectrogram visualisation
├── feature_extraction.ipynb        # Librosa feature extraction pipeline
├── classifier_stage1.ipynb         # Binary rain/no-rain classifier
├── acoustic_preprocessing.py       # Preprocessing utilities
├── classifier_panns_xgboost.py     # PANNs + XGBoost hybrid experiment
│
├── aligned_data/                   # Timestamp-aligned audio-label pairs
├── features/                       # Extracted parquet feature files
├── output_images/                  # Spectrograms and confusion matrices
├── audit_report/                   # Dataset quality audit outputs
│
├── BATCH_WORKFLOW.md               # Incremental feature extraction guide
├── FEATURE_ENGINEERING_NOTES.md    # Feature rationale and analysis
└── DATA_FLOW_DIAGRAM.md            # End-to-end pipeline diagram
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo

```bash
git clone https://github.com/Amarnathdj10/ML-based-Acoustic-Rain-Gauge.git
cd ML-based-Acoustic-Rain-Gauge
```

### 2. Install dependencies

```bash
pip install librosa xgboost scikit-learn pandas numpy matplotlib seaborn pyarrow
```

### 3. Run feature extraction

Open `feature_extraction.ipynb` and point it to your audio dataset directory. Outputs a `.parquet` feature file per monthly batch.

> For large datasets (30,000+ clips), see `BATCH_WORKFLOW.md` for the incremental Kaggle-based extraction pipeline.

### 4. Train the classifier

```python
import xgboost as xgb
import pandas as pd

feature_df = pd.read_parquet("features/features_all_combined.parquet")

# Time-based split — never use random split on temporal data
split_date = feature_df["timestamp"].quantile(0.7)
train_df = feature_df[feature_df["timestamp"] <= split_date]
test_df  = feature_df[feature_df["timestamp"] > split_date]

X_train = train_df.drop(columns=["rainfall_mm", "timestamp", "label"])
y_train = train_df["label"]

model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
```

---

## 🚧 Known Challenges

- **Class imbalance:** Heavy rain events account for <5% of samples
- **Ambient noise:** Wind, insects, and traffic overlap with rain frequencies in some clips
- **Temporal leakage:** Must use time-based train/test splits — random splits give falsely optimistic results
- **Label alignment:** ±90-second GPS/timestamp drift between audio and gauge required custom alignment logic

---

## 🔭 Future Work

- [ ] CNN/LSTM on raw spectrograms instead of hand-crafted features
- [ ] PANNs (Pretrained Audio Neural Networks) as feature extractor backbone
- [ ] Real-time inference pipeline on Raspberry Pi
- [ ] Extended dataset across multiple geographic locations

---

## 🏛️ Acknowledgements

This project is conducted as part of a research internship at [ICFOSS](https://icfoss.in/) (International Centre for Free and Open Source Software), Karyavattom, Thiruvananthapuram, Kerala, India.

---

## 📚 References

- [Librosa Documentation](https://librosa.org/) — audio feature extraction
- [XGBoost Documentation](https://xgboost.readthedocs.io/) — gradient boosted trees
- WMO No. 8 — Guide to Meteorological Instruments and Methods of Observation (rainfall intensity thresholds)

---

## 👤 Author

**Amarnath D.J.**  
B.Tech CSE (AI & ML), SCTCE Thiruvananthapuram  
[Portfolio](https://amarnath-dj.me) · [LinkedIn](https://linkedin.com/in/amarnath-dj-b710abc) · [GitHub](https://github.com/Amarnathdj10)
