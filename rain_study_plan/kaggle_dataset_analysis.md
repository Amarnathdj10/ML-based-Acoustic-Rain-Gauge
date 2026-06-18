# 🌧️ Kaggle Dataset Analysis — Acoustic Rain Prediction ML Model
**User:** Amarnath DJ (`amarnathdj`) | **Total Datasets Found:** 20 | **Combined Size:** ~212 GB

---

## Summary Verdict

> [!IMPORTANT]
> You already have **exactly the right kind of data** to build a high-quality supervised ML/DL model. Your datasets divide cleanly into three groups with distinct roles in the training pipeline.

---

## Dataset Groups

The 20 datasets fall into three distinct categories based on their naming convention and size:

| Group | Type | Count | Total Size | ML Role |
|---|---|---|---|---|
| **A** | Audio + Mechanical (labeled) | 7 | ~14.4 GB | 🟢 Primary training data |
| **B** | Audio-only (unlabeled) | 11 | ~192 GB | 🟡 Pre-training / self-supervised |
| **C** | Mechanical CSV only | 2 | ~90 KB | 🟠 Pairable labels for Group B |

---

## 🟢 GROUP A — Audio + Mechanical Ground Truth (Most Valuable)

These datasets contain **both audio recordings AND mechanical tipping bucket data** — the essential labeled pairs for supervised regression. Named with `_with_mech_data` or `_with_mechanical_data`.

> [!NOTE]
> Based on your README, these likely contain: WAV audio clips (`yyyy_mm_dd_hh_mm_ss_micro.wav`, 3.12 sec each) recorded by a USB mic inside the metallic enclosure, alongside CSV logs from the **Davis AeroCone 6466M** mechanical rain gauge (3-minute accumulated rainfall in mm).

| # | Dataset | URL | Size | Period | Usefulness |
|---|---|---|---|---|---|
| 1 | `December_2023_rainfall_audios_with_mech_data` | [link](https://www.kaggle.com/datasets/amarnathdj/dec-2023-rainfall-data) | 748 MB | Dec 2023 | ⭐⭐⭐⭐⭐ |
| 2 | `November_2023_rainfall_audios_with_mech_data` | [link](https://www.kaggle.com/datasets/amarnathdj/nov-2023-rainfall-data) | 320 MB | Nov 2023 | ⭐⭐⭐⭐⭐ |
| 3 | `January_2024_rainfall_audios_with_mech_data` | [link](https://www.kaggle.com/datasets/amarnathdj/jan-2024-rainfall-data) | 297 MB | Jan 2024 | ⭐⭐⭐⭐⭐ |
| 4 | `May_2024_rainfall_audios_with_mechanical_data` | [link](https://www.kaggle.com/datasets/amarnathdj/may-2024-rainfall-data) | 3.2 GB | May 2024 | ⭐⭐⭐⭐⭐ |
| 5 | `July_2024_rainfall_audios_with_mech_data` | [link](https://www.kaggle.com/datasets/amarnathdj/july-2024-rainfall-data) | 3.8 GB | Jul 2024 | ⭐⭐⭐⭐⭐ |
| 6 | `September_2024_rainfall_audios_with_mech_data` | [link](https://www.kaggle.com/datasets/amarnathdj/sept-2024-rainfall-data) | 1.7 GB | Sep 2024 | ⭐⭐⭐⭐⭐ |
| 7 | `April_2024_rainfall_audios_with_mech_data` | [link](https://www.kaggle.com/datasets/amarnathdj/april-2024-rainfall-data) | 4.3 GB | Apr 2024 | ⭐⭐⭐⭐⭐ |

**Why these are gold:**
- **Paired data** — every audio window has a ground-truth rainfall label (mm/3-min)
- **Multi-month coverage** — covers monsoon peaks (Jul, Sep), dry spells (Jan), and shoulder seasons
- **Real hardware** — recorded on actual deployment hardware (Jieli mic + Raspberry Pi), so the model will generalize to the real system
- **~14.4 GB total** — sufficient for deep learning (CNNs on spectrograms, LSTMs on features)

---

## 🟡 GROUP B — Large Audio-Only Datasets (Unlabeled)

These are massive WAV recording datasets **without an accompanying mechanical CSV** uploaded. They are unlabeled from a supervised-learning standpoint.

| # | Dataset | URL | Size | Period |
|---|---|---|---|---|
| 1 | `December_2024_rain_data` | [link](https://www.kaggle.com/datasets/amarnathdj/december-2024-rain-data) | **51.3 GB** | Dec 2024 |
| 2 | `October_2025_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/oct-2025-rainfall-data) | **27.8 GB** | Oct 2025 |
| 3 | `September_2025_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/sept-2025-rainfall-data) | **20.6 GB** | Sep 2025 |
| 4 | `June_2025_Rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/june-2025-rainfall-data) | **19.8 GB** | Jun 2025 |
| 5 | `Feb_to_Mar_2026_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/feb-mar-2026-rain-data) | **17.9 GB** | Feb–Mar 2026 |
| 6 | `January_2025_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/jan-2025-rainfall-data) | **18.2 GB** | Jan 2025 |
| 7 | `May_2026_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/may-2026-rainfall-data) | **9.4 GB** | May 2026 |
| 8 | `November_2024_Rainfall_Data` | [link](https://www.kaggle.com/datasets/amarnathdj/nov-2024-rainfall-data) | **9.5 GB** | Nov 2024 |
| 9 | `May_2025_Rainfall_Data` | [link](https://www.kaggle.com/datasets/amarnathdj/may-2025-rainfall-data) | **6.5 GB** | May 2025 |
| 10 | `June_2026_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/june-2026-rainfall-data) | **4.5 GB** | Jun 2026 |
| 11 | `August_2025_Rain_Data` | [link](https://www.kaggle.com/datasets/amarnathdj/august-2-25-rain-data) | **5.7 GB** | Aug 2025 |

**How these can still be useful:**

1. **Self-supervised / Contrastive pre-training** — Train an audio encoder (e.g., AudioMAE, wav2vec-style) on these to learn rich acoustic representations of rain, then fine-tune on Group A labeled data.
2. **Semi-supervised learning** — Use Group A labels + pseudo-labels from Group B.
3. **Data augmentation** — Mine audio clips with known high/zero rainfall from timestamps (e.g., dry season months for "no rain" class).
4. **Pairing with Group C** — Dec 2024 audio pairs with the Dec 2024 mechanical CSV (see Group C below). Nov 2024 audio pairs with Oct–Nov 2024 mechanical CSV.

> [!WARNING]
> Do NOT use these as direct supervised training data without the matching mechanical CSV. The audio timestamps must be aligned to gauge readings — without the gauge log, you have no labels.

---

## 🟠 GROUP C — Mechanical Gauge CSVs Only (Small Tabular)

Tiny datasets containing **only the tipping bucket readings** (no audio). These are the ground-truth label files that can be paired with Group B audio.

| # | Dataset | URL | Size | Period | Can Pair With |
|---|---|---|---|---|---|
| 1 | `December_2024_mechanical_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/dec-2024-mechanical-rainfall) | **38 KB** | Dec 2024 | Group B #1 (Dec 2024, 51.3 GB) |
| 2 | `Oct_to_Nov_2024_mechanical_rainfall_data` | [link](https://www.kaggle.com/datasets/amarnathdj/oct-nov-2024-rainfall-data) | **51 KB** | Oct–Nov 2024 | Group B #8 (Nov 2024, 9.5 GB) |

**Expected CSV columns** (based on Davis AeroCone + your pipeline):
- `timestamp` — date/time of reading
- `rainfall_mm` — accumulated precipitation per interval (likely 0.2 mm/tip)
- Possibly: `temperature`, `humidity`, `wind_speed` (standard Davis weather station outputs)

**Action:** Merge these CSVs with the timestamp-matched audio from the corresponding Group B datasets to create **two more fully-labeled datasets** — effectively doubling your labeled training corpus.

---

## 📊 Complete Dataset Inventory

| # | Title | Size | Group | Directly Usable for Supervised ML? |
|---|---|---|---|---|
| 1 | December_2024_rain_data | 51.3 GB | B | ❌ (needs mech CSV) |
| 2 | August_2025_Rain_Data | 5.7 GB | B | ❌ |
| 3 | December_2024_mechanical_rainfall_data | 38 KB | C | 🟠 (labels only, pair with #1) |
| 4 | June_2025_Rainfall_data | 19.8 GB | B | ❌ |
| 5 | January_2025_rainfall_data | 18.2 GB | B | ❌ |
| 6 | May_2025_Rainfall_Data | 6.5 GB | B | ❌ |
| 7 | November_2024_Rainfall_Data | 9.5 GB | B | ❌ (pair with #8 mech CSV) |
| 8 | Oct_to_Nov_2024_mechanical_rainfall_data | 51 KB | C | 🟠 (labels only, pair with #7) |
| 9 | May_2026_rainfall_data | 9.4 GB | B | ❌ |
| 10 | Feb_to_Mar_2026_rainfall_data | 17.9 GB | B | ❌ |
| 11 | January_2024_rainfall_audios_with_mech_data | 297 MB | A | ✅ |
| 12 | April_2024_rainfall_audios_with_mech_data | 4.3 GB | A | ✅ |
| 13 | December_2023_rainfall_audios_with_mech_data | 748 MB | A | ✅ |
| 14 | July_2024_rainfall_audios_with_mech_data | 3.8 GB | A | ✅ |
| 15 | October_2025_rainfall_data | 27.8 GB | B | ❌ |
| 16 | June_2026_rainfall_data | 4.5 GB | B | ❌ |
| 17 | September_2025_rainfall_data | 20.6 GB | B | ❌ |
| 18 | September_2024_rainfall_audios_with_mech_data | 1.7 GB | A | ✅ |
| 19 | November_2023_rainfall_audios_with_mech_data | 320 MB | A | ✅ |
| 20 | May_2024_rainfall_audios_with_mechanical_data | 3.2 GB | A | ✅ |

---

## 🧠 Recommendations for Model Building

### Immediate: Improve the Supervised Model (Group A)
Your current pipeline uses only **hand-crafted features** (Mel stats, ZCR, RMS, Spectral Centroid) achieving **~50% Pearson r**. With all 7 Group A datasets combined:

```
Estimated labeled training data: ~14.4 GB audio + 7 mechanical CSVs
Estimated 3-min windows: ~2,000–5,000 labeled examples (depending on rain event density)
```

**Recommended DL upgrades:**
1. **CNN on Mel spectrograms** — Feed 64×T spectrograms directly into a Conv1D/Conv2D network instead of hand-crafted features. This lets the model learn its own features.
2. **CNN-LSTM hybrid** — CNN extracts spatial features from spectrograms, LSTM models temporal rainfall dynamics across windows.
3. **1D CNN on raw waveform** — Use raw PCM audio as input (WaveNet-inspired approach).

### Medium-term: Unlock Group B with Self-Supervised Learning
1. Train an **audio encoder** (e.g., masked autoencoder on spectrograms) on all ~192 GB of Group B audio.
2. Fine-tune the encoder on Group A labeled pairs.
3. This approach is proven to massively boost performance when labeled data is scarce.

### Immediate Action: Pair Group C with Group B
- Download **Dec 2024 mech CSV** + **Dec 2024 audio** → run your alignment pipeline → gain ~51 GB of new labeled data
- Download **Oct–Nov 2024 mech CSV** + **Nov 2024 audio** → gain ~9.5 GB more

### Missing Mech Data for Group B
For the remaining unlabeled Group B months (Jun/Aug/Sep/Oct 2025, Jan/May/Jun 2026, Feb–Mar 2026), check if the mechanical CSV data exists locally on your Raspberry Pi but wasn't uploaded to Kaggle. If so, upload them to unlock ~150 GB of additional supervised training data.

---

## ⚡ Key Insight

> [!TIP]
> Your datasets are **uniquely valuable** — they are real-world, long-duration, multi-season acoustic recordings from a custom-built sensor with calibrated ground-truth labels. No public dataset comes close to this for the specific task of acoustic rainfall rate estimation. The biggest opportunity is **converting Group B + Group C into labeled training data** by completing the pairing for Dec 2024 and Oct–Nov 2024, and uploading any missing mechanical CSVs for the remaining months.
