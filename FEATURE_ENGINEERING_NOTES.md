# Expanded Acoustic Feature Set — Rationale

## Overview

Your original `classifier_training.ipynb` extracted **~46 features per audio clip** (20 MFCCs × mean+std, plus 5 spectral summary stats + ZCR). 

The updated batch notebook extracts **~130 features per clip** (after mean+std aggregation across clips in the 3-minute window). This is not padding — each addition directly addresses a modeling challenge specific to your dataset.

---

## Key Challenge: Heavy Rainfall Intensity Imbalance

From the audit report:
- 78% of rain windows have < 0.5 mm
- 95% have < 1.3 mm
- Median rain window: 0.44 mm

A model trained only on coarse acoustic stats (e.g., "average RMS energy") will struggle to distinguish 0.4 mm from 0.5 mm, or to reliably spot drizzle at all.

**Solution**: Extract features that capture *variability and texture*, not just averages.

---

## Expanded Features by Category

### 1. **RMS Energy (Time Domain)**

**Original**: `rms` (single mean value per clip)

**Updated**: `rms_mean`, `rms_std`, `rms_max` per clip, then aggregated across clips with both mean and std
- `rms_mean_mean`: Average loudness across clips
- `rms_mean_std`: Variability of loudness *across* clips in the window (burst vs. steady)
- `rms_std_mean`: Variability of loudness *within* each clip (dynamic vs. flat)
- `rms_max_mean`: Peak energy per clip

**Why**: Light rain often sounds bursty and variable, not like a steady tone. A window with 18 clips where RMS varies widely (std=0.05) looks different from one where it's steady (std=0.002), even if mean RMS is similar. The model needs to see that variability.

---

### 2. **Spectral Centroid (Spectral Shape)**

**Original**: Single mean per clip

**Updated**: `centroid_mean` and `centroid_std` per clip, aggregated with mean and std across clips

**Why**: Rain has no pitch — its spectral centroid wanders across frequencies as drops hit at different times. A steady synthetic tone stays centered; rain bounces around. Both the average centroid and its variance carry signal.

---

### 3. **Spectral Bandwidth & Rolloff**

**Original**: Included but only mean

**Updated**: Now include std across time frames within a clip, and then mean/std across clips

**Why**: Bandwidth and rolloff both measure "spread" in the spectrum. Rain spreads energy across many frequencies (high bandwidth); background noise or a single stuck valve might concentrate it (narrow bandwidth). Seeing how bandwidth *varies* within and across clips is richer than a single summary.

---

### 4. **Spectral Flatness** ⭐ NEW

```python
flatness = librosa.feature.spectral_flatness(y=y)
```

A metric from 0 (very tonal) to 1 (very noise-like). 
- Rain sound is noise-like (flatness ≈ 0.7–0.9)
- Background tones, hum, traffic, voices are tonal (flatness ≈ 0.1–0.3)

**Why**: Your acoustic sensor will pick up background noise, mechanical sounds, etc. Flatness lets the model learn to recognize rain's broadband, relatively featureless acoustic signature and ignore tonal contamination.

---

### 5. **Spectral Contrast Across 7 Frequency Bands** ⭐ NEW

```python
contrast = librosa.feature.spectral_contrast(y=y)  # shape (7, n_frames)
```

Measures the peak-to-valley depth in each of 7 frequency bands:
- Band 1: <200 Hz (sub-bass)
- Band 2: 200–393 Hz
- ... (logarithmically spaced)
- Band 7: >6500 Hz (high frequencies)

**Why**: Rain is typically broadband but may have characteristic "bumps" in the mid-to-high frequencies (2–6 kHz region where drops dominate). A stuck valve or motor hum will have a sharp peak in a single band. Contrast captures that shape, and the model can learn which bands are "rainy."

---

### 6. **Chroma Features** ⭐ NEW

```python
chroma = librosa.feature.chroma_stft(y=y)  # 12-bin pitch class distribution
```

Chroma projects the spectrum onto the 12 pitch classes (C, C#, D, ..., B), regardless of octave. 

**Why**: Rain has no clear pitch, so chroma should be flat/uniform. Background sounds (music, speech, mechanical noise) will show strong, repeating peaks in certain pitch classes. Aggregating chroma across a window shows the model "this sound has no musical pitch — it's probably rain."

---

### 7. **Delta-MFCCs (First-Order Time Derivative)** ⭐ NEW

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfcc)  # Rate of change of each MFCC coefficient
```

MFCCs capture overall timbre (the "color" of the sound). Delta-MFCCs capture how that timbre is *changing* over time.

**Why**: Rain sound texture changes as drops hit and decay (attack → decay envelope). A steady background tone has low delta-MFCCs; rain has high ones. Capturing the rate of timbral change is crucial for distinguishing transient rain from sustained background.

---

## Aggregation Strategy: Mean + Std Across Clips

The original notebook averaged clip-level features to get one row per window. This new version computes *both* mean and std across the 17–18 (or 57–60) clips in each 3-minute window.

**Example**: RMS features become 6 columns instead of 1:
```
Original: rms_mean = 0.025  (average loudness in this window)

New:
  rms_mean_mean = 0.025  (average clip loudness)
  rms_mean_std  = 0.003  (how much clip loudness varies across the 18 clips)
  rms_std_mean  = 0.008  (average within-clip variability)
  rms_std_std   = 0.002  (how much within-clip variability changes across clips)
  rms_max_mean  = 0.045  (average peak loudness per clip)
  rms_max_std   = 0.005  (how much peak loudness varies)
```

**Why this matters**: A window with stable, steady rain energy (low rms_mean_std) looks different from a bursty, variable window (high rms_mean_std), even if the total average energy is the same. The std *is* signal.

---

## Feature Count Breakdown

| Category | Original | New | Notes |
|----------|----------|-----|-------|
| RMS (mean, std, max) | 1 | 6 | mean + std per clip → mean + std across clips |
| Centroid | 1 | 4 | Now include std within clips |
| Bandwidth | 1 | 2 | Now include std |
| Rolloff | 1 | 2 | Now include std |
| Flatness | 0 | 2 | ⭐ NEW: noise-likeness metric |
| Spectral Contrast (7 bands) | 0 | 14 | ⭐ NEW: peak-valley shape per frequency band |
| Chroma | 0 | 2 | ⭐ NEW: pitch class distribution (should be flat for rain) |
| Zero-Crossing Rate | 1 | 2 | Now include std |
| MFCCs (13 coefficients) | 40 | 52 | 13 × (mean+std) per clip → mean+std across clips |
| Delta-MFCCs (13 coefficients) | 0 | 26 | ⭐ NEW: rate of timbre change |
| **Total per clip** | **~46** | **~114** | — |
| **Aggregation** | Mean only | Mean + Std | Doubles count again |
| **Final per window** | ~46 | ~130+ | Plus metadata (n_clips_used) |

---

## Overfitting Risk: Is 130 features too many?

Not for your dataset size. You have:
- 28,657 total 3-minute windows
- ~7,000 rain windows for classification
- ~2.5 years of data (good temporal diversity, helps generalization)

**Feature-to-sample ratio**: 130 / 7,000 ≈ 0.02, which is very healthy. XGBoost with built-in regularization (max_depth, subsample, colsample) can comfortably handle this.

**However**, you should:
1. **Use time-based cross-validation** (not random splits) to avoid temporal leakage
2. **Stratify by rainfall intensity bins** to ensure splits have balanced rain amounts
3. **Run feature importance analysis** (XGBoost's native feature importance, or SHAP values) to prune irrelevant features after training
4. **Start with a simpler model** (XGBoost with default regularization) before moving to CNNs — the expanded features may be all you need

---

## Computational Cost

Processing all 28,657 windows with these features will take **longer** than the original notebook:

- **Original** (46 features, Sept 2024): ~4 min
- **Updated** (130 features, all 16 months): estimate **1–2 hours** on Kaggle (depending on CPU availability and librosa performance)

**Mitigation**: 
- Test on 2–3 smaller months first (to validate the feature extraction works)
- Then batch process remaining months in groups of 4–5

---

## Next Steps: Model Training

Once you have `features_all_combined.parquet`, train your binary rain/no-rain classifier:

1. **Load combined features**
2. **Drop rows with NaN features** (shouldn't be many, but check)
3. **Log-transform `rainfall_mm`** for the 7,000 rain-only samples (for Stage 2 regression later)
4. **Split by time** (e.g., first 70% of unique dates → train, last 30% → test, no temporal overlap)
5. **Stratify by rainfall intensity bins** within each split
6. **Train XGBoost** with early stopping
7. **Evaluate on test set** (not random CV, but time-forward validation)

Once the binary classifier works well, train Stage 2: regression on rain-only samples to estimate intensity.

---

## References & Feature Justification

- **Spectral Flatness**: Measured in dB, captures the degree to which a signal is noise-like vs. tonal. High flatness (closer to 1 in librosa) = broadband noise (like rain). Low flatness = tonal (like background machinery). See: *Timbral and Spectral Features for Automatic Music Tagging* (Tzanetakis & Cook, 2002).

- **Spectral Contrast**: Captures the local peak-to-valley relationships in the spectrum across frequency bands. Useful for distinguishing transient events (rain drops) from sustained background. See: librosa docs.

- **Delta Features**: In speech recognition and music analysis, the time derivative of perceptual features (MFCCs, MFSCCs) significantly improves classification because it captures *dynamic* properties (how a sound is *changing*) rather than static properties (what a sound *is*). Rain texture changes rapidly; steady background noise does not.

---

## Troubleshooting: Feature Extraction Fails on Some Clips

If a clip is corrupt or extremely short, `extract_clip_features()` has a guard:
```python
min_len = 2048
if len(y) < min_len:
    y = np.pad(y, (0, min_len - len(y)))
```

And the main loop skips individual windows if *all* clips fail:
```python
if len(clip_features) == 0:
    continue
```

So your feature dataframe will have slightly fewer rows than the label dataframe (e.g., 28,650 instead of 28,657 if 7 windows had all corrupt clips). Check:
```python
print(f"Labels: {len(master_df)}")
print(f"Features: {len(feature_df)}")
```

If the gap is large (>1%), there's likely a data issue worth investigating.

---

## Appendix: Feature List (Full)

Per window, after aggregation across clips:

```
Zero-crossing rate:
  zcr_mean, zcr_std

RMS energy:
  rms_mean_mean, rms_mean_std
  rms_std_mean, rms_std_std
  rms_max_mean, rms_max_std

Spectral centroid:
  centroid_mean_mean, centroid_mean_std
  centroid_std_mean, centroid_std_std

Spectral bandwidth:
  bandwidth_mean_mean, bandwidth_mean_std

Spectral rolloff:
  rolloff_mean_mean, rolloff_mean_std

Spectral flatness:
  flatness_mean_mean, flatness_mean_std

Spectral contrast (7 bands):
  contrast_1_mean_mean, contrast_1_mean_std
  contrast_2_mean_mean, contrast_2_mean_std
  ... (through contrast_7)

Chroma:
  chroma_mean_mean, chroma_mean_std

MFCCs (13 coefficients):
  mfcc_1_mean_mean, mfcc_1_mean_std, ..., mfcc_13_mean_mean, mfcc_13_mean_std

Delta-MFCCs (13 coefficients):
  mfcc_delta_1_mean_mean, mfcc_delta_1_mean_std, ..., mfcc_delta_13_mean_std

Metadata (not aggregated):
  rainfall_mm, timestamp, wav_count, rain (binary), n_clips_used
```

**Total**: ~130 features + 5 metadata/label columns = 135–140 columns per window.
