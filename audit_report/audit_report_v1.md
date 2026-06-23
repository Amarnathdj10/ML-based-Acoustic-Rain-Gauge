# Acoustic Rain Gauge Dataset Audit Report

## Project

ML-Based Acoustic Rain Gauge

## Objective

To audit the aligned rainfall-audio dataset before feature extraction and model development. The audit focuses on dataset integrity, timestamp alignment quality, rainfall label quality, audio coverage consistency, duplicate records, and rainfall distribution characteristics.

---

# 1. Dataset Consolidation

All aligned pickle files were loaded and merged into a single master dataset.

## Result

| Metric        |  Value |
| ------------- | -----: |
| Total Samples | 28,657 |
| Total Columns |      4 |

Columns present:

* timestamp
* rainfall_mm
* wav_count
* wav_files

Additional audit column:

* source_pickle

---

# 2. Rainfall Presence Distribution

## Results

| Category       | Samples | Percentage |
| -------------- | ------: | ---------: |
| No Rain (0 mm) |  21,651 |     75.55% |
| Rain (>0 mm)   |   7,006 |     24.45% |

## Observation

The dataset exhibits moderate class imbalance.

Ratio:

No Rain : Rain ≈ 3 : 1

This imbalance is considered manageable for machine learning applications.

---

# 3. Audio Alignment Audit

The alignment process associates rainfall measurements with audio recordings captured within the corresponding time window.

## WAV Count Statistics

| Metric  | Value |
| ------- | ----: |
| Mean    | 17.58 |
| Std Dev |  4.24 |
| Minimum |    17 |
| Maximum |    59 |

## WAV Count Distribution

| WAV Count | Samples |
| --------- | ------: |
| 17        |  24,641 |
| 18        |   3,699 |
| 57        |     164 |
| 58        |     145 |
| 59        |       8 |

## Observation

Approximately 98.9% of samples contain 17–18 audio clips.

This matches expectations for datasets containing 10-second audio recordings:

180 seconds / 10 seconds ≈ 18 clips

Alignment quality is therefore considered highly reliable.

---

# 4. Multi-Duration Audio Investigation

Rows with wav_count values between 57 and 59 were initially flagged as anomalies.

Investigation showed these rows originated from:

* december_2023_aligned_dataset.pkl
* january_2024_aligned_dataset.pkl

These datasets contain 3-second audio clips.

Expected clips in a 3-minute interval:

180 seconds / 3 seconds = 60 clips

Observed values:

57–59 clips

## Conclusion

No alignment issue exists.

The dataset contains two valid audio recording schemes:

| Audio Duration | Expected Clips |
| -------------- | -------------: |
| 10 seconds     |          17–18 |
| 3 seconds      |          57–60 |

---

# 5. Rainfall Label Quality Audit

## Extreme Rainfall Investigation

The following rainfall values were detected:

655.09 mm
655.15 mm
655.20 mm
655.23 mm
655.24 mm
655.26 mm
655.28 mm
655.30 mm
655.32 mm
655.33 mm

Affected datasets:

* october_2025_aligned_dataset.pkl
* may_2025_aligned_dataset.pkl
* feb_to_march_2026_aligned_dataset.pkl

## Observation

A large discontinuity exists between:

Maximum valid rainfall: 21.60 mm

and

Next largest value: 655.09 mm

No rainfall values exist between 21.60 mm and 655.09 mm.

These values are highly likely to represent:

* Sensor overflow values
* Sentinel values
* Data conversion artifacts
* Corrupted labels

## Recommendation

Remove all samples where:

rainfall_mm > 100

These records represent a negligible proportion of the dataset and are not physically realistic.

---

# 6. Duplicate Timestamp Audit

Duplicate timestamps were investigated across the consolidated dataset.

## Findings

A total of 118 duplicate rows were identified.

Affected dataset:

* september_2024_aligned_dataset.pkl

Example:

Timestamp duplicated with identical:

* timestamp
* rainfall label
* source dataset

## Conclusion

Duplicates likely originated from:

* Source CSV duplication
  or
* Data preprocessing duplication

These rows provide no additional information and should be removed.

## Recommendation

Remove duplicated rows using:

(timestamp, source_pickle)

as the uniqueness criteria.

---

# 7. Per-Dataset Statistics

| Dataset           | Samples | Rain Samples | Rain % | Max Rain (mm) |
| ----------------- | ------: | -----------: | -----: | ------------: |
| december_2024     |   7,946 |           30 |  0.38% |          8.60 |
| june_2026         |   4,959 |          259 |  5.22% |          5.99 |
| june_2025         |   3,230 |          348 | 10.77% |          5.22 |
| feb_to_march_2026 |   3,068 |        3,066 | 99.93% |          4.68 |
| jan_2025          |   3,024 |          191 |  6.32% |          4.67 |
| may_2026          |   1,647 |           63 |  3.83% |          5.80 |
| august_2025       |   1,344 |        1,330 | 98.96% |          1.71 |
| may_2025          |     915 |          232 | 25.36% |          2.34 |
| october_2025      |     684 |          388 | 56.73% |          5.60 |
| may_2024          |     410 |          374 | 91.22% |         21.60 |
| april_2024        |     326 |          110 | 33.74% |          9.00 |
| july_2024         |     295 |           76 | 25.76% |          4.80 |
| december_2023     |     269 |          202 | 75.09% |          2.40 |
| september_2024    |     266 |          171 | 64.29% |          9.80 |
| november_2023     |     155 |           95 | 61.29% |          8.00 |
| january_2024      |      48 |           11 | 22.92% |          0.40 |

## Observations

The dataset spans a wide variety of rainfall conditions.

Some months are predominantly dry:

* December 2024
* June 2026
* January 2025

Others are predominantly rainy:

* February–March 2026
* August 2025
* May 2024

This diversity is beneficial for model development.

---

# 8. Rainfall Intensity Distribution

After excluding no-rain samples:

Total rain samples:

6,946

## Rainfall Range Distribution

| Range      | Samples | Percentage |
| ---------- | ------: | ---------: |
| 0 – 0.5 mm |   5,409 |      77.9% |
| 0.5 – 1 mm |   1,070 |      15.4% |
| 1 – 2 mm   |     244 |       3.5% |
| 2 – 5 mm   |     171 |       2.5% |
| > 5 mm     |      52 |       0.7% |

## Observation

The dataset is heavily dominated by light rainfall events.

This represents the primary modeling challenge.

---

# 9. Rainfall Descriptive Statistics

Rain samples only:

| Metric          |    Value |
| --------------- | -------: |
| Count           |    6,946 |
| Mean            |  0.59 mm |
| Std Dev         |  0.87 mm |
| Minimum         |  0.01 mm |
| Median          |  0.44 mm |
| 75th Percentile |  0.48 mm |
| 90th Percentile |  0.74 mm |
| 95th Percentile |  1.30 mm |
| 99th Percentile |  4.33 mm |
| Maximum         | 21.60 mm |

## Interpretation

* 50% of rain samples are ≤ 0.44 mm
* 75% of rain samples are ≤ 0.48 mm
* 95% of rain samples are ≤ 1.30 mm
* Only 1% exceed 4.33 mm

The target distribution is therefore strongly skewed toward light rainfall.

---

# 10. Key Findings

## Confirmed

* Dataset successfully consolidated
* Alignment quality is excellent
* Audio coverage is highly consistent
* Dataset contains both 3-second and 10-second audio schemes
* Temporal coverage spans November 2023 to June 2026
* Class imbalance is manageable

## Issues Identified

1. Invalid rainfall labels around 655.xx mm
2. Duplicate rows in September 2024 dataset
3. Strong rainfall intensity imbalance

---

# Recommended Cleaning Steps

1. Remove all rows where rainfall_mm > 100
2. Remove duplicated September 2024 records
3. Save cleaned dataset as:

   * master_df_clean.parquet
   * master_df_clean.pkl

---

# Recommended Next Phase

Proceed to feature engineering.

Potential acoustic features:

* MFCCs
* Mel Spectrogram Statistics
* RMS Energy
* Zero Crossing Rate
* Spectral Centroid
* Spectral Bandwidth
* Spectral Roll-off
* Spectral Contrast

Followed by:

* XGBoost Regression
* Random Forest Regression
* Deep Learning Models (CNN / CNN-LSTM)

---

# 11. Dataset Coverage Analysis

The consolidated aligned dataset contains:

| Metric                    |          Value |
| ------------------------- | -------------: |
| Total Samples             |         28,657 |
| Time Window per Sample    |      3 minutes |
| Total Monitoring Duration | 85,971 minutes |
| Total Monitoring Duration |  1,432.9 hours |
| Total Monitoring Duration |      59.7 days |


Months Available:

Nov 2023
Dec 2023

Apr 2024
May 2024
Jul 2024
Sep 2024
Dec 2024

Jan 2025
May 2025
Jun 2025
Aug 2025
Oct 2025

Feb–Mar 2026
May 2026
Jun 2026

# 12. Temporal Coverage Analysis

A distinction must be made between cumulative monitoring duration and the number of recording days represented in the dataset.

## Distinct Recording Days

Analysis of the timestamp field showed that recordings were collected on:

**136 distinct calendar days**

between:

**November 2023 and June 2026**

This indicates that data collection occurred across multiple recording campaigns spanning approximately 2.5 years.

## Cumulative Monitoring Duration

The aligned dataset contains:

* 28,657 samples
* Each sample represents a 3-minute rainfall observation window

Therefore:

28,657 × 3 minutes = 85,971 minutes

Equivalent to:

* 1,432.9 hours
* 59.7 days of cumulative monitoring duration

## Interpretation

The cumulative duration of approximately 60 days should not be interpreted as continuous recording over only 60 calendar days.

Instead:

* Data collection spans approximately 2.5 years
* Recordings were obtained on 136 separate calendar days
* The aligned samples collectively represent approximately 1,433 hours of rainfall monitoring windows

This distinction is important because the dataset captures rainfall events across multiple seasons, years, environmental conditions, and recording campaigns rather than representing a single continuous recording period.

## Significance

The combination of:

* 136 recording days
* 28,657 aligned samples
* Approximately 7,000 rain events
* Coverage from November 2023 to June 2026

provides substantial temporal diversity for rainfall estimation and acoustic modeling. The dataset includes observations from multiple monsoon periods, dry periods, and varying rainfall intensities, improving the potential robustness and generalizability of machine learning models trained on the data.


## Observation

The dataset provides approximately 1,433 hours of aligned rainfall monitoring data and approximately 1,364 hours of corresponding acoustic recordings collected between November 2023 and June 2026.

This represents a large-scale real-world acoustic rainfall dataset spanning multiple seasons, rainfall intensities, and environmental conditions. The temporal coverage and audio volume are considered sufficient for both traditional machine learning and deep learning approaches to rainfall estimation.

# Final Assessment

The dataset is of good overall quality and suitable for machine learning development after removal of a small number of invalid labels and duplicate records. The principal challenge for future modeling is the highly skewed rainfall intensity distribution, where nearly 78% of rain samples correspond to rainfall amounts below 0.5 mm.


