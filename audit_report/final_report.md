MODEL 1:

### 1. Class Distribution

No Rain        23529 (76.9%)
Light Rain      6823 (22.3%)
Heavy Rain       243 (0.8%)

This shows how many training samples belong to each class.

Nearly 77% of the recordings contain no rain.
About 22% contain light rain.
Less than 1% contain heavy rain.

This is known as an imbalanced dataset.

An ML model naturally tries to minimize total errors. Since predicting "No Rain" is correct most of the time, the model becomes biased toward that class.

This imbalance is the biggest challenge of this project.

### 2. StandardScaler

The extracted audio features have different numerical ranges.

For example,

ZCR may vary from 0–0.3
MFCC values may range from -200 to 80
Spectral Contrast has another scale

StandardScaler converts every feature into approximately

mean = 0
standard deviation = 1

This prevents large-valued features from dominating smaller-valued ones.

The scaler is saved so that future unseen audio is transformed exactly the same way before prediction.

### 3. Class Weights

No Rain     : 0.433
Light Rain  : 1.495
Heavy Rain  : 41.968

Since Heavy Rain is extremely rare, XGBoost is instructed to care much more when it misclassifies those samples.

The weight values mean

Mistake on No Rain counts only 0.43x
Mistake on Light Rain counts about 1.5x
Mistake on Heavy Rain counts nearly 42 times more

Without this weighting the model would almost completely ignore Heavy Rain.

### 4. Optuna Hyperparameter Optimization

60 trials

Instead of manually choosing XGBoost parameters, Optuna searched automatically.

It tested 60 different combinations of

number of trees
tree depth
learning rate
sampling ratios
regularization
etc.

and selected the best one based on cross-validation Macro-F1

### 5. Best Hyperparameters

Example:

n_estimators = 518
max_depth = 6
learning_rate = 0.088

# Meaning

n_estimators = 518:
    The model consists of 518 decision trees.
    More trees generally improve learning but increase training time.

max_depth = 6:
    Each decision tree can grow to depth 6.
    This is a moderate depth that captures complex relationships without severe overfitting.

learning_rate = 0.088:
    Each tree only makes a small correction.
    Small learning rates usually improve generalization.

subsample = 0.94: 
    Each tree is trained using about 94% of the training samples.
    This reduces overfitting.

colsample_bytree = 0.96:
    Each tree only sees 96% of the features.
    Again helps improve robustness.

min_child_weight = 9:
    Prevents trees from growing based on very small groups of samples.
    Helps reduce noise.

gamma:
    Controls how difficult it is to create new tree splits.
    Higher values make trees simpler.

reg_alpha:
    L1 regularization.
    Encourages sparse trees.

reg_lambda:
    L2 regularization.
    Reduces overfitting.

Overall, these values indicate Optuna found a relatively conservative model that prioritizes generalization over memorization.

### 6. Classification Report

# No Rain

Precision = 0.76
Recall    = 0.90
F1        = 0.82

Precision:
    "When the model predicts No Rain, how often is it correct?"
    76%

Recall:
    "Out of all actual No Rain samples, how many did it find?"
    90%

This is very good.

F1-score:
    Balances Precision and Recall.
    0.82 indicates good performance.

# Light Rain

Precision = 0.16
Recall    = 0.07
F1        = 0.09

This is the weakest class.
The model misses most light rain events.
Only 7% of actual light rain samples are correctly identified.

Reasons include

sounds resemble no rain
low rainfall intensity
class imbalance
limited training data

# Heavy Rain

Precision = 0.56
Recall = 0.59
F1 = 0.58

These results are actually encouraging.
Although only 243 samples exist,
the model still detects about 59%
of them.

That suggests heavy rainfall has distinctive acoustic characteristics.

### 7. Accuracy

71%

Accuracy means

correct predictions
-------------------
total predictions

Although 71% appears decent, it is misleading because of class imbalance.
A model predicting only "No Rain" would already achieve around 77% accuracy.
Therefore, accuracy is not the most informative metric for this problem.

### 8. Balanced Accuracy

0.5186

Balanced Accuracy calculates the average recall across all classes.
This prevents the majority class from dominating the score.
A value of 0.5 is only slightly better than random guessing across the three classes, indicating that minority classes remain challenging.

### 9. Macro F1

0.4986

Macro-F1 computes the F1-score for each class independently and averages them.
Each class contributes equally.
This is one of the best metrics for imbalanced classification.
It shows the overall performance is moderate but significantly affected by the Light Rain class.

### 10. Feature Importance

The model identified the most useful audio features.

Top features include

zcr_mean
mfcc_11_std_mean
flatness_mean_std
zcr_std
MFCCs
Spectral Contrast
RMS
Chroma

# Zero Crossing Rate (ZCR)

Most important feature.
Measures how often the waveform changes sign.
Rain produces many rapid waveform fluctuations.

# MFCCs

MFCCs describe the spectral shape of audio.
They are standard features in
    speech recognition
    environmental sound classification
    acoustic event detection

Multiple MFCC coefficients appear among the most important features, indicating they effectively capture differences between rain intensities.

# Spectral Flatness

Indicates whether a sound is noise-like or tone-like.
Rain behaves similarly to broadband noise, making this feature useful.

# RMS Energy

Represents the loudness of the recording.
Heavy rainfall typically produces higher energy levels.

# Spectral Contrast

Measures differences between spectral peaks and valleys.
Useful for distinguishing rainfall from background environmental sounds.

# Chroma

Usually associated with music
Although less intuitive for rainfall, it still contributes useful information according to the model.

### Overall Interpretation

The extracted audio features clearly contain meaningful information.

The model successfully learned

characteristics of rain sounds
characteristics of heavy rainfall
differences between rainfall and background noise

The main weakness lies in detecting light rain.

This is primarily caused by

very small Heavy Rain class
dominant No Rain class
overlap between No Rain and Light Rain acoustic signatures

Rather than indicating a flaw in the model, these results highlight the limitations imposed by the available data.


# Internship Project Report

## Title

**Development of a Machine Learning-Based Acoustic Rain Gauge Using Environmental Audio Signals**

---

# 1. Introduction

Rainfall measurement is an essential component of meteorological monitoring, agriculture, disaster management, and climate research. Conventional rain gauges require dedicated hardware, regular maintenance, and installation costs, making large-scale deployment challenging.

The objective of this project is to investigate whether rainfall intensity can be estimated using only environmental sound recordings captured by a low-cost microphone. By leveraging machine learning techniques, the project aims to classify rainfall conditions based on acoustic characteristics extracted from recorded audio.

The project is being carried out using real-world rainfall recordings collected through Raspberry Pi-based acoustic monitoring devices together with measurements obtained from a mechanical tipping bucket rain gauge.

---

# 2. Aim of the Project

The primary aim of this project is to design and develop an intelligent acoustic rain gauge capable of identifying rainfall intensity from environmental audio recordings.

The proposed system should:

* Detect the presence or absence of rainfall.
* Classify rainfall into different intensity levels.
* Operate using inexpensive microphone-based hardware.
* Serve as a low-cost alternative or supplementary solution to conventional rain gauges.
* Demonstrate the feasibility of machine learning for acoustic rainfall estimation.

---

# 3. Objectives

The project objectives include:

* Collect and organize rainfall audio recordings.
* Synchronize rainfall measurements with corresponding audio clips.
* Perform preprocessing and dataset cleaning.
* Extract discriminative acoustic features from audio signals.
* Train machine learning models for rainfall classification.
* Evaluate model performance using suitable metrics.
* Identify limitations and potential improvements for future work.

---

# 4. Dataset Description

The project utilizes two synchronized data sources:

### Acoustic Data

* Audio recordings collected using Raspberry Pi-based microphones.
* Sampling rate standardized to 8 kHz.
* Audio segmented into 3-second clips for model training.

### Rainfall Labels

* Mechanical tipping bucket rain gauge measurements.
* Rainfall values recorded at regular intervals.
* Used as ground truth labels for supervised learning.

The synchronized dataset consists of more than 30,000 labelled audio samples collected over approximately 140 recording days spanning multiple months.

The dataset includes recordings representing:

* No Rain
* Light Rain
* Heavy Rain

---

# 5. Work Completed During the Internship

## 5.1 Data Collection and Organization

The initial phase involved organizing a large collection of environmental recordings obtained over multiple years.

Activities performed:

* Organizing recordings month-wise.
* Cleaning folder structures.
* Removing corrupted audio files.
* Merging datasets collected across different recording periods.
* Uploading datasets to Kaggle for cloud-based processing.

---

## 5.2 Timestamp Alignment

Since rainfall measurements and audio recordings were collected independently, timestamp synchronization was required.

The alignment process involved:

* Parsing timestamps from audio filenames.
* Parsing timestamps from mechanical rainfall logs.
* Matching audio recordings with rainfall measurements using temporal windows.
* Assigning rainfall labels to corresponding audio clips.

This produced the final labelled dataset used for model development.

---

## 5.3 Dataset Cleaning

Several preprocessing steps were performed to improve dataset quality.

These included:

* Removal of duplicate samples.
* Handling missing values.
* Verification of timestamp consistency.
* Validation of rainfall labels.
* Dataset auditing.
* Generation of statistical summaries.
* Conversion into efficient Parquet format.

---

## 5.4 Exploratory Data Analysis (EDA)

Extensive analysis was carried out to understand dataset characteristics.

The analysis included:

* Rainfall distribution
* Class imbalance
* Recording duration
* Recording days
* Rainfall intensity statistics
* Audio availability
* Monitoring duration
* Timestamp analysis

The study revealed a significant class imbalance, with No Rain samples dominating the dataset and Heavy Rain representing less than 1% of all observations.

---

## 5.5 Audio Feature Engineering

A comprehensive feature extraction pipeline was developed to convert raw audio into numerical representations suitable for machine learning.

Extracted features include:

### Time-domain Features

* Root Mean Square (RMS)
* Zero Crossing Rate (ZCR)

### Frequency-domain Features

* Mel Frequency Cepstral Coefficients (MFCCs)
* Spectral Centroid
* Spectral Bandwidth
* Spectral Contrast
* Spectral Flatness
* Spectral Rolloff

### Harmonic Features

* Chroma Features

For every feature, statistical descriptors such as mean and standard deviation were computed across multiple audio clips to generate fixed-length feature vectors.

The final dataset consists of hundreds of numerical acoustic features representing each labelled recording.

---

## 5.6 Dataset Preparation

After feature extraction:

* Features were standardized using StandardScaler.
* Labels were encoded into three rainfall classes.
* Monthly feature files were stored as Parquet datasets.
* Feature datasets were combined into a master training dataset.

---

## 5.7 Machine Learning Model Development

Several machine learning experiments were performed using XGBoost.

The workflow included:

* Time-series cross-validation.
* Class-weight balancing.
* Hyperparameter optimization using Optuna.
* Feature importance analysis.
* Model evaluation.

Optuna performed automated hyperparameter optimization over 60 trials to maximize Macro F1-score while preserving chronological ordering of the dataset.

---

## 5.8 Model Performance

The final classifier achieved:

* Accuracy: 71%
* Balanced Accuracy: 51.86%
* Macro F1-score: 49.86%

Performance analysis showed:

* Strong performance for No Rain detection.
* Moderate detection of Heavy Rain despite very limited training samples.
* Poor performance for Light Rain due to significant overlap with environmental background sounds.

Feature importance analysis indicated that Zero Crossing Rate, MFCCs, Spectral Contrast, Spectral Flatness, RMS Energy, and Chroma features contributed most significantly to classification.

---

# 6. Challenges Encountered

Several practical challenges were encountered during the project.

## Severe Class Imbalance

Approximately:

* 76.9% No Rain
* 22.3% Light Rain
* 0.8% Heavy Rain

This imbalance limited the model's ability to learn minority rainfall classes effectively.

---

## Limited Heavy Rain Samples

Heavy rainfall events were extremely scarce, making reliable learning difficult.

---

## Acoustic Similarity

Light rainfall often exhibits acoustic characteristics similar to ambient environmental noise, reducing classification performance.

---

## Environmental Noise

Recordings include:

* Wind
* Human activity
* Vehicle sounds
* Birds
* Insects

These introduce unwanted variability into the extracted features.

---

## Large Dataset Processing

Processing tens of thousands of audio clips required optimization of storage, feature extraction, and cloud-based workflows using Kaggle.

---

# 7. Current Status

At the completion of the internship, the following milestones have been achieved:

✔ Dataset collection completed.

✔ Dataset synchronization completed.

✔ Dataset cleaning completed.

✔ Exploratory data analysis completed.

✔ Feature extraction pipeline completed.

✔ Machine learning model developed.

✔ Hyperparameter optimization completed.

✔ Performance evaluation completed.

✔ Feature importance analysis completed.

✔ Baseline acoustic rainfall classifier successfully established.

---

# 8. Future Work

Several improvements have been identified to enhance system performance.

## Improved Audio Preprocessing

* Spectral subtraction for noise removal.
* Audio denoising.
* Signal normalization.

---

## Additional Features

Future experiments will include:

* Log-Mel Spectrogram statistics.
* Delta and Delta-Delta MFCCs.
* Temporal energy descriptors.
* Wavelet-based features.
* Advanced spectral descriptors.

---

## Deep Learning Models

Future work may investigate:

* CNNs using spectrogram images.
* CRNN architectures.
* Transformer-based audio classification.
* Self-supervised audio representations.

---

## Dataset Expansion

Future improvements require:

* Collection of additional rainfall recordings.
* More Heavy Rain samples.
* Increased seasonal diversity.
* Recordings under different environmental conditions.

---

## Model Improvement

Potential techniques include:

* Ensemble learning.
* Synthetic minority oversampling.
* Focal Loss.
* Cost-sensitive learning.
* Feature selection.
* Model stacking.

---

## Real-Time Deployment

The long-term objective is deployment on Raspberry Pi hardware for real-time rainfall monitoring using live microphone input.

---

# 9. Conclusion

During this internship, a complete machine learning pipeline for acoustic rainfall classification was successfully developed. The project involved large-scale data organization, timestamp synchronization, preprocessing, feature extraction, exploratory analysis, supervised learning, hyperparameter optimization, and model evaluation.

The experimental results demonstrate that rainfall possesses distinctive acoustic characteristics that can be learned using machine learning techniques. Although the current classifier performs well for identifying No Rain conditions and reasonably detects Heavy Rain events, the classification of Light Rain remains challenging due to class imbalance and acoustic similarity with background noise.

Nevertheless, the developed system establishes a strong baseline for future research on low-cost acoustic rainfall monitoring. Further improvements through enhanced preprocessing, richer feature representations, larger datasets, and deep learning techniques are expected to significantly improve performance and move the project closer to real-world deployment.