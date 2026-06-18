# 📚 CNN & LSTM — Study Guide for the Acoustic Rain Gauge

---

## Part 1: Convolutional Neural Networks (CNN)

### 1.1 What Problem Does a CNN Solve?

A regular Dense (fully-connected) layer treats every input pixel as independent. For a `1025×2672` spectrogram, that's **2.7 million** inputs — and a Dense layer would need billions of weights.

More importantly, **spatial relationships matter**. A raindrop pattern in the spectrogram at position (x, y) is just as meaningful as the same pattern at (x+10, y+5). Dense layers can't exploit this.

CNNs solve this with two key ideas:
- **Local connectivity** — each neuron only "sees" a small patch (kernel)
- **Weight sharing** — the same kernel slides over the entire input

---

### 1.2 The Convolution Operation

A kernel (filter) is a small matrix of learnable weights. It slides (convolves) over the input, computing a dot product at each position.

```
Input patch (3×3):        Kernel (3×3):         Output value:
┌───┬───┬───┐            ┌───┬───┬───┐
│ 1 │ 2 │ 3 │            │ 0 │ 1 │ 0 │
├───┼───┼───┤     *      ├───┼───┼───┤   →   (1×0 + 2×1 + ... ) = scalar
│ 4 │ 5 │ 6 │            │ 1 │ 0 │ 1 │
├───┼───┼───┤            ├───┼───┼───┤
│ 7 │ 8 │ 9 │            │ 0 │ 1 │ 0 │
└───┴───┴───┘            └───┴───┴───┘
```

Sliding this kernel over every position of the input produces a **feature map** — one scalar per position showing how strongly that kernel's pattern was found there.

**Multiple kernels = multiple feature maps** (channels). Each kernel learns to detect a different pattern: edges, textures, frequency bands, etc.

---

### 1.3 Activation (ReLU)

After convolution, apply `ReLU(x) = max(0, x)`.

- Kills negative responses (pattern NOT present)
- Keeps positive responses (pattern IS present)
- Introduces non-linearity so the network can learn complex relationships

---

### 1.4 Pooling

After convolution, the feature map is still large. **MaxPooling** downsamples it by taking the maximum value in each small region:

```
Feature map (4×4):          After MaxPool(2×2):
┌───┬───┬───┬───┐           ┌───┬───┐
│ 1 │ 3 │ 2 │ 4 │           │ 3 │ 4 │
├───┼───┼───┼───┤    →      ├───┼───┤
│ 5 │ 2 │ 1 │ 6 │           │ 5 │ 6 │
├───┼───┼───┼───┤           └───┴───┘
│ 3 │ 1 │ 4 │ 2 │
├───┼───┼───┼───┤
│ 2 │ 5 │ 3 │ 1 │
└───┴───┴───┴───┘
```

Why max? Because "was this feature present anywhere in this region?" is more useful than "what was the average response?".

**Effect:** Reduces spatial size, builds **translation invariance** (a pattern slightly shifted still gets detected), and reduces computation.

---

### 1.5 Stacking Layers — Hierarchy of Features

```
Layer 1: Detects low-level patterns (edges, frequency blobs)
    ↓
Layer 2: Combines edges into textures (rhythmic rain patterns)
    ↓
Layer 3: Combines textures into high-level concepts (heavy rain vs drizzle)
```

This hierarchy is why deep CNNs work — each layer abstracts the previous.

---

### 1.6 CNN for Images vs Spectrograms

In image tasks, a CNN detects visual objects. Here, the "image" is an **STFT spectrogram**:

```
Axis              Meaning
──────────────────────────────────────────
Vertical (y)      Frequency (0 Hz → 4000 Hz, 1025 bins)
Horizontal (x)    Time (0s → ~171s, 2672 frames)
Pixel value       Magnitude |Z(f,t)| — energy at that freq/time
```

Rain has a characteristic **spectral fingerprint**:
- Heavy rain → broadband high-energy across many frequencies
- Light drizzle → lower energy, concentrated in mid frequencies
- Silence → near-zero magnitude everywhere

The CNN learns to **read these patterns** just like it would read a face in a photo.

---

## Part 2: Long Short-Term Memory (LSTM)

### 2.1 Why Not Just Use Dense for Sequences?

Dense layers have no memory. Given a sequence `[x₁, x₂, x₃, ...]`, they treat each step independently. But rainfall intensity at time `t` might depend on what happened at `t-1`, `t-2`, etc.

A standard RNN (Recurrent Neural Network) tries to solve this with a hidden state `h`:

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

But standard RNNs suffer from the **vanishing gradient problem**: gradients shrink exponentially when backpropagating through many timesteps. The network "forgets" long-range dependencies.

---

### 2.2 LSTM — The Solution

LSTM adds a **cell state** `C_t` (like a conveyor belt of memory) alongside the hidden state `h_t`. Three learnable **gates** control information flow:

```
┌─────────────────────────────────────────────────────────┐
│                        LSTM Cell                        │
│                                                         │
│  x_t ──┬──────────────────────────────────────────┐    │
│         │                                          │    │
│  h_{t-1}┤                                          │    │
│         │                                          │    │
│         ▼                                          ▼    │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐  ┌──────┐ │
│   │ Forget   │   │  Input   │   │  Cell    │  │Output│ │
│   │  Gate    │   │  Gate    │   │  Update  │  │ Gate │ │
│   │ σ(W_f·x) │   │ σ(W_i·x) │   │tanh(W_c·x)│  │σ(W_o)│ │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘  └──┬───┘ │
│        │              │              │             │     │
│        ▼              ▼              ▼             │     │
│   C_{t-1} ──×────── + ────×─────► C_t ──tanh──×──┘     │
│                                              ▲           │
│                                              h_t ────────►│
└─────────────────────────────────────────────────────────┘
```

| Gate | Formula | Purpose |
|---|---|---|
| **Forget** | `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` | What fraction of old cell state to keep (0=forget, 1=keep) |
| **Input** | `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` | What new information to write to cell state |
| **Cell update** | `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)` | Candidate values to add |
| **Output** | `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)` | What part of cell state to expose as output |

**Cell state update:**
```
C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ C̃_t
       ↑                    ↑
  (forget old)         (write new)
```

**Hidden state output:**
```
h_t = o_t ⊙ tanh(C_t)
```

The sigmoid (σ) outputs values in [0,1] — perfect for "how much" of something to pass through. The tanh outputs [-1,1] — good for the actual values.

---

### 2.3 LSTM in Practice — Sequence Learning

```
Input sequence:    [x_1,  x_2,  x_3,  ..., x_T]
                      ↓     ↓     ↓          ↓
LSTM processes:   h_1 → h_2 → h_3 → ... → h_T  ← final output
```

The final `h_T` is a compressed representation of the entire sequence, with memory of long-range dependencies.

**Classic LSTM use cases:**
- Time series forecasting (stock prices, sensor readings)
- Natural language processing (each word is a timestep)
- Speech recognition (each audio frame is a timestep)
- Music generation (each note is a timestep)

---

## Part 3: CNN + LSTM — The Hybrid Architecture

### 3.1 Why Combine Them?

| Model | Good at | Weak at |
|---|---|---|
| CNN | Spatial patterns, local features | Temporal/sequential dependencies |
| LSTM | Sequential memory, long-range time dependencies | High-dimensional spatial features |
| **CNN + LSTM** | **Both** | — |

The typical design: **CNN acts as a feature extractor → LSTM learns temporal dynamics over those features**.

---

### 3.2 The Classic Pattern

```
Raw Input (sequence of frames)
        │
        ▼
  ┌─────────────┐
  │    CNN      │  ← Extracts spatial features from each frame
  │ (per frame) │
  └──────┬──────┘
         │ Feature vector per timestep
         ▼
  ┌─────────────┐
  │    LSTM     │  ← Learns temporal patterns across timesteps
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Dense Head │  ← Final prediction (classification or regression)
  └─────────────┘
```

**Example:** Video classification
- CNN processes each frame → gets a feature vector
- LSTM sees a sequence of feature vectors → understands motion over time

---

### 3.3 Real-World CNN+LSTM Applications

| Domain | Input | CNN extracts | LSTM learns |
|---|---|---|---|
| Video analysis | Frame sequence | Objects per frame | Motion across frames |
| Audio classification | Spectrogram | Spectral patterns | Temporal evolution |
| Weather forecasting | Satellite images | Spatial weather patterns | How patterns evolve |
| ECG analysis | 1D signal in windows | Local waveform shapes | Rhythm over time |
| Activity recognition | Accelerometer windows | Motion features | Activity transitions |

---

## Part 4: How This Project Uses CNN + LSTM

### 4.1 The Full Signal Flow

```
18 × 10s WAV clips (at 8000 Hz)
        │
        ▼ librosa.load() + np.append()
Combined 171s audio array: shape (1,368,000,)
        │
        ▼ audio[:seq_len] — trim to exact length
        │
        ▼ librosa.stft() — Short-Time Fourier Transform
STFT output: shape (1025, 2672) complex
        │
        ▼ np.abs(Zxx) — magnitude spectrum
Spectrogram: shape (1025, 2672) float
        │
        ▼ [np.newaxis, :, :] — add batch dimension
Model input: shape (1, 1025, 2672)  ← treated as (batch=1, H=1025, W=2672, C=1)
        │
        ▼ CNN layers
        │
        ▼ LSTM
        │
        ▼ Dense(1)
Output: scalar mm̂ (rainfall in millimetres)
```

---

### 4.2 Why STFT? The Physics of Rain Sound

The STFT converts time-domain audio into a **time-frequency representation**:

```
y-axis (frequency):
4000 Hz ─────────────────────────────────────────
         Heavy rain: high energy across ALL frequencies
         ████████████████████████████████████████
         
2000 Hz ─────────────────────────────────────────
         Drizzle: energy concentrated in mid frequencies
         ░░░░████████████░░░░░░░░░░░░░░░░░░░░░░░
         
 500 Hz ─────────────────────────────────────────
         Wind noise: energy concentrated in low frequencies
         ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   0 Hz ─────────────────────────────────────────
         t=0s                              t=171s
                    x-axis (time)
```

Rain intensity is correlated with **spectral energy distribution** — hence the CNN can learn to map spectral patterns to mm/3min.

---

### 4.3 CNN Layers in This Project

```python
model.add(Input((1025, 2672, 1)))
```
Input: STFT spectrogram as a single-channel "image"

```python
model.add(Conv2D(64, kernel_size=(8, 8), activation="relu"))
model.add(MaxPooling2D(pool_size=(8, 8)))
```
**Block 1:** 64 filters, each 8×8.
- Output shape after conv: `(1018, 2665, 64)` — 64 feature maps
- After MaxPool(8×8): `(127, 333, 64)`
- Learns: Low-level spectral blobs, energy concentrations

```python
model.add(Conv2D(32, kernel_size=(4, 4), activation="relu"))
model.add(MaxPooling2D(pool_size=(4, 4)))
```
**Block 2:** 32 filters, each 4×4.
- After conv: `(124, 330, 32)`
- After MaxPool(4×4): `(31, 82, 32)`
- Learns: Mid-level patterns — frequency band combinations

```python
model.add(Conv2D(16, kernel_size=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
```
**Block 3:** 16 filters, each 2×2.
- After conv: `(30, 81, 16)`
- After MaxPool(2×2): `(15, 40, 16)` = **9,600 values**
- Learns: High-level abstractions of rain intensity

> **Why decreasing filter counts (64→32→16)?**
> Lower layers need many filters to capture diverse low-level features. Higher layers combine these into fewer, more abstract concepts. This is standard CNN design practice.

---

### 4.4 The Reshape + LSTM — A Critical Design Quirk

```python
model.add(Reshape((1, -1)))   # (15, 40, 16) → (1, 9600)
model.add(LSTM(20))           # processes sequence of length 1
```

⚠️ **This is the architectural quirk I mentioned in the analysis.**

The `Reshape((1, -1))` collapses all 9,600 CNN features into **a single timestep** of length 9,600. An LSTM seeing a sequence of length 1 has **nothing to be sequential over** — it processes exactly one step.

In this configuration, the LSTM behaves mathematically like:

```
h = o ⊙ tanh(i ⊙ C̃)   where C_{t-1} = 0 (no prior state)
```

This is essentially equivalent to:
```python
Dense(20, activation='tanh')  # ← approximately what it's doing
```

**Why it still works:** The LSTM cell's gating mechanism still acts as a learned non-linear projection from 9,600 → 20 dimensions. It just doesn't use its sequential memory capability.

**What a correct CNN+LSTM for audio would look like:**
```python
# Process audio in windows, each window → CNN → feature vector
# Then LSTM over the sequence of window features
model.add(TimeDistributed(Conv2D(...)))   # CNN per time window
model.add(LSTM(64))                       # LSTM over time windows
model.add(Dense(1))
```
This would let the LSTM learn "rain was building up over the first minute, then heavy in the last two minutes → estimate X mm."

---

### 4.5 Dense Regression Head

```python
model.add(Dense(32))    # 20 → 32 learned features
model.add(Dense(16))    # 32 → 16 refined features
model.add(Dense(1))     # 16 → 1 rainfall estimate (mm)
```

No activation on the final Dense(1) — this makes it a **linear regression output**, appropriate for predicting a continuous value (mm).

The intermediate Dense layers act as a learned non-linear mapping from LSTM's compressed representation to the final prediction.

---

## Part 5: STFT — The Bridge Between Audio and CNN

### 5.1 What is STFT?

The **Short-Time Fourier Transform** slides a window over the signal and computes the FFT within each window:

```
Audio signal:  ────────────────────────────────────
               [──window──]
                    [──window──]
                         [──window──]
                              ...
```

For each window position, you get a **frequency spectrum**. Stack them side by side → spectrogram.

**librosa defaults (used in this project):**
| Parameter | Default | Effect |
|---|---|---|
| `n_fft` | 2048 | FFT window size → `2048/2+1 = 1025` frequency bins |
| `hop_length` | 512 | Window step → `8000/512 ≈ 15.6` frames/sec |
| Frames for 171s | `171 × 15.6 ≈ 2667` | ≈ 2672 time frames |

This is why the spectrogram is `(1025, 2672)`.

### 5.2 Why Magnitude Only?

```python
stft_sample = np.abs(Zxx)   # Take magnitude, discard phase
```

`librosa.stft()` returns **complex numbers** `Z = a + jb` (amplitude + phase). `|Z| = √(a² + b²)` is the magnitude — how much energy is at that frequency.

For rain intensity estimation, **phase is irrelevant** — what matters is how much energy is in each frequency band, not the fine timing of the wave cycles.

---

## Part 6: Summary Comparison

| Concept | CNN | LSTM | CNN+LSTM (this project) |
|---|---|---|---|
| **Input type** | Grid/image data | Sequences | Image treated as single spectrogram |
| **Core operation** | Sliding kernel convolution | Gated recurrent state update | CNN features → LSTM projection |
| **What it learns** | Spatial patterns, textures | Temporal dependencies, trends | Spectral fingerprint of rain |
| **Memory** | None (stateless) | Explicit (cell state) | Effectively none (1 timestep) |
| **Weights** | Shared across spatial positions | Shared across time steps | — |
| **Output here** | Feature map (9600 values) | 20-dim vector | Scalar mm |
| **Training signal** | Backprop through conv layers | Backprop through time (BPTT) | Combined backprop |

---

## Part 7: How to Improve the Model

Given the architectural quirk, here are directions if you want to improve the model:

### Option A: Fix the LSTM (proper temporal modeling)
```python
# Split the 171s window into 18 sub-windows (one per WAV clip)
# Process each 10s clip with CNN → get feature vector per clip
# Feed sequence of 18 feature vectors to LSTM

model = Sequential()
model.add(TimeDistributed(Conv2D(64, (4,4), activation='relu'),
          input_shape=(18, 113, 148, 1)))   # 18 clips, each STFT ~113×148
model.add(TimeDistributed(MaxPooling2D((4,4))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

### Option B: Replace LSTM with GlobalAveragePooling
If temporal ordering isn't important, just globally pool the CNN output:
```python
model.add(GlobalAveragePooling2D())  # instead of Reshape + LSTM
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
```

### Option C: Try a regression CNN without LSTM
Similar to what VGG/ResNet do for image regression tasks — keep the CNN deep and finish with Dense layers directly.

### Option D: 1D CNN over frequency axis
Since the y-axis represents frequency and rain patterns span frequency ranges:
```python
# Process column by column (each time frame as a sequence of frequency bins)
model.add(Conv1D(64, 8, activation='relu'))  # along frequency axis
model.add(LSTM(32, return_sequences=True))   # along time axis
```

---

> [!TIP]
> The current model (`seq_stft_enc3.hdf5`) **works** despite the LSTM quirk because the CNN features are rich enough that even a single-step LSTM projection produces a useful regression. But fixing it to use LSTM properly over 18 timesteps would likely improve accuracy on dynamic rain events.

> [!NOTE]
> For further reading:
> - **CNNs:** "A Guide to Convolutional Neural Networks" — CS231n (Stanford)
> - **LSTMs:** "Understanding LSTMs" — Christopher Olah's blog (colah.github.io)
> - **STFT for ML:** "Librosa: Audio and Music Analysis in Python"
> - **CNN+LSTM:** "Convolutional LSTM Network" — Shi et al. 2015 (the paper that introduced ConvLSTM)
