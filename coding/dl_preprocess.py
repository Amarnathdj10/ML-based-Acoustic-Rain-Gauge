"""
dl_preprocess.py — Acoustic Rain Gauge DL Preprocessing Pipeline
=================================================================
Converts raw WAV audio + mechanical CSV into Mel spectrograms (.npz)
ready for CNN / CNN-LSTM training.

Usage:
    # Preprocess a single month (Group A dataset):
    python dl_preprocess.py \
        --wav_dir   "D:/ARG_dataset/Jul_2024/audio" \
        --csv_path  "D:/ARG_dataset/Jul_2024/mech_data.csv" \
        --output    "D:/ARG_dataset/preprocessed/group_a/jul_2024.npz" \
        --label     "jul_2024"

    # Merge all monthly .npz files into train/val/test:
    python dl_preprocess.py \
        --merge \
        --npz_dir   "D:/ARG_dataset/preprocessed/group_a" \
        --out_dir   "D:/ARG_dataset/preprocessed/master"

Author: Amarnath DJ (ICFOSS Acoustic Rain Gauge Project)
"""

import os
import glob
import bisect
import argparse
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from datetime import datetime, timedelta

# ─── GLOBAL PARAMETERS ───────────────────────────────────────────────────────
SR          = 8000      # Target sample rate (Hz) — matches your downsampled dataset
N_MELS      = 64        # Mel filterbank bins
N_FFT       = 1024      # FFT window size
HOP_LENGTH  = 256       # Hop length for STFT (~32ms at 8kHz)
SPEC_WIDTH  = 128       # Fixed time-axis width (pad/crop to this)
WINDOW_MIN  = 3         # Mechanical gauge reporting interval (minutes)
TIMESTAMP_TOLERANCE = timedelta(seconds=30)  # Buffer for clock drift
MIN_WAV_FILES = 3       # Minimum audio clips needed per window (skip otherwise)
# ─────────────────────────────────────────────────────────────────────────────


def parse_wav_timestamp(filename: str) -> datetime | None:
    """
    Parse timestamp from WAV filename format: YYYY_MM_DD_HH_MM_SS[_micro].wav
    Example: 2024_07_15_08_32_01_123456.wav → datetime(2024,7,15,8,32,1)
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) >= 6:
        try:
            dt_str = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def clean_rainfall_value(val) -> float:
    """
    Clean rainfall values from Davis AeroCone CSV.
    Handles: float, '0.2 mm', '200 µm', 'N/A', etc.
    """
    val_str = str(val).strip().lower()
    if "µm" in val_str or "um" in val_str:
        num_str = val_str.replace("µm", "").replace("um", "").strip()
        try:
            return float(num_str) / 1000.0  # µm → mm
        except ValueError:
            return 0.0
    elif "mm" in val_str:
        try:
            return float(val_str.replace("mm", "").strip())
        except ValueError:
            return 0.0
    else:
        try:
            return max(float(val_str), 0.0)
        except (ValueError, TypeError):
            return 0.0


def load_mechanical_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the mechanical rain gauge CSV.
    Returns DataFrame with columns: ['datetime', 'rainfall_mm']
    Supports both old format (Time + device_frmpayload_data_rainfall)
    and new format (time + Rainfall (mm)).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Detect timestamp column
    ts_col = None
    for col in ["time", "Time", "timestamp", "Timestamp", "DATE", "date"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {csv_path}. Columns: {list(df.columns)}")

    # Detect rainfall column
    rain_col = None
    for col in ["Rainfall (mm)", "rainfall_mm", "device_frmpayload_data_rainfall",
                "device_frmpayload_data_rain", "Rain", "rain_mm"]:
        if col in df.columns:
            rain_col = col
            break
    if rain_col is None:
        raise ValueError(f"No rainfall column found in {csv_path}. Columns: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["rainfall_mm"] = df[rain_col].apply(clean_rainfall_value)

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"  [CSV] Loaded {len(df)} rows | "
          f"Rain range: {df['rainfall_mm'].min():.3f}–{df['rainfall_mm'].max():.3f} mm | "
          f"Non-zero: {(df['rainfall_mm'] > 0).sum()} rows")

    return df[["datetime", "rainfall_mm"]]


def scan_wav_files(wav_dir: str) -> list[tuple[datetime, str]]:
    """
    Recursively scan wav_dir for .wav files.
    Returns sorted list of (timestamp, filepath) tuples.
    """
    wav_files = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(wav_dir, "**", "*.WAV"), recursive=True)

    result = []
    skipped = 0
    for fpath in wav_files:
        ts = parse_wav_timestamp(fpath)
        if ts is not None:
            result.append((ts, fpath))
        else:
            skipped += 1

    result.sort(key=lambda x: x[0])
    print(f"  [WAV] Found {len(result)} parseable files | Skipped {skipped} (bad filenames)")
    return result


def align_windows(df_mech: pd.DataFrame, wav_data: list) -> list[dict]:
    """
    Align each mechanical CSV row (3-min window) to the set of WAV files
    recorded during that window using binary search.
    """
    wav_times = [w[0] for w in wav_data]
    wav_paths = [w[1] for w in wav_data]

    aligned = []
    for _, row in df_mech.iterrows():
        t_end   = row["datetime"]
        t_start = t_end - timedelta(minutes=WINDOW_MIN) - TIMESTAMP_TOLERANCE
        t_end_tol = t_end + TIMESTAMP_TOLERANCE

        idx_s = bisect.bisect_left(wav_times, t_start)
        idx_e = bisect.bisect_right(wav_times, t_end_tol)

        matching = wav_paths[idx_s:idx_e]
        if len(matching) >= MIN_WAV_FILES:
            aligned.append({
                "time":        t_end,
                "rainfall_mm": row["rainfall_mm"],
                "wav_files":   matching,
            })

    non_zero = sum(1 for s in aligned if s["rainfall_mm"] > 0.0)
    print(f"  [ALIGN] {len(aligned)} windows aligned | "
          f"{non_zero} rain ({non_zero/max(len(aligned),1)*100:.1f}%) | "
          f"{len(aligned)-non_zero} dry")
    return aligned


def load_and_concatenate_audio(wav_files: list[str]) -> np.ndarray | None:
    """
    Load and concatenate all WAV clips for a 3-minute window.
    Resamples to SR if needed. Converts stereo → mono.
    """
    combined = []
    for fpath in wav_files:
        try:
            audio, loaded_sr = sf.read(fpath, always_2d=False)
            # Stereo → mono
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            # Resample if needed
            if loaded_sr != SR:
                audio = librosa.resample(audio, orig_sr=loaded_sr, target_sr=SR)
            combined.append(audio)
        except Exception as e:
            pass  # Skip corrupt files silently

    if not combined:
        return None
    return np.concatenate(combined, axis=0).astype(np.float32)


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute log-Mel spectrogram and normalize to fixed shape (N_MELS, SPEC_WIDTH).
    
    Returns:
        spec: float32 array of shape (N_MELS, SPEC_WIDTH)
    """
    # Peak normalize audio (avoid division by zero)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=4000,
    )
    # Log compression (dB scale)
    spec = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    # Fix time axis to SPEC_WIDTH (pad with silence or crop)
    T = spec.shape[1]
    if T < SPEC_WIDTH:
        pad = np.full((N_MELS, SPEC_WIDTH - T), spec.min(), dtype=np.float32)
        spec = np.concatenate([spec, pad], axis=1)
    else:
        spec = spec[:, :SPEC_WIDTH]

    return spec.astype(np.float32)


def preprocess_month(wav_dir: str, csv_path: str, output_path: str, label: str = ""):
    """
    Full pipeline for one month of data:
    1. Load CSV
    2. Scan WAV files
    3. Align windows
    4. Extract Mel spectrograms
    5. Save as .npz

    Output .npz contains:
        X          : float32 (N, N_MELS, SPEC_WIDTH) — spectrograms
        y          : float32 (N,) — rainfall_mm per window
        y_log      : float32 (N,) — log1p(rainfall_mm) (model target)
        timestamps : str (N,)    — ISO format window end times
        label      : str         — dataset identifier
        params     : dict        — preprocessing parameters
    """
    print(f"\n{'='*60}")
    print(f"  Preprocessing: {label or os.path.basename(csv_path)}")
    print(f"{'='*60}")

    df_mech = load_mechanical_csv(csv_path)
    wav_data = scan_wav_files(wav_dir)
    aligned  = align_windows(df_mech, wav_data)

    if not aligned:
        print("  [!] No aligned windows found. Check paths and timestamps.")
        return

    X_list, y_list, t_list = [], [], []
    skipped = 0

    for i, sample in enumerate(aligned):
        if i % 100 == 0:
            print(f"  [FEAT] Processing window {i+1}/{len(aligned)} ...", end="\r")

        audio = load_and_concatenate_audio(sample["wav_files"])
        if audio is None or len(audio) < SR:  # Skip if less than 1 second of audio
            skipped += 1
            continue

        spec = compute_mel_spectrogram(audio)
        X_list.append(spec)
        y_list.append(sample["rainfall_mm"])
        t_list.append(sample["time"].isoformat())

    print(f"\n  [FEAT] Done. {len(X_list)} samples extracted | {skipped} skipped")

    if not X_list:
        print("  [!] No samples extracted. Aborting save.")
        return

    X = np.stack(X_list, axis=0)           # (N, 64, 128)
    y = np.array(y_list, dtype=np.float32) # (N,)
    y_log = np.log1p(y)                    # log1p transform for heavy-tailed labels
    timestamps = np.array(t_list, dtype=object)

    # ── Summary statistics ───────────────────────────────────────────
    non_zero = np.sum(y > 0)
    print(f"\n  [STATS] Final dataset ({label}):")
    print(f"    Samples (N):        {len(X)}")
    print(f"    Spectrogram shape:  {X.shape}")
    print(f"    Rainfall (mm):      mean={y.mean():.4f}, max={y.max():.4f}, "
          f"min={y.min():.4f}")
    print(f"    Non-zero windows:   {non_zero} ({non_zero/len(y)*100:.1f}%)")
    print(f"    Zero-rain windows:  {len(y)-non_zero} ({(len(y)-non_zero)/len(y)*100:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    params = {
        "sr": SR, "n_mels": N_MELS, "n_fft": N_FFT,
        "hop_length": HOP_LENGTH, "spec_width": SPEC_WIDTH,
        "window_min": WINDOW_MIN,
    }
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        y_log=y_log,
        timestamps=timestamps,
        label=np.array([label]),
        params=np.array([json.dumps(params)]),
    )
    size_mb = os.path.getsize(output_path + ".npz" if not output_path.endswith(".npz")
                              else output_path) / 1e6
    print(f"\n  ✅ Saved → {output_path}  ({size_mb:.1f} MB)")


def merge_datasets(npz_dir: str, out_dir: str,
                   train_frac: float = 0.70,
                   val_frac:   float = 0.15):
    """
    Merge all .npz files in npz_dir chronologically, then split into
    train / val / test sets (no shuffling — chronological order preserved).

    Also saves norm_stats.json with train-set mean/std for denormalization.
    """
    print(f"\n{'='*60}")
    print(f"  Merging datasets from: {npz_dir}")
    print(f"{'='*60}")

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        print(f"  [!] No .npz files found in {npz_dir}")
        return

    all_X, all_y, all_y_log, all_t = [], [], [], []

    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        all_X.append(data["X"])
        all_y.append(data["y"])
        all_y_log.append(data["y_log"])
        all_t.append(data["timestamps"])
        print(f"  Loaded: {os.path.basename(fpath)}  → {len(data['X'])} samples")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    y_log = np.concatenate(all_y_log, axis=0)
    timestamps = np.concatenate(all_t, axis=0)

    # Sort chronologically by timestamp
    order = np.argsort(timestamps)
    X, y, y_log, timestamps = X[order], y[order], y_log[order], timestamps[order]

    N = len(X)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    splits = {
        "train": (X[:n_train],          y[:n_train],          y_log[:n_train],          timestamps[:n_train]),
        "val":   (X[n_train:n_train+n_val], y[n_train:n_train+n_val], y_log[n_train:n_train+n_val], timestamps[n_train:n_train+n_val]),
        "test":  (X[n_train+n_val:],    y[n_train+n_val:],    y_log[n_train+n_val:],    timestamps[n_train+n_val:]),
    }

    os.makedirs(out_dir, exist_ok=True)

    # Compute normalization stats from TRAIN SET ONLY
    X_train = splits["train"][0]
    train_mean = float(X_train.mean())
    train_std  = float(X_train.std())

    norm_stats = {
        "spectrogram_mean": train_mean,
        "spectrogram_std":  train_std,
        "y_mean":           float(y[:n_train].mean()),
        "y_std":            float(y[:n_train].std()),
        "y_log_mean":       float(y_log[:n_train].mean()),
        "y_log_std":        float(y_log[:n_train].std()),
        "n_total": N, "n_train": n_train,
        "n_val":   n_val, "n_test": N - n_train - n_val,
    }

    norm_path = os.path.join(out_dir, "norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\n  ✅ Saved norm stats → {norm_path}")

    for split_name, (Xs, ys, ys_log, ts) in splits.items():
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, X=Xs, y=ys, y_log=ys_log, timestamps=ts)
        non_zero = (ys > 0).sum()
        print(f"  ✅ {split_name}.npz  → {len(Xs)} samples "
              f"({non_zero} rain, {len(ys)-non_zero} dry)")

    print(f"\n  Total dataset: {N} samples")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {N-n_train-n_val}")
    print(f"  Overall non-zero rain: {(y > 0).sum()} ({(y > 0).mean()*100:.1f}%)")
    print(f"\n  Spectrogram stats (train): mean={train_mean:.4f}, std={train_std:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic Rain Gauge — DL Preprocessing Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── Preprocess command ─────────────────────────────────────────────────
    p_pre = subparsers.add_parser("preprocess", help="Preprocess a single month")
    p_pre.add_argument("--wav_dir",  required=True, help="Directory of WAV files")
    p_pre.add_argument("--csv_path", required=True, help="Path to mechanical CSV")
    p_pre.add_argument("--output",   required=True, help="Output .npz path")
    p_pre.add_argument("--label",    default="",    help="Dataset label (e.g. jul_2024)")

    # ── Merge command ──────────────────────────────────────────────────────
    p_mrg = subparsers.add_parser("merge", help="Merge monthly .npz files")
    p_mrg.add_argument("--npz_dir", required=True, help="Directory of .npz files")
    p_mrg.add_argument("--out_dir", required=True, help="Output directory for splits")
    p_mrg.add_argument("--train_frac", type=float, default=0.70)
    p_mrg.add_argument("--val_frac",   type=float, default=0.15)

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_month(
            wav_dir=args.wav_dir,
            csv_path=args.csv_path,
            output_path=args.output,
            label=args.label,
        )
    elif args.command == "merge":
        merge_datasets(
            npz_dir=args.npz_dir,
            out_dir=args.out_dir,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
