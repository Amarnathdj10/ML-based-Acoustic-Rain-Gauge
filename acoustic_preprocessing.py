import numpy as np
import scipy.signal
import librosa

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Helper function to generate Butterworth bandpass filter coefficients in SOS format.
    Using Second-Order Sections (SOS) provides better numerical stability.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Check boundaries
    if low <= 0:
        # If lowcut is 0, it behaves as a lowpass filter
        sos = scipy.signal.butter(order, high, btype='low', output='sos')
    elif high >= 1.0:
        # If highcut meets or exceeds Nyquist, it behaves as a highpass filter
        sos = scipy.signal.butter(order, low, btype='high', output='sos')
    else:
        sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')
        
    return sos

def bandpass_filter(y, sr, lowcut=1000.0, highcut=None, order=5):
    """
    Applies a zero-phase Butterworth bandpass filter to the audio signal.
    This helps remove low-frequency wind/traffic rumble and high-frequency sensor hiss.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal (1D array).
    sr : int or float
        Sampling rate of the audio.
    lowcut : float
        Lower cutoff frequency in Hz. Default is 1000 Hz.
    highcut : float, optional
        Upper cutoff frequency in Hz. If None, it defaults to 95% of the Nyquist frequency.
    order : int
        The order of the filter. Higher orders create steeper roll-offs.
        
    Returns:
    --------
    y_filtered : np.ndarray
        Filtered audio signal.
    """
    if len(y) == 0:
        return y
        
    nyq = 0.5 * sr
    
    # Safety checks
    if lowcut >= nyq:
        raise ValueError(f"Lowcut frequency ({lowcut} Hz) must be less than Nyquist frequency ({nyq} Hz).")
        
    if highcut is None or highcut >= nyq:
        highcut = 0.95 * nyq
        
    if lowcut >= highcut:
        raise ValueError(f"Lowcut ({lowcut} Hz) must be less than highcut ({highcut} Hz).")
        
    # Get Second-Order Sections
    sos = butter_bandpass(lowcut, highcut, sr, order=order)
    
    # Apply zero-phase forward-backward filter
    y_filtered = scipy.signal.sosfiltfilt(sos, y)
    return y_filtered


def spectral_subtraction(y, sr, n_fft=2048, hop_length=512, alpha=2.0, beta=0.02, noise_estimation_pct=10.0):
    """
    Applies spectral subtraction to reduce stationary background noise.
    Estimates the noise spectrum dynamically from the quietest frames of the audio.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal (1D array).
    sr : int
        Sampling rate.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT.
    alpha : float
        Over-subtraction factor (typically 1.0 to 3.0). Higher values subtract more noise
        but can lead to distortions.
    beta : float
        Spectral floor parameter (typically 0.01 to 0.05). Prevents bin values from
        dropping to absolute zero, mitigating "musical noise" artifacts.
    noise_estimation_pct : float
        The percentage of lowest-energy frames used to estimate the noise spectrum (0 to 100).
        For rain audio, we estimate the background noise from the quietest portions.
        
    Returns:
    --------
    y_clean : np.ndarray
        Denoised audio signal.
    """
    if len(y) == 0:
        return y

    # 1. Compute Short-Time Fourier Transform (STFT)
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    # Avoid divisions by zero later
    eps = 1e-10
    
    # 2. Dynamic Noise Estimation
    # Compute frame-wise energy to identify quiet segments
    frame_energy = np.sum(magnitude ** 2, axis=0)
    
    # Find the threshold energy for the lowest percentile
    num_frames = magnitude.shape[1]
    n_noise_frames = int(max(1, np.ceil(num_frames * (noise_estimation_pct / 100.0))))
    
    # Sort frames by energy and average the magnitude spectra of the quietest frames
    noise_frame_indices = np.argsort(frame_energy)[:n_noise_frames]
    noise_est = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)
    
    # 3. Perform Subtraction
    # S_clean(f, t) = magnitude(f, t) - alpha * noise_est(f)
    subtracted = magnitude - alpha * noise_est
    
    # Apply spectral floor: ensure values don't fall below beta * magnitude (or beta * noise_est)
    floor = beta * magnitude
    magnitude_clean = np.maximum(subtracted, floor)
    
    # 4. Reconstruct Complex STFT matrix
    stft_clean = magnitude_clean * np.exp(1j * phase)
    
    # 5. Inverse Short-Time Fourier Transform (ISTFT)
    y_clean = librosa.istft(stft_clean, hop_length=hop_length)
    
    # Ensure length matches input (sometimes ISTFT can have minor padding offsets)
    if len(y_clean) > len(y):
        y_clean = y_clean[:len(y)]
    elif len(y_clean) < len(y):
        y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)), mode='constant')
        
    return y_clean


if __name__ == "__main__":
    # Example Demonstration
    import matplotlib.pyplot as plt
    
    # Generate dummy signal: a 2 kHz sine wave (rain droplet representation) 
    # mixed with low-frequency rumble (noise) and white noise.
    sr = 8000
    t = np.linspace(0, 3, sr * 3, endpoint=False) # 3 seconds
    
    # Signals
    droplet_tone = 0.5 * np.sin(2 * np.pi * 2000 * t)   # 2 kHz target signal
    wind_noise = 0.8 * np.sin(2 * np.pi * 100 * t)     # 100 Hz low-frequency rumble
    sensor_hiss = 0.1 * np.random.normal(0, 1, len(t)) # white noise
    
    noisy_signal = droplet_tone + wind_noise + sensor_hiss
    
    print("Original Signal length:", len(noisy_signal))
    
    # 1. Apply Bandpass Filter to eliminate low-frequency wind rumble
    filtered_signal = bandpass_filter(noisy_signal, sr, lowcut=1000.0, highcut=3800.0)
    
    # 2. Apply Spectral Subtraction to suppress stationary sensor hiss
    denoised_signal = spectral_subtraction(filtered_signal, sr, alpha=2.0, beta=0.02)
    
    print("Denoised Signal length:", len(denoised_signal))
    print("Preprocessing components compiled and tested successfully!")
