import librosa
import numpy as np

audio, sr = librosa.load(r"D:\Coding journey\ML-based Acoustic Rain Gauge\audio files\hi.wav",sr=None)
#print(audio)

duration = len(audio)/sr
#print(duration)

stft = librosa.stft(audio)
#print(stft)

magnitude = np.abs(stft)
#print(magnitude)
#print(magnitude.shape)

spectrogram = librosa.amplitude_to_db(magnitude)
#print(spectrogram)
#print(spectrogram.shape)

mel_spec = librosa.feature.melspectrogram(y=audio,sr=sr)
#print(mel_spec)
#print(mel_spec.shape)

zcr = librosa.feature.zero_crossing_rate(audio)
#print(zcr)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)
print(mfcc)