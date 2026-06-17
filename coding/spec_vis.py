import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load(r"audio files/hi.wav", sr=8000)

D = librosa.stft(y)

S_db = librosa.amplitude_to_db(abs(D))

librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis='time',
    y_axis='hz',
    cmap='inferno',
    vmin=-50,
    vmax=0
)

plt.colorbar()
plt.show()