import librosa
#import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io


def load_audio(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    #y_denoised = nr.reduce_noise(y=y, sr=sr)
    y_trimmed, index = librosa.effects.trim(y, top_db=20)
    return y_trimmed


def get_spectrogram(y):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
        y_axis="log",
        x_axis="time"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return image
