import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import noisereduce as nr


def process_audio_and_create_spectrogram(audio_file_path, spectrogram_folder):
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_trimmed, index = librosa.effects.trim(y_denoised, top_db=20)
        spectrogram = np.abs(librosa.stft(y_trimmed))
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {os.path.basename(audio_file_path)}')

        plt.savefig(os.path.join(spectrogram_folder, f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.png"))
        plt.close()
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")


audio_folder = "audios"

categories = ["belly_pain", "discomfort", "hungry", "tired", "burping"]

for category in categories:
    category_path = os.path.join(audio_folder, category)
    spectrogram_folder = os.path.join(category_path, "spectrograms")
    os.makedirs(spectrogram_folder, exist_ok=True)

    for filename in os.listdir(category_path):
        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(category_path, filename)
        process_audio_and_create_spectrogram(file_path, spectrogram_folder)
