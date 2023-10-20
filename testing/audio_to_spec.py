import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt


def audio_to_spec(clean_audio, spectrogram_folder):
    try:
        y, sr = librosa.load(clean_audio, sr=None)
        spectrogram = np.abs(librosa.stft(y))

        plt.figure(figsize=(10, 6))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {os.path.basename(clean_audio)}')

        # Extract the category from the clean audio path
        category = os.path.basename(os.path.dirname(clean_audio))

        # Create subfolder path based on the category
        subfolder = f"spectrogram_{category}"
        subfolder_path = os.path.join(spectrogram_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        # Save the spectrogram in the subfolder
        spectrogram_path = os.path.join(subfolder_path,
                                        f"{os.path.splitext(os.path.basename(clean_audio))[0]}.png")
        plt.savefig(spectrogram_path)
        plt.close()

        return spectrogram_path
    except Exception as e:
        print(f"Error creating spectrogram for {clean_audio}: {e}")
        return None


audio_folder = "clean_audio"

categories = [
    "clean_belly_pain",
    "clean_discomfort",
    "clean_hungry",
    "clean_tired",
    "clean_burping"
]

spectrogram_folder = "spectrogram"
os.makedirs(spectrogram_folder, exist_ok=True)

for category in categories:
    category_path = os.path.join(audio_folder, category)

    for filename in os.listdir(category_path):
        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(category_path, filename)
        audio_to_spec(file_path, spectrogram_folder)
