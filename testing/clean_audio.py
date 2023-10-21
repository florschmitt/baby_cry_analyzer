import librosa
import noisereduce as nr
import os
import soundfile as sf


def clean_safe(audio_file_path, cleaned_audio_folder):
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_trimmed, index = librosa.effects.trim(y_denoised, top_db=20)

        # Extract the category from the original file path
        category = os.path.basename(os.path.dirname(audio_file_path))

        # Mapping of raw categories to clean categories
        category_mapping = {
            "raw_belly_pain": "clean_belly_pain",
            "raw_discomfort": "clean_discomfort",
            "raw_hungry": "clean_hungry",
            "raw_tired": "clean_tired",
            "raw_burping": "clean_burping"
        }

        # Get the corresponding clean category
        clean_category = category_mapping.get(category, category)

        subfolder_path = os.path.join(cleaned_audio_folder, clean_category)
        os.makedirs(subfolder_path, exist_ok=True)

        # Save the cleaned audio in the subfolder
        cleaned_audio_path = os.path.join(subfolder_path,
                                          f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_cleaned.wav")
        sf.write(cleaned_audio_path, y_trimmed, sr)

        return cleaned_audio_path
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None


audio_folder = "raw_audio"

categories = [
    "raw_belly_pain",
    "raw_discomfort",
    "raw_hungry",
    "raw_tired",
    "raw_burping"
]

clean_audio_folder = "clean_audio"
os.makedirs(clean_audio_folder, exist_ok=True)

for category in categories:
    category_path = os.path.join(audio_folder, category)

    for filename in os.listdir(category_path):
        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(category_path, filename)
        clean_safe(file_path, clean_audio_folder)
