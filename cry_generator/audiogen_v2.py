import os
import fnmatch
import time
import torchaudio

from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write


BABY_AUDIO_PATH = "./donateacry_corpus_cleaned_and_updated_data"

model = AudioGen.get_pretrained("facebook/audiogen-medium")
current_path = os.getcwd()
descriptions = ["baby crying"]
GENERATED_AUDIO_PATH = "generated_audios"


def get_audio_files_length(path: str) -> int:
    """Function count wav files."""
    wav_files = fnmatch.filter(os.listdir(path), '*.wav')
    return len(wav_files)


def create_folder_if_not_exists(folder_name: str) -> None:
    """Function creates folder if does not exist"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def gen_audios_per_category(category_quantity) -> None:
    """Function generating audios."""
    print("generated_audio_path", GENERATED_AUDIO_PATH)
    create_folder_if_not_exists(GENERATED_AUDIO_PATH)
    for category in os.listdir(BABY_AUDIO_PATH):
        print("category:", category)
        category_gen_path = os.path.join(
            current_path, GENERATED_AUDIO_PATH, category)
        create_folder_if_not_exists(category_gen_path)
        sample_cat_path = os.path.join(current_path, BABY_AUDIO_PATH, category)
        while get_audio_files_length(category_gen_path) < category_quantity:
            for file_name in os.listdir(sample_cat_path):
                if file_name.endswith(".wav"):
                    print("file_name:", file_name)
                    model.set_generation_params(
                        use_sampling=True, top_k=250, duration=7)
                    audio_path = os.path.join(
                        sample_cat_path, file_name)
                    prompt_waveform, prompt_sample_rate = torchaudio.load(
                        audio_path)
                    prompt_duration = 2
                    prompt_waveform = prompt_waveform[..., : int(
                        prompt_duration * prompt_sample_rate)]
                    output = model.generate_continuation(
                        prompt_waveform,
                        prompt_sample_rate,
                        descriptions,
                        progress=True,
                    )
                    for idx, one_wav in enumerate(output):
                        audio_write(
                            f'{category_gen_path}/{idx}-{time.time_ns()}',
                            one_wav.cpu(),
                            model.sample_rate,
                            strategy="loudness",
                            loudness_compressor=True,
                        )
                    if get_audio_files_length(category_gen_path) >= category_quantity:
                        break


gen_audios_per_category(500)
