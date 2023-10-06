from audiocraft.models import AudioGen
import torchaudio
from audiocraft.data.audio import audio_write
import os


def gen_audios(run: int = 0) -> None:
    """Function generating audios."""
    baby_audio_folder = r"donateacry_corpus_cleaned_and_updated_data/"
    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    current_path = os.getcwd()
    descriptions = ["baby crying"]
    categories = ["burping", "belly_pain", "tired", "discomfort"]
    # for category in os.listdir(baby_audio_folder):
    for category in categories:
        print("category:", category)
        category_path = os.path.join(current_path, baby_audio_folder, category)
        for file_name in os.listdir(category_path):
            if file_name.endswith(".wav"):
                print("file_name:", file_name)
                model.set_generation_params(
                    use_sampling=True, top_k=250, duration=7)
                audio_path = os.path.join(
                    category_path, file_name)
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
                        f'gen-{category}-{run}-{idx}-{file_name.split(".")[0]}',
                        one_wav.cpu(),
                        model.sample_rate,
                        strategy="loudness",
                        loudness_compressor=True,
                    )


gen_audios(0)
gen_audios(1)
