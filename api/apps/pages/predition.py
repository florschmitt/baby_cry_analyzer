from fastapi import APIRouter, Request, File, UploadFile, BackgroundTasks
from api.apps.render_template import render_template
from pathlib import Path
from api.components import forms
import librosa
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import io

prediction_endpoint = APIRouter()


@prediction_endpoint.get("/")
async def get_prediction_page(request: Request):
    return render_template("predict.html", {"request": request})


async def convert_and_save_file(file: UploadFile, destination: Path):
    try:
        # Read audio file
        audio_data = file.file.read()
        audio = io.BytesIO(audio_data)

        # Load the audio
        y, sr = librosa.load(audio, sr=None)

        # Removing noise
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        # Removing silences
        y_trimmed, index = librosa.effects.trim(y_denoised, top_db=20)

        # Create a spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(
            librosa.amplitude_to_db(librosa.stft(y_trimmed), ref=np.max),
            y_axis="log",
            x_axis="time"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")

        # Save the spectrogram to a file
        with destination.open("wb") as buffer:
            plt.savefig(buffer, format="png")
        print(f"Spectrogram saved to {destination}")
    except Exception as e:
        print(f"Failed to convert audio to spectrogram and save: {e}")


@prediction_endpoint.post("/")
async def upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    errors = []
    msg = None

    form = forms.FileUploadForm(request)
    await form.load_data()

    if await form.file_is_valid():
        # Changed the file extension to .png since we are now saving a spectrogram
        destination = Path(f"data/input_data/spectogram/{file.filename}.png")
        background_tasks.add_task(convert_and_save_file, file, destination)
        msg = f"Spectrogram of file {file.filename} created and saved successfully!"
    else:
        errors = form.errors

    return render_template("predict.html", {"request": request, "errors": errors, "msg": msg})
