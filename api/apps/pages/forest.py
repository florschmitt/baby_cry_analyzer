from fastapi import APIRouter, Request, File, UploadFile
from api.apps.render_template import render_template
from api.components import forms
from api.ml_logic.preprpcessings import extract_mfcc
from api.ml_logic.preprpcessings import load_audio, get_spectrogram, image_to_base64
import numpy as np
from typing import Dict
import base64
from io import BytesIO

forest_endpoint = APIRouter()


@forest_endpoint.get("/")
async def get_forest_page(request: Request):
    return await render_template("forest.html", {"request": request})


@forest_endpoint.post("/")
async def forest_predict(request: Request, file: UploadFile = File(...)) -> Dict:
    class_labels = {
        0: "belly_pain",
        1: "burping",
        2: "discomfort",
        3: "hungry",
        4: "tired"
    }

    from main import get_forest_model
    form = forms.FileUploadForm(request)
    await form.load_data()
    if not await form.file_is_valid():
        return await render_template("forest.html", {"request": request, "errors": form.errors})

    # Read the uploaded audio file
    audio_content = await file.read()
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")
    with open("temp_audio_file.wav", "wb") as audio_file:
        audio_file.write(audio_content)
    mfccs = await extract_mfcc("temp_audio_file.wav")
    prediction = await predict(mfccs, get_forest_model())
    y_clean = await load_audio(BytesIO(audio_content))
    spectrogram = await get_spectrogram(y_clean)
    spectrogram_base64 = await image_to_base64(spectrogram)
    prediction_label = class_labels.get(prediction[0])

    return await render_template("forest.html", {
        "request": request,
        "msg": "File Loaded",
        "audio_base64": audio_base64,
        "spectrogram": spectrogram_base64,
        "filename": file.filename,
        "prediction_label": prediction_label
    })


async def predict(audio_features, model):
    features_flatten = audio_features.flatten()
    required_features = 1200
    repeat_times = required_features // len(features_flatten) + 1
    extended_features = np.tile(features_flatten, repeat_times)[:required_features]
    extended_features = np.expand_dims(extended_features, axis=0)

    prediction = model.predict(extended_features)
    return prediction
