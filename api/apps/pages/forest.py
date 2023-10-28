import numpy as np
from typing import Dict
from api.ml_logic.preprpcessings import extract_mfcc
from .services import (
    APIRouter, Request,
    File, UploadFile,
    render_template,
    forms, get_audio_data,
    random_pics
)

import time
from api.core.constants import CLASS_LABELS

forest_endpoint = APIRouter()


@forest_endpoint.get("/")
async def get_forest_page(request: Request):
    return await render_template("forest.html", {"request": request})


async def predict(audio_features, model):
    features_flatten = audio_features.flatten()
    repeat_times = 1200 // len(features_flatten) + 1
    extended_features = np.tile(features_flatten, repeat_times)[:1200]
    extended_features = np.expand_dims(extended_features, axis=0)
    return model.predict(extended_features)


@forest_endpoint.post("/")
async def forest_predict(request: Request, file: UploadFile = File(...)) -> Dict:
    start = time.perf_counter()

    from main import get_forest_model
    form = forms.FileUploadForm(request)
    await form.load_data()
    if not await form.file_is_valid():
        return await render_template("forest.html", {"request": request, "errors": form.errors})

    audio_base64, audio_content_b, y_clean, _, spectrogram_base64 = await get_audio_data(file)

    mfccs = extract_mfcc(audio_content_b)
    prediction = await predict(mfccs, get_forest_model())
    prediction_label = CLASS_LABELS.get(prediction[0])

    print(time.perf_counter() - start)
    return await render_template("forest.html", {
        "request": request,
        "msg": "File Loaded",
        "audio_base64": audio_base64,
        "spectrogram": spectrogram_base64,
        "filename": file.filename,
        "prediction_label": prediction_label,
        "random_image": random_pics(prediction_label)
    })
