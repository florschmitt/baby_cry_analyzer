from fastapi import APIRouter, Request, File, UploadFile
from api.apps.render_template import render_template
from api.components import forms
from api.ml_logic.preprpcessings import load_audio, get_spectrogram
from keras.utils import img_to_array
import numpy as np
from io import BytesIO

prediction_endpoint = APIRouter()


@prediction_endpoint.get("/")
async def get_prediction_page(request: Request):
    return render_template("predict.html", {"request": request})


@prediction_endpoint.post("/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    from main import get_baby_model
    form = forms.FileUploadForm(request)
    await form.load_data()
    if not await form.file_is_valid():
        return render_template("predict.html", {"request": request, "errors": form.errors})
    classes = [
        "belly_pain",
        "burping",
        "discomfort",
        "hungry",
        "tired"
    ]

    audio_content = await file.read()
    y_clean = load_audio(BytesIO(audio_content))
    spectrogram = get_spectrogram(y_clean)
    spectrogram = spectrogram.resize((224, 224))
    spectrogram_array = img_to_array(spectrogram)
    spectrogram_array = np.expand_dims(spectrogram_array, axis=0)
    # model = get_baby_model()
    prediction = get_baby_model().predict(spectrogram_array)
    predicted_class = np.argmax(prediction)
    return render_template("predict.html", {"request": request, "msg": f"{classes[predicted_class]}"})
