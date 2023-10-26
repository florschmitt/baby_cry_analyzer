from fastapi import APIRouter, Request, File, UploadFile
from api.apps.render_template import render_template
from api.components import forms
from api.ml_logic.preprpcessings import load_audio, get_spectrogram, image_to_base64
from keras.utils import img_to_array
import numpy as np
from io import BytesIO
import base64

vgg16_endpoint = APIRouter()


@vgg16_endpoint.get("/")
async def get_prediction_page(request: Request):
    return render_template("vgg16.html", {"request": request})


@vgg16_endpoint.post("/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    from main import get_vgg16_model
    form = forms.FileUploadForm(request)
    await form.load_data()
    if not await form.file_is_valid():
        return render_template("vgg16.html", {"request": request, "errors": form.errors})

    classes = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

    audio_content = await file.read()
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")
    filename = file.filename
    y_clean = load_audio(BytesIO(audio_content))
    spectrogram = get_spectrogram(y_clean)
    spectrogram_base64 = image_to_base64(spectrogram)
    spectrogram = spectrogram.resize((224, 224))
    spectrogram_array = img_to_array(spectrogram)
    spectrogram_array = np.expand_dims(spectrogram_array, axis=0)
    prediction = get_vgg16_model().predict(spectrogram_array)
    predicted_class = np.argmax(prediction)
    prediction_label = classes[predicted_class]
    return render_template("vgg16.html", {
        "request": request,
        "msg": f"{prediction_label}",
        "spectrogram": spectrogram_base64,
        "audio_base64": audio_base64,
        "filename": filename,
        "prediction_label": prediction_label
    })
