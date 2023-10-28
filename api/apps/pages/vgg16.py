import time
import numpy as np
from keras.utils import img_to_array
from api.ml_logic.preprpcessings import get_spectrogram
from .services import (
    APIRouter, Request,
    File, UploadFile,
    render_template,
    forms, get_audio_data,
    random_pics
)
from api.core.constants import CLASSES

vgg16_endpoint = APIRouter()


@vgg16_endpoint.get("/")
async def get_vgg16_page(request: Request):
    return await render_template("vgg16.html", {"request": request})


def get_spectrogram_array(y_clean):
    spectrogram = get_spectrogram(y_clean)
    spectrogram = spectrogram.resize((224, 224))
    spectrogram_array = img_to_array(spectrogram)
    return np.expand_dims(spectrogram_array, axis=0)


def predict_and_sort(predictions):
    prediction_percentages = {label: round(p * 100, 2) for label, p in zip(CLASSES, predictions)}
    return dict(sorted(prediction_percentages.items(), key=lambda item: item[1], reverse=True))


@vgg16_endpoint.post("/")
async def vgg16_predict(request: Request, file: UploadFile = File(...)):
    start = time.perf_counter()
    from main import get_vgg16_model

    form = forms.FileUploadForm(request)
    await form.load_data()
    if not await form.file_is_valid():
        return await render_template("vgg16.html", {"request": request, "errors": form.errors})

    audio_base64, audio_content_b, y_clean, _, spectrogram_base64 = await get_audio_data(file)

    prediction = get_vgg16_model().predict(get_spectrogram_array(y_clean))
    sorted_predictions = predict_and_sort(prediction[0])

    print(time.perf_counter() - start)
    return await render_template("vgg16.html", {
        "request": request,
        "msg": "File Loaded",
        "spectrogram": spectrogram_base64,
        "audio_base64": audio_base64,
        "filename": file.filename,
        "prediction_percentages": sorted_predictions,
        "random_image": random_pics("different_cry")
    })
