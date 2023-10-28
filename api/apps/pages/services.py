from fastapi import APIRouter, Request, File, UploadFile
from api.apps.render_template import render_template
from api.components import forms
from api.ml_logic.preprpcessings import load_audio, get_spectrogram
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import os
import random
from api.core.constants import PICS_PATH


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


async def get_audio_data(file: UploadFile):
    audio_content = await file.read()
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")
    audio_content_b = BytesIO(audio_content)
    y_clean = load_audio(BytesIO(audio_content))
    spectrogram = get_spectrogram(y_clean)
    spectrogram_base64 = image_to_base64(spectrogram)
    return audio_base64, audio_content_b, y_clean, spectrogram, spectrogram_base64


def random_pics(sub_folder: str):
    image_folder = os.path.join(PICS_PATH, sub_folder)
    image_files = os.listdir(image_folder)
    return random.choice(image_files)
