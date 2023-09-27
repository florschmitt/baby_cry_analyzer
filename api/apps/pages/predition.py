from fastapi import APIRouter, Request, File, UploadFile, BackgroundTasks
from api.apps.render_template import render_template
from pathlib import Path
from api.components import forms

prediction_endpoint = APIRouter()


@prediction_endpoint.get("/")
async def get_prediction_page(request: Request):
    return render_template("predict.html", {"request": request})


async def save_upload_file(file: UploadFile, destination: Path):
    try:
        with destination.open("wb") as buffer:
            buffer.write(file.file.read())
    except Exception as e:
        print(f"Failed to save file: {e}")


@prediction_endpoint.post("/")
async def upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    errors = []
    msg = None

    form = forms.FileUploadForm(request)
    await form.load_data()

    if await form.file_is_valid():
        destination = Path(f"data/input_data/audio/{file.filename}")
        background_tasks.add_task(save_upload_file, file, destination)
        msg = f"File {file.filename} uploaded successfully!"
    else:
        errors = form.errors

    return render_template("predict.html", {"request": request, "errors": errors, "msg": msg})
