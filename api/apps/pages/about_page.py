from fastapi import APIRouter, Request
from api.apps.render_template import render_template

about_endpoint = APIRouter()


@about_endpoint.get("/")
async def get_prediction_page(request: Request):
    # return {"Work": "In progress"}
    return render_template("about2.html", {"request": request})
