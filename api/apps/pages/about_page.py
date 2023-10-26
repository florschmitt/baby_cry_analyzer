from fastapi import APIRouter, Request
from api.apps.render_template import render_template

about_endpoint = APIRouter()


@about_endpoint.get("/presentation")
async def get_presentation_page(request: Request):
    # return {"Work": "In progress"}
    return render_template("presentation.html", {"request": request})


@about_endpoint.get("/about")
async def get_about_page(request: Request):
    # return {"Work": "In progress"}
    return render_template("about.html", {"request": request})
