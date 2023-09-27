from fastapi import APIRouter, Request
from api.apps.render_template import render_template

home_endpoint = APIRouter()


@home_endpoint.get("/")
async def get_home_page(request: Request):
    return render_template("home.html", {"request": request})
