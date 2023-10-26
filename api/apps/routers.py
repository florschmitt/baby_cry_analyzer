from fastapi import APIRouter
from api.apps.pages.home_page import home_endpoint
from api.apps.pages.vgg16 import vgg16_endpoint
from api.apps.pages.forest import forest_endpoint
from api.apps.pages.about_page import about_endpoint

apps_router = APIRouter()

apps_router.include_router(home_endpoint, prefix="", tags=["home_page"])
apps_router.include_router(vgg16_endpoint, prefix="/vgg16", tags=["vgg16_page"])
apps_router.include_router(forest_endpoint, prefix="/forest", tags=["forest_page"])
apps_router.include_router(about_endpoint, prefix="/about", tags=["about_page"])
