from fastapi import APIRouter
from api.apps.pages.home_page import home_endpoint
from api.apps.pages.prediction import prediction_endpoint

apps_router = APIRouter()

apps_router.include_router(home_endpoint, prefix="", tags=["home_page"])
apps_router.include_router(prediction_endpoint, prefix="/predict", tags=["prediction_page"])
print("")