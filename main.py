from uvicorn import run
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api.core.configurations import server_settings, security_settings
from api.ml_logic.model import vgg16_model_load, forest_model_load
from api.apps.routers import apps_router
import warnings
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore", category=UserWarning)
ALLOWED_HOSTS = ["*"]
forest_model = None
vgg16_model = None


def get_vgg16_model():
    if vgg16_model is not None:
        return vgg16_model
    else:
        raise Exception("The model is not loaded yet!")


def get_forest_model():
    if forest_model is not None:
        return forest_model
    else:
        raise Exception("The model is not loaded yet!")


def middleware(server):
    server.add_middleware(
        SessionMiddleware,
        secret_key=security_settings.baby_secret_key
    )
    server.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


def include_routers(server):
    server.include_router(apps_router)


def configure_static(server):
    server.mount("/statics", StaticFiles(directory="statics"), name="statics")


def start_application():
    server = FastAPI()
    include_routers(server)
    middleware(server)
    configure_static(server)
    return server


app = start_application()


@app.on_event("startup")
async def startup():
    global vgg16_model
    global forest_model
    vgg16_path = "VGG16_Baby_prod.h5"
    forest_path = "Forest_5_98.pkl"
    vgg16_model = await vgg16_model_load(vgg16_path)
    forest_model = await forest_model_load(forest_path)


if __name__ == "__main__":
    run(
        "main:app",
        host=server_settings.server_host,
        port=server_settings.server_port,
        log_level="info",
        reload=True,
    )
