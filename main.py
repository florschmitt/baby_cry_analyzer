from uvicorn import run
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api.core.configurations import server_settings, security_settings

from api.apps.routers import apps_router

ALLOWED_HOSTS = ["*"]


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


def start_application():
    server = FastAPI()
    include_routers(server)
    middleware(server)
    return server


app = start_application()

if __name__ == "__main__":
    run(
        "main:app",
        host=server_settings.server_host,
        port=server_settings.server_port,
        log_level="info",
        reload=True,
    )
