import yaml
from typing import Any
from pydantic import BaseSettings
from api.handlers.log_handler import log_errors

config_path = "./api/core/configurations.yaml"


class Settings(BaseSettings):
    config_path: str
    data: Any

    def __init__(self, **data: Any):
        super().__init__(**data)
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f)
        except Exception as e:
            log_errors(status_code=500, detail=f"{str(e)}")


class ServerSettings(Settings):

    @property
    def server_host(self):
        return self.data["server"]["server_host"]

    @property
    def server_port(self):
        return self.data["server"]["server_port"]


class SecuritySettings(Settings):
    @property
    def baby_secret_key(self):
        return self.data["security"]["baby_api_key"]


server_settings = ServerSettings(config_path=config_path)
security_settings = SecuritySettings(config_path=config_path)
