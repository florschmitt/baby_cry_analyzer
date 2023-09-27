import os
import logging
from fastapi import HTTPException
from datetime import date

logs_dir = "./logs"
os.makedirs(logs_dir, exist_ok=True)

today = date.today()
file_name = f"./logs/{today}_error.log"

logger = logging.getLogger("error_logger")

logging.basicConfig(
    filename=file_name,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)


def log_errors(status_code, detail):
    logging.error(f"Error : {status_code}, {detail}")


def raise_http_exception(status_code, detail):
    error = HTTPException(status_code=status_code, detail=detail)
    raise error
