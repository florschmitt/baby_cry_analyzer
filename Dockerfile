FROM python:3.10-slim
FROM tensorflow/tensorflow:2.13.0
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn main:app --host 0.0.0.0 --port 8000