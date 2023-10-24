FROM tensorflow/tensorflow:2.13.0
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY api /app
CMD uvicorn main:app --host 127.0.0.1 --port 8000
