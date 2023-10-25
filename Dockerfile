FROM tensorflow/tensorflow:2.13.0
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#COPY api /api
COPY . .
#COPY templates /templates
#CMD uvicorn main:app --host 0.0.0.0
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
