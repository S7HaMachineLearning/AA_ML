FROM python:3.11-slim-buster

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY /src .
COPY automations.db .
COPY tokenizer.pickle .
COPY model.h5 .


CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]