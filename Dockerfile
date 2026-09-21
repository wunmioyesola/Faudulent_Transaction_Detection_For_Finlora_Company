FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc

RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn mlflow dagshub xgboost

COPY . .

CMD uvicorn main.app:app --host 0.0.0.0 --port 8000