# Dockerfile: minimal image to run the FastAPI inference app
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for lightgbm and shap
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use Uvicorn to serve the API
EXPOSE 8000
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
