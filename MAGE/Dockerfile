FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système (souvent utiles pour wheels)
RUN apt-get update \
    && apt-get install git -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Déps Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# Port FastAPI
EXPOSE 8000

# module:variable ASGI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
