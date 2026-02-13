FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendances système (souvent utiles pour wheels)
RUN apt-get update \
    && apt-get install git -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Déps Python
COPY ./MAGE/pyproject.toml ./MAGE/uv.lock ./

RUN pip install --upgrade pip && pip install uv && uv sync --active

WORKDIR /app

# Code
COPY ./MAGE/ .

# Port FastAPI
EXPOSE 8000

# module:variable ASGI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "4"]
