FROM python:3.13-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendances système (souvent utiles pour wheels)
RUN apt-get update \
    && apt-get install git -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install uv

# Déps Python
COPY ./MAGE/pyproject.toml ./MAGE/uv.lock ./
RUN uv sync --frozen

# Code
COPY ./MAGE/ ${APP_HOME}

# Port FastAPI
EXPOSE 8000

# module:variable ASGI
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "4"]
