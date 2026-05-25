FROM python:3.13-slim

ARG APP_HOME=/app
ARG UV_VERSION=0.9.28
WORKDIR ${APP_HOME}

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# Dépendances système (souvent utiles pour wheels)
RUN apt-get update     && apt-get install git -y --no-install-recommends build-essential     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

# Déps Python
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# Code
COPY . ${APP_HOME}

RUN addgroup --system appgroup     && adduser --system --ingroup appgroup appuser     && chown -R appuser:appgroup ${APP_HOME}

USER appuser

# Port FastAPI
EXPOSE 10000

# module:variable ASGI
CMD ["uv", "run", "--locked", "--no-dev", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "10000", "--workers", "4", "api.main:app"]
