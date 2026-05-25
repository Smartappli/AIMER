FROM python:3.13-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# Dépendances système (souvent utiles pour wheels)
RUN apt-get update     && apt-get install git -y --no-install-recommends build-essential     && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install uv

# Déps Python
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Code
COPY . ${APP_HOME}

RUN addgroup --system appgroup     && adduser --system --ingroup appgroup appuser     && chown -R appuser:appgroup ${APP_HOME}

USER appuser

# Port FastAPI
EXPOSE 8000

# module:variable ASGI
CMD ["uv", "run", "--frozen", "--no-dev", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "10000", "--workers", "4", "api.main:app"]
