FROM python:3.13-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System dependencies used by ML wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --only-binary=:all: \
        "uv==0.9.28" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-build

COPY api ./api
COPY info.py __init__.py ./

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup "${APP_HOME}"

USER appuser

EXPOSE 10000

CMD ["uv", "run", "--locked", "--no-dev", "--no-build", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "10000", "--workers", "4", "api.main:app"]
