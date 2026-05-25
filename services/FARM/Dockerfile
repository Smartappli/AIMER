FROM python:3.13-slim

ARG APP_HOME=/app
ARG UV_VERSION=0.9.28
WORKDIR ${APP_HOME}

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

# deps (cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# app
COPY . ${APP_HOME}

# entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh     && addgroup --system appgroup     && adduser --system --ingroup appgroup appuser     && chown -R appuser:appgroup ${APP_HOME} /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "--locked", "--no-dev", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "FARM.asgi:application"]
