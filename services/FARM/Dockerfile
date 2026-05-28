FROM python:3.13-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV HOME=${APP_HOME} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --only-binary=:all: \
    "uv==0.9.28"

# deps
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-build --no-cache

ENV VIRTUAL_ENV=${APP_HOME}/.venv \
    PATH="${APP_HOME}/.venv/bin:${PATH}"

COPY FARM ./FARM
COPY info.py manage.py __init__.py ./

# entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup "${APP_HOME}" /entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz/', timeout=3).read()" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["granian", "--interface", "asgi", "--host", "0.0.0.0", \
    "--port", "8000", "--workers", "4", "FARM.asgi:application"]
