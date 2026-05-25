FROM python:3.13-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

# set environment variables
ENV HOME=${APP_HOME} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=${APP_HOME}/.cache/uv \
    XDG_CACHE_HOME=${APP_HOME}/.cache

RUN python -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --only-binary=:all: \
    "uv==0.9.28"

COPY pyproject.toml uv.lock ./

# ibm-cos-sdk is locked but currently has no Python 3.13 wheel.
RUN uv sync --locked --no-dev # NOSONAR

COPY AIMER ./AIMER
COPY RAG ./RAG
COPY auth ./auth
COPY static ./static
COPY templates ./templates
COPY website ./website
COPY manage.py ./

# entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup "${APP_HOME}" /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uv", "run", "--locked", "--no-dev", "--no-build", \
    "granian", "--interface", "asgi", "--host", "0.0.0.0", \
    "--port", "8000", "--workers", "4", "AIMER.asgi:application"]
