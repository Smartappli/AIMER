FROM python:3.13-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --upgrade pip && pip install uv

# deps (cache)
COPY ./FARM/pyproject.toml ./FARM/uv.lock ./
RUN uv sync --frozen

# app
COPY ./FARM/ ${APP_HOME}

# entrypoint
COPY ./FARM/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "FARM.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
