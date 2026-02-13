FROM python:3.13-slim

ARG APP_HOME=/app
ARG SECRET_KEY_BUILD=build-only-not-for-prod
WORKDIR ${APP_HOME}

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# install python dependencies
RUN pip install --upgrade pip && pip install uv

# Files copy
COPY ./AIMER-ROOT/pyproject.toml ./AIMER-ROOT/uv.lock ./

RUN uv sync --frozen

COPY ./AIMER-ROOT/ ${APP_HOME}

# running migrations
RUN SECRET_KEY=${SECRET_KEY_BUILD} uv run python manage.py migrate

# gunicorn
CMD ["uv", "run", "uvicorn", "AIMER.asgi.application", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
