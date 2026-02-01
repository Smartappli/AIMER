FROM python:3.14-slim

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# install python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . ${APP_HOME}

# running migrations
RUN python manage.py migrate

# gunicorn
CMD ["uvicorn", "AIMER.asgi.application", "--host", "0.0.0.0", "--port", "9000", "--workers", "4"]
