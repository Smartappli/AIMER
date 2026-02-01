FROM python:3.13-slim

ARG APP_HOME=/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirement files
COPY requirements.txt .

# install python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Files copy
WORKDIR ${APP_HOME}
COPY . ${APP_HOME}

# running migrations
RUN python manage.py migrate

# gunicorn
CMD ["uvicorn", "AIMER.asgi.application", "--host", "0.0.0.0", "--port", "9000", "--workers", "4"]
