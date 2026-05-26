#!/bin/sh
set -e

if [ "${RUN_DJANGO_COLLECTSTATIC:-1}" = "1" ]; then
  uv run --locked --no-dev --no-build python manage.py collectstatic --noinput
fi

if [ "${RUN_DJANGO_MIGRATIONS:-1}" = "1" ]; then
  uv run --locked --no-dev --no-build python manage.py migrate --noinput
fi

exec "$@"
