#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# uv run --locked --no-dev python manage.py collectstatic --noinput

if [ "${RUN_DJANGO_MIGRATIONS:-1}" = "1" ]; then
  uv run --locked --no-dev --no-build python manage.py migrate --noinput
fi

exec "$@"
