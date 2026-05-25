#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# uv run --frozen --no-dev python manage.py collectstatic --noinput

uv run --frozen --no-dev python manage.py migrate --noinput

exec "$@"
