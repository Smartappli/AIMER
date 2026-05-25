#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# uv run --locked --no-dev python manage.py collectstatic --noinput

uv run --locked --no-dev python manage.py migrate --noinput

exec "$@"
