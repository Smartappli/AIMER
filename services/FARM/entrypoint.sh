#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# uv run python manage.py collectstatic --noinput

uv run python manage.py migrate --noinput

exec "$@"
