#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# uv run --locked --no-dev --no-build python manage.py collectstatic --noinput

if [ "${RAG_VERIFY_ON_START:-0}" = "1" ]; then
  echo "[entrypoint] Running OpenRAG readiness check..."
  uv run --locked --no-dev --no-build verify-openrag
fi

if [ "${RUN_DJANGO_MIGRATIONS:-1}" = "1" ]; then
  uv run --locked --no-dev --no-build python manage.py migrate --noinput
fi

exec "$@"
