#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# uv run --frozen --no-dev python manage.py collectstatic --noinput

if [ "${RAG_VERIFY_ON_START:-0}" = "1" ]; then
  echo "[entrypoint] Running OpenRAG readiness check..."
  uv run --frozen --no-dev verify-openrag
fi

uv run --frozen --no-dev python manage.py migrate --noinput

exec "$@"
