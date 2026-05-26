#!/bin/sh
set -e

# Optionnel: collectstatic si tu en as besoin
# python manage.py collectstatic --noinput

VENV_BIN="${VIRTUAL_ENV:-/app/.venv}/bin"

if [ "${RAG_VERIFY_ON_START:-0}" = "1" ]; then
  echo "[entrypoint] Running OpenRAG readiness check..."
  "${VENV_BIN}/python" -m RAG.verify_openrag
fi

if [ "${RUN_DJANGO_MIGRATIONS:-1}" = "1" ]; then
  "${VENV_BIN}/python" manage.py migrate --noinput
fi

exec "$@"
