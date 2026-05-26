#!/bin/sh
set -e

VENV_BIN="${VIRTUAL_ENV:-/app/.venv}/bin"

if [ -f manage.py ] && [ "${RUN_DJANGO_COLLECTSTATIC:-1}" = "1" ]; then
  "${VENV_BIN}/python" manage.py collectstatic --noinput
fi

if [ "${RAG_VERIFY_ON_START:-0}" = "1" ]; then
  echo "[entrypoint] Running OpenRAG readiness check..."
  "${VENV_BIN}/python" -m RAG.verify_openrag
fi

if [ "${RUN_DJANGO_MIGRATIONS:-1}" = "1" ]; then
  "${VENV_BIN}/python" manage.py migrate --noinput
fi

exec "$@"
