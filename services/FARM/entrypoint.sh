#!/bin/sh
set -e

VENV_BIN="${VIRTUAL_ENV:-/app/.venv}/bin"

if [ "${RUN_DJANGO_COLLECTSTATIC:-1}" = "1" ]; then
  "${VENV_BIN}/python" manage.py collectstatic --noinput
fi

if [ "${RUN_DJANGO_MIGRATIONS:-1}" = "1" ]; then
  "${VENV_BIN}/python" manage.py migrate --noinput
fi

exec "$@"
