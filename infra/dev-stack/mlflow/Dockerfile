FROM ghcr.io/mlflow/mlflow:v3.12.0

RUN pip install --no-cache-dir psycopg2-binary

USER 1000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).read()" || exit 1
