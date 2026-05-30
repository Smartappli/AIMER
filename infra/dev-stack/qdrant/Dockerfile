FROM qdrant/qdrant:v1.18.1

USER qdrant

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD qdrant --version >/dev/null || exit 1
