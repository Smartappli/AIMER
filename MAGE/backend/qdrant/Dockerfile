FROM qdrant/qdrant:latest
USER 0
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
USER 0