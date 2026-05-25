# Infrastructure

Infrastructure code and local integration dependencies live outside application
services.

## Directories

- `dev-stack/`: Docker Compose stack for local Postgres, Qdrant, Airflow,
  MLflow, OMOP, Langfuse, and supporting services.

Keep service Dockerfiles in `services/<service>/`. Keep third-party/local
dependency Dockerfiles that are not application services under `infra/`.

## Local Stack

```powershell
cd infra\dev-stack
docker compose up -d
```

