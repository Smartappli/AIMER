# Infrastructure

Infrastructure code and local integration dependencies live outside application
services.

## Directories

- `dev-stack/`: Docker Compose stack for local Postgres, Qdrant, Airflow,
  MLflow, OMOP, Langfuse, and supporting services.
- `prod/`: production planning and orchestration baseline. It keeps secrets out
  of source control and uses digest-pinned image placeholders.

Keep service Dockerfiles in `services/<service>/`. Keep third-party/local
dependency Dockerfiles that are not application services under `infra/`.

## Local Stack

```powershell
cd infra\dev-stack
docker compose up -d
```
