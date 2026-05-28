# Production Infrastructure Skeleton

`infra/dev-stack` is for local integration only. Production deployments must use
separate infrastructure with managed secrets, private networking, encrypted
storage, monitored ingress and auditable change control.

## Required Boundaries

- Public ingress: reverse proxy or API gateway only.
- Private services: `aimer-rag`, `MAGE`, `FARM`, databases, vector stores,
  observability and workflow tools.
- No direct public ports for Postgres, Qdrant, MLflow, OMOP WebAPI, Langfuse,
  MinIO, Airflow or Neo4j.
- Service-to-service calls require an API key or stronger workload identity.

## Deployment Requirements

- `ENVIRONMENT=production` for `aimer-web`.
- `DJANGO_ENVIRONMENT=production` for `FARM`.
- `AIMER_RAG_ENVIRONMENT=production` for `aimer-rag`.
- `DEBUG=false` and `DJANGO_DEBUG=false`.
- HTTPS `BASE_URL` / `DJANGO_BASE_URL`.
- HTTPS `OPENRAG_ENDPOINT` with `OPENRAG_API_KEY`.
- `RAG_STRICT_OPENRAG=true`.
- `RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS=false`.
- `RAG_ALLOW_EXTERNAL_PLUGINS=false`.
- `QDRANT_API_KEY` set for vector store access.
- Flower server TLS certificate paths configured for FARM federated traffic.
- Secrets provided by vault/KMS, never `.env` files on hosts.
- Images pinned by digest and signed before deployment.
- Cosign verification succeeds for every image digest admitted to production.
- Network policy denies all by default.
- Backups encrypted and restore-tested.

## Files In This Directory

- `.env.example`: non-secret variable inventory for production planning.

Do not add real secrets to this directory.
