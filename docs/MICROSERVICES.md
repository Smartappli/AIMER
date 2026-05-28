# Microservices Architecture

The repository is organized as a service-first monorepo.

## Boundaries

| Service | Path | Runtime | Responsibility |
| --- | --- | --- | --- |
| AIMER Web | `services/aimer-web` | Django/Granian | Web UI, auth, dashboard, and RAG-facing UI/API client |
| AIMER RAG | `services/aimer-web/RAG` | FastAPI/Granian | RAG health, readiness, and recommendation API |
| MAGE API | `services/MAGE` | FastAPI/Granian/MCP | ML model, encoder, augmentation, and library metadata APIs |
| FARM | `services/FARM` | Django/Granian/Flower | Federated learning and data workflow primitives |
| Dev Stack | `infra/dev-stack` | Docker Compose | Local databases, vector stores, workflow engines, and observability |

## Rules

- A service may depend on shared external infrastructure, but infrastructure
  definitions belong in `infra/`.
- Each service owns its dependency lock and Docker image.
- Cross-service calls should go through HTTP/MCP or an explicit client module,
  not direct imports between service directories.
- Keep generated data, local databases, virtual environments, and caches out of
  version control.
- Move shared Python code into an explicit shared package only after at least two
  services depend on the same stable contract.

## Current Transitional State

`services/aimer-web/RAG` is now exposed through `RAG.service:app` as a separate
HTTP runtime, built with `services/aimer-web/Dockerfile.rag`. Django uses
`website.rag_client` and will call the remote service when `RAG_SERVICE_URL` is
set, while retaining a local fallback for development and tests.

This is the first extraction step. The remaining cleanup is to move the RAG
package into its own service directory or shared package once the HTTP contract
has stabilized.

## Local App Profile

The optional Docker Compose app profile wires the services together:

```sh
docker compose --env-file infra/dev-stack/.env -f infra/dev-stack/docker-compose.yml --profile apps up --build
```

`aimer-web` calls `aimer-rag` through `RAG_SERVICE_URL=http://aimer-rag:8000`.
`RUN_DJANGO_MIGRATIONS=0` can be set on Django containers when migrations are
run as a separate deployment job.

When `AIMER_RAG_API_KEY` is set on `aimer-rag`, the RAG service requires
`Authorization: Bearer <key>` or `X-API-Key` for `/recommend` and `/readyz`.
Production startup fails if `ENVIRONMENT=production` or
`AIMER_RAG_ENVIRONMENT=production` and `AIMER_RAG_API_KEY` is missing or uses a
development/test prefix. Configure the same value as `RAG_SERVICE_API_KEY` on
`aimer-web`.

When `MAGE_API_KEY` is set, MAGE protects REST and MCP routes while leaving `/`
and `/healthz` available for liveness checks. Production startup fails if
`ENVIRONMENT=production` or `MAGE_ENVIRONMENT=production` and `MAGE_API_KEY` is
missing or uses a development/test prefix.
