# Microservices Architecture

The repository is organized as a service-first monorepo.

## Boundaries

| Service | Path | Runtime | Responsibility |
| --- | --- | --- | --- |
| AIMER Web | `services/aimer-web` | Django/Granian | Web UI, auth, dashboard, and current RAG-facing UI/API |
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

`services/aimer-web/RAG` remains inside the web service because Django views and
tests import it directly. The next clean extraction would be a dedicated RAG
service with an HTTP API, after the current in-process API has a stable request
and response contract.

