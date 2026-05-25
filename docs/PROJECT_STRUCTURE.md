# Project Structure

This repository is a Python 3.13 monorepo. The current readable boundary is by
product area, not by shared source tree.

## Top-Level Areas

- `services/aimer-web/`: main Django web service.
  Contains the `AIMER` settings package, auth/website apps, templates, static
  assets, and RAG modules.
- `services/MAGE/`: ML/API service.
  Contains FastAPI/MCP API code, model tests, and service modules.
- `services/FARM/`: supporting Django service.
  Contains the `FARM` settings package and federated/data workflow code.
- `infra/dev-stack/`: local integration infrastructure.
  Contains Docker Compose and supporting Dockerfiles/config for Postgres,
  Qdrant, Airflow, MLflow, OMOP, and related local dependencies.
- `.github/`: repository automation.
  Contains CI workflows, Dependabot, Renovate workflow, release drafting, and
  issue templates.
- `docs/`: contributor-facing architecture notes.
  Keep cross-project explanations here instead of scattering them through
  configs.

## Dependency Automation

The repository root owns shared tooling configuration. Runtime dependencies are
resolved independently in each product directory because the Django, ML/API, and
federated-learning services do not need a single compatible dependency graph.
Keep package-level `pyproject.toml` and `uv.lock` files authoritative for local
development, Docker builds, and CI jobs.

Dependency automation should mirror the real manifests in the repository:

| Ecosystem | Paths |
| --- | --- |
| GitHub Actions | `/` for `.github/workflows` |
| uv | `/`, `/services/aimer-web`, `/services/MAGE`, `/services/FARM` |
| pre-commit | `/` |
| Docker | See Docker paths below. |
| Docker Compose | `/infra/dev-stack` |

Docker paths:

- `/services/aimer-web`
- `/services/FARM`
- `/services/MAGE`
- `/infra/dev-stack/airflow`
- `/infra/dev-stack/mlflow`
- `/infra/dev-stack/qdrant`

`renovate.json` is also enabled. If both Renovate and Dependabot stay active,
keep their schedules and labels intentional so maintainers do not receive
duplicate update noise.

## Local Artifacts

The following files and directories are development artifacts and should stay
out of version control:

- `.env`
- `.venv/`
- `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`
- `__pycache__/`
- `.coverage`
- `db.sqlite3`

The root `.gitignore` already covers these patterns. If any of them are tracked
in Git, remove them from the index instead of expanding the ignore list.

## RAG Data

`services/aimer-web/RAG` is currently an application-internal RAG module used by the
Django app. Keep code, tests, scripts, and small metadata indexes in the module.
Large source corpora such as PDFs should be treated as external datasets or
release artifacts. If large files are already tracked, remove them from the Git
index in a dedicated change after agreeing on the replacement storage location.

Runtime images exclude the local RAG corpora and generated extraction outputs.
The bundled `RAG/data/timm_model_articles.json` seed index is the lightweight
fallback that should remain in Git. Configure ingestion/runtime integrations
through environment variables instead of editing code:

- `RAG_PDF_DIR`, `RAG_MARKDOWN_DIR`, `RAG_FIGURES_DIR`,
  `RAG_FIGURE_DESCRIPTIONS_DIR`, `RAG_TABLES_DIR`
- `OLLAMA_BASE_URL`, `RAG_VISION_MODEL`, `RAG_EMBEDDING_MODEL`,
  `RAG_SPARSE_EMBEDDING_MODEL`
- `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_COLLECTION_NAME`

## Restructuring Rules

- Move Django packages only with corresponding updates to settings modules,
  URL configuration, Docker contexts, and CI workflows.
- Keep package-level `pyproject.toml` and `uv.lock` files beside the code they
  resolve.
- Keep `infra/dev-stack/docker-compose.yml` with the backend service Dockerfiles
  it orchestrates.
- Prefer adding structure documentation before moving runtime code. The current
  package names are part of import paths and deployment commands.
