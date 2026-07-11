# AGENTS.md

This repository is a Python 3.13 monorepo with three independently developed services plus shared infrastructure and operational documentation.

## Scope

Use this file as the default operating guide for automated coding agents working anywhere in this repository.

## Repository layout

- `services/aimer-web/`: primary Django web application, auth flows, website views, templates, static assets, and the application-local RAG module.
- `services/MAGE/`: ML/API service with FastAPI/MCP-oriented code and model/library tests.
- `services/FARM/`: supporting Django/Flower service for federated and data workflows.
- `infra/dev-stack/`: local-only Docker Compose stack and dependency containers.
- `infra/prod/`: production manifests and deployment planning baseline.
- `docs/`: cross-project architecture and regulated-operations documentation.
- `scripts/`: repository-level validation scripts.

## Working rules

- Keep changes scoped to the relevant service or infra area. Avoid unrelated cross-service edits.
- Prefer service-local documentation and manifests over assumptions from another service.
- Runtime dependencies are managed per package. Treat each service's `pyproject.toml` and `uv.lock` as authoritative.
- Do not add large datasets, generated corpora, secrets, `.env` files, or local caches to source control.
- `db.sqlite3` is a local artifact and should not be expanded or normalized as part of routine work.
- For RAG work, keep large external corpora out of Git. The lightweight seed index under `services/aimer-web/RAG/data/` may remain tracked.

## Common commands

Run commands from the repository root unless a task clearly belongs inside a service directory.

### Tests

```powershell
$env:SECRET_KEY = "local-test-secret"
uv --directory services/aimer-web run --locked python manage.py test auth website --testrunner django.test.runner.DiscoverRunner --verbosity 1
uv --directory services/aimer-web run --locked python -m pytest -q RAG/tests
uv --directory services/FARM run --locked python -m pytest -q tests
uv --directory services/MAGE run --locked python -m pytest -q -m "not slow" tests
```

### Repository validation

```powershell
python scripts/validate_production_evidence.py
python scripts/validate_deployment_manifests.py
```

### Local infrastructure

```powershell
cd infra\dev-stack
docker compose up -d
```

## Code quality conventions

- Python target version is 3.13.
- Ruff is configured at the repository root with line length 88 and target `py313`.
- Preserve existing package structure unless the task explicitly includes coordinated moves across settings, URLs, Docker, and CI.
- When changing Django packages, verify imports, settings, URL configuration, and container context together.
- When changing dependency manifests, keep Dockerfiles and automation config aligned with the real manifest locations.

## Operational context

- Production-readiness and audit artefacts live under `docs/operations/`.
- `infra/dev-stack` is for local integration only; production changes should start from `infra/prod/`.
- Secrets and production certificates must stay external to the repository.

## Agent expectations

- Read nearby README/docs files before changing a service you have not touched yet.
- Prefer the smallest viable change that matches existing patterns in that service.
- Verify with the narrowest relevant tests first, then broader validation when the change crosses boundaries.
- Call out blockers when a requested change would require inventing missing infrastructure, secrets, or external datasets.
