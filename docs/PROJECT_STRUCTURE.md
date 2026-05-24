# Project Structure

This repository is a Python monorepo. The current readable boundary is by
product area, not by shared source tree.

## Top-Level Areas

- `AIMER-ROOT/`: main Django web application.
  Contains the `AIMER` settings package, auth/website apps, templates, static
  assets, and RAG modules.
- `MAGE/`: ML and API services.
  Contains FastAPI/MCP API code, model tests, service modules, and backend
  Docker Compose infrastructure.
- `FARM/`: supporting Django application.
  Contains the `FARM` settings package and federated/data workflow code.
- `.github/`: repository automation.
  Contains CI workflows, Dependabot, Renovate workflow, release drafting, and
  issue templates.
- `docs/`: contributor-facing architecture notes.
  Keep cross-project explanations here instead of scattering them through
  configs.

## Dependency Automation

Dependency automation should mirror the real manifests in the repository:

| Ecosystem | Paths |
| --- | --- |
| GitHub Actions | `/` for `.github/workflows` |
| uv | `/`, `/AIMER-ROOT`, `/MAGE`, `/FARM` |
| pip | `/MAGE`, limited to standalone requirements files |
| pre-commit | `/` |
| Docker | See Docker paths below. |
| Docker Compose | `/MAGE/backend` |

Docker paths:

- `/AIMER-ROOT`
- `/FARM`
- `/MAGE`
- `/MAGE/backend/airflow`
- `/MAGE/backend/mlflow`
- `/MAGE/backend/qdrant`

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

## Restructuring Rules

- Move Django packages only with corresponding updates to settings modules,
  URL configuration, Docker contexts, and CI workflows.
- Keep package-level `pyproject.toml` and `uv.lock` files beside the code they
  resolve.
- Keep `MAGE/backend/docker-compose.yml` with the backend service Dockerfiles it
  orchestrates.
- Prefer adding structure documentation before moving runtime code. The current
  package names are part of import paths and deployment commands.
