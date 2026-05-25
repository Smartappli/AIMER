# Services

Application code is grouped by independently deployable service.

## Service Map

- `aimer-web/`: Django web service, templates, static assets, auth, website, and
  the current application-local RAG module.
- `MAGE/`: FastAPI/MCP ML API service.
- `FARM/`: Django/FLOWER-oriented service for federated and data workflows.

Each service owns its own `pyproject.toml`, `uv.lock`, Dockerfile, tests, and
runtime commands. Keep cross-service infrastructure in `../infra` rather than
inside a service directory.

## Local Commands

From the repository root:

```powershell
.\.venv\Scripts\uv.exe --directory services/aimer-web run python manage.py test website
.\.venv\Scripts\uv.exe --directory services/FARM run python -m pytest -q tests
.\.venv\Scripts\uv.exe --directory services/MAGE run python -m pytest -q tests
```
