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
$env:SECRET_KEY = "local-test-secret"
uv --directory services/aimer-web run --locked python manage.py test auth website --testrunner django.test.runner.DiscoverRunner --verbosity 1
uv --directory services/aimer-web run --locked python -m pytest -q RAG/tests
uv --directory services/FARM run --locked python -m pytest -q tests
uv --directory services/MAGE run --locked python -m pytest -q -m "not slow" tests
```
