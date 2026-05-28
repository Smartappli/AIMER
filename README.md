# AIMER Workspace

AIMER is a Python monorepo that groups three complementary projects behind a
single quality and CI/CD surface. Runtime dependencies are locked per package so
the web, ML/API, and federated-learning services can evolve independently.

## Repository Map

- `services/aimer-web/`: primary Django web application, UI templates, static
  assets, and RAG code.
- `services/MAGE/`: ML/API service, model-oriented tests, and FastAPI/MCP
  endpoints.
- `services/FARM/`: supporting Django service for data and platform workflows.
- `infra/dev-stack/`: local integration stack for stores, observability,
  workflow tools, and RAG dependencies.
- `.github/`: GitHub Actions, Dependabot, release automation, and issue
  templates.
- `pyproject.toml`: repository-level Python and tooling configuration.
- `renovate.json`: Renovate dependency automation configuration.

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the detailed
layout and [docs/MICROSERVICES.md](docs/MICROSERVICES.md) for service
boundaries.

Production readiness and regulated-operation artefacts live under
[`docs/operations`](docs/operations/), with the evidence index in
[`docs/operations/README.md`](docs/operations/README.md). `infra/dev-stack` is
local-only; use [`infra/prod`](infra/prod/) as the starting checklist for
production planning.

Run the production evidence gate locally with:

```sh
python scripts/validate_production_evidence.py
```

Run the deployment manifest tests locally with:

```sh
python scripts/validate_deployment_manifests.py
```

## Continuous Integration

[![Microservices Validation](https://github.com/Smartappli/AIMER/actions/workflows/microservices_validation.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/microservices_validation.yml)
[![AIMER Docker Builder](https://github.com/Smartappli/AIMER/actions/workflows/aimer_docker_builder.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/aimer_docker_builder.yml)

### MAGE

[![MAGE Coverage](https://github.com/Smartappli/AIMER/actions/workflows/mage_coverage.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/mage_coverage.yml)
[![MAGE Docker Builder](https://github.com/Smartappli/AIMER/actions/workflows/mage_docker_builder.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/mage_docker_builder.yml)

### Global CI/CD

[![Global Code Quality](https://github.com/Smartappli/AIMER/actions/workflows/global_code_quality.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_code_quality.yml)
[![CodeQL](https://github.com/Smartappli/AIMER/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/github-code-scanning/codeql)
[![Global Linter](https://github.com/Smartappli/AIMER/actions/workflows/global_linter.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_linter.yml)
[![Global Security](https://github.com/Smartappli/AIMER/actions/workflows/global_security.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_security.yml)
[![Global License Checker](https://github.com/Smartappli/AIMER/actions/workflows/global_license_check.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_license_check.yml)

## Getting Started

1. Clone the repository.
2. Choose the service you want to work on: `services/aimer-web`, `services/MAGE`,
   or `services/FARM`.
3. Use Python 3.13.
4. Install dependencies from that package directory with the local `uv.lock`.
5. Run package-specific tests and linters before opening a pull request.

Each package may carry its own README and operational constraints. Prefer local
package documentation for day-to-day commands.

## Contributing

- Keep changes scoped to the relevant package.
- Do not commit local environments, caches, `.env` files, or SQLite databases.
- Keep dependency automation in sync with real manifests and Dockerfiles.
- Ensure CI checks are green before merge.

## License

Refer to the repository license and package-level metadata.
