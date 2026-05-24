# AIMER Workspace

AIMER is a Python monorepo that groups three complementary projects behind a
single quality, dependency, and CI/CD surface.

## Repository Map

| Path | Role |
| --- | --- |
| `AIMER-ROOT/` | Primary Django web application, UI templates, static assets, and RAG code. |
| `MAGE/` | ML/API services, model-oriented tests, and backend infrastructure manifests. |
| `FARM/` | Supporting Django project for data and platform workflows. |
| `.github/` | GitHub Actions, Dependabot, release automation, and issue templates. |
| `pyproject.toml` | Workspace-level Python and tooling configuration. |
| `renovate.json` | Renovate dependency automation configuration. |

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the detailed
layout and maintenance notes.

## Continuous Integration

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
2. Choose the package you want to work on: `AIMER-ROOT`, `MAGE`, or `FARM`.
3. Install dependencies from that package directory with the local `uv.lock`.
4. Run package-specific tests and linters before opening a pull request.

Each package may carry its own README and operational constraints. Prefer local
package documentation for day-to-day commands.

## Contributing

- Keep changes scoped to the relevant package.
- Do not commit local environments, caches, `.env` files, or SQLite databases.
- Keep dependency automation in sync with real manifests and Dockerfiles.
- Ensure CI checks are green before merge.

## License

Refer to the repository license and package-level metadata.
