# AIMER Workspace

AIMER is a multi-package workspace that groups three complementary projects:

- **AIMER-ROOT**: the primary web application and UI layer.
- **MAGE**: ML/AI-oriented services and model test coverage.
- **FARM**: supporting package(s) for data and platform workflows.

This repository is managed as a unified workspace to keep development, quality checks, and CI/CD aligned across all components.

## Repository Structure

- `AIMER-ROOT/` — core application code and templates.
- `MAGE/` — model-related services and tests.
  - `MAGE/api/main.py` — API gateway (REST compatibility + MCP).
  - `MAGE/api/services/` — microservices split by domain (`libraries`, `models`, `modules`).
- `FARM/` — additional workspace module(s).
- `pyproject.toml` — workspace-level configuration.

## Continuous Integration Status

[![AIMER Docker Builder](https://github.com/Smartappli/AIMER/actions/workflows/aimer_docker_builder.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/aimer_docker_builder.yml)

### MAGE

[![MAGE Coverage](https://github.com/Smartappli/AIMER/actions/workflows/mage_coverage.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/mage_coverage.yml)
[![MAGE Docker Builder](https://github.com/Smartappli/AIMER/actions/workflows/mage_docker_builder.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/mage_docker_builder.yml)

### Global CI / CD

[![Global Code Quality](https://github.com/Smartappli/AIMER/actions/workflows/global_code_quality.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_code_quality.yml)
[![CodeQL](https://github.com/Smartappli/AIMER/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/github-code-scanning/codeql)
[![Global Linter](https://github.com/Smartappli/AIMER/actions/workflows/global_linter.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_linter.yml)
[![Global Security](https://github.com/Smartappli/AIMER/actions/workflows/global_security.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_security.yml)
[![Global License Checker](https://github.com/Smartappli/AIMER/actions/workflows/global_license_check.yml/badge.svg)](https://github.com/Smartappli/AIMER/actions/workflows/global_license_check.yml)

## Getting Started

1. Clone the repository.
2. Install dependencies for the package you want to work on (`AIMER-ROOT`, `MAGE`, or `FARM`).
3. Run package-specific tests and linters before opening a pull request.

> Tip: Each package may have its own README and tooling instructions. Always prefer local package documentation for day-to-day development commands.

## Contributing

- Keep changes scoped to the relevant package.
- Follow formatting/linting conventions used by the target module.
- Ensure CI checks are green before merge.

## License

Please refer to the license information in this repository and package-level metadata.
