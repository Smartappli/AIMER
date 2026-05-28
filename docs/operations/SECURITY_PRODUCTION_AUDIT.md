# Security Production Audit

Audit date: 2026-05-28

## Verdict

Production release is blocked.

The repository has useful production controls in place, but the current release
candidate does not meet the release bar for security and test coverage:

- Deployment manifests still contain release placeholders.
- The strict deployment gate is optional rather than enforced by default.
- Test coverage is below 80% for every service measured locally.
- Production Django validation allows SQLite database URLs.
- Local secret-bearing `.env` files exist and must be verified as untracked.

## Checks Executed

| Check | Result |
| --- | --- |
| `python scripts/validate_production_evidence.py` | Pass |
| `python scripts/validate_deployment_manifests.py` | Pass with warnings |
| `python scripts/validate_deployment_manifests.py --require-real-digests --require-real-domains --strict-network` | Fail |
| `manage.py check --deploy` for `aimer-web` with complete production env | Pass |
| `manage.py check --deploy` for `FARM` with complete production env | Pass |
| `MAGE` import with `MAGE_ENVIRONMENT=production` and API key | Pass |
| `MAGE` import with `MAGE_ENVIRONMENT=production` and no API key | Fail as expected |

Local limitations: `git`, `docker`, `kubectl`, `uv`, `ruff`, `bandit` and
`pip-audit` were not available in the local shell environment. CI contains jobs
for several of these checks, but local verification could not execute them.

## Coverage

Coverage was measured with each service virtual environment and the repository
`.coveragerc`.

| Service | Test result | Coverage | Meets 80% |
| --- | ---: | ---: | --- |
| `aimer-web` | 41 passed | 31.0% | No |
| `MAGE` | 26 passed, 2 deselected | 69.0% | No |
| `FARM` | 19 passed | 73.4% | No |

Coverage is not currently enforced with `--fail-under=80` in the CI coverage
workflows. The Sonar coverage job emits XML reports, but the workflow itself
does not fail when service coverage is below the production threshold.

## Findings

### P0 - Release manifests are not deployable as production artifacts

Strict validation fails because:

- `infra/prod/kubernetes/kustomization.yaml` contains all-zero image digests.
- `infra/prod/kubernetes/configmap.yaml` and ingress still use `example.org`.
- `infra/prod/kubernetes/network-policies.yaml` allows broad private egress
  CIDRs (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).

Action: replace placeholders with release image digests and real domains, then
scope egress to the actual database, observability, SMTP, OpenRAG and Qdrant
destinations for the target platform.

### P0 - Coverage threshold is not met

All services are below 80% with the current coverage scope:

- `aimer-web`: 31.0%
- `MAGE`: 69.0%
- `FARM`: 73.4%

Action: add focused tests for auth flows, RAG service paths, MAGE service
branches, FARM federated runner/settings validation, and enforce
`coverage report --fail-under=80` in CI once the threshold is reachable.

### P1 - Strict deployment validation is opt-in

`.github/workflows/deployment_tests.yml` runs strict checks only when manually
dispatched with `require_real_digests`, `require_real_domains` and
`strict_network` enabled.

Action: make strict mode mandatory for release branches/tags or for any
production promotion workflow. Warning-only checks are not sufficient evidence
for a regulated go-live.

### P1 - Production Django settings accept SQLite

`aimer-web` and `FARM` both accept `sqlite://` URLs in production when the rest
of the production environment is complete. Local `manage.py check --deploy`
passes with SQLite database URLs.

Action: reject SQLite in `ENVIRONMENT=production` and
`DJANGO_ENVIRONMENT=production`; require PostgreSQL with TLS parameters for
regulated production deployments.

### P1 - Local secret-bearing environment files exist

The repository working tree contains:

- `services/aimer-web/.env`
- `services/FARM/.env`
- `services/MAGE/.env`
- `infra/dev-stack/.env.ci`

`.gitignore` ignores `.env`, but local `git` was unavailable, so tracking
status could not be verified.

Action: verify these files were never committed, rotate any values that may
have been exposed, and keep production values only in the secret manager or
sealed-secret workflow.

### P2 - MAGE runtime image contains build toolchain packages

`services/MAGE/Dockerfile` installs `build-essential` and `git` in the runtime
image. This increases attack surface and patch burden.

Action: move build tooling to a builder stage, copy only the virtual
environment/runtime artifacts into the final image, and keep the runtime image
minimal.

### P2 - Supply-chain scanning has blind spots

The secret scan uses TruffleHog `--only-verified`, and Trivy filesystem/image
scans ignore unfixed HIGH/CRITICAL findings. This reduces false positives but
can hide risk requiring explicit acceptance.

Action: keep the exception register current, review ignored vulnerabilities
before release, and consider a release-only job that reports unverified secrets
and unfixed critical vulnerabilities for human triage.

### P2 - Build inputs are mutable

Production Dockerfiles use tag-based base images such as `python:3.13-slim`,
and many GitHub Actions are version-tag pinned rather than full-SHA pinned.

Action: pin base images by digest and pin production CI actions by SHA for
release workflows.

## Positive Controls Observed

- Production evidence validator passes.
- Kubernetes workloads use non-root execution, no privilege escalation,
  read-only root filesystems and dropped Linux capabilities.
- Production config requires key secrets and secure cookie/HSTS settings.
- `MAGE` refuses production startup without `MAGE_API_KEY`.
- Docker builds emit SBOM/provenance and sign image digests with Cosign.
- Vulnerability exceptions are documented with review dates.
- Network policies include default deny ingress and egress.

## Go-Live Minimum Bar

Before production promotion:

1. Run deployment manifest validation in strict mode with real release values.
2. Enforce coverage `--fail-under=80` and show all services above threshold.
3. Reject SQLite in production settings.
4. Verify no local `.env` file or SQLite database artifact is tracked.
5. Run CI security scans with current advisory data and review every exception.
6. Pin production base images and release-critical Actions by immutable digest or
   SHA.
