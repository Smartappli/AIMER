# Production Readiness - AIMER

This checklist is the release gate for regulated medical deployments. A release
must be blocked while any `P0` item is open.

## Scope

- Services: `aimer-web`, `aimer-rag`, `MAGE`, `FARM`.
- Infrastructure: PostgreSQL, vector store, RAG runtime, observability stack,
  workflow tooling and any clinical data stores.
- Regulatory drivers: NIS2 for healthcare/critical services and, when AIMER is
  used by a financial entity or an ICT provider to one, DORA.

## P0 Go/No-Go Controls

| Control | Evidence required |
| --- | --- |
| No development stack in production | Production deployment does not reference `infra/dev-stack`. |
| Strong secrets | All secrets come from a secret manager; no default or `dev-*` values. |
| HTTPS enforced | `ENVIRONMENT=production`, HTTPS `BASE_URL`, HSTS, secure cookies. |
| Internal API auth | `RAG_SERVICE_API_KEY`, `AIMER_RAG_API_KEY`, and `MAGE_API_KEY` set. |
| MFA and privileged access | MFA enforced for staff/admin/ops accounts. |
| Audit logging | Auth, admin, RAG recommendations and privileged actions exported to SIEM. |
| Backup and restore | Encrypted backups tested with measured RTO/RPO. |
| Incident reporting | 24h/72h/1-month NIS2 workflow and DORA workflow, if applicable. |
| Supply chain evidence | SBOM, container scan, IaC scan, dependency audit and signed images. |
| Clinical validation | Model/RAG output validated, drift monitored, human oversight defined. |

## Runtime Configuration

`aimer-web` now fails fast in production for unsafe settings such as `DEBUG`,
wildcard hosts, non-HTTPS `BASE_URL`, insecure cookies and RAG service calls
without `RAG_SERVICE_API_KEY`.

`FARM` also fails fast for unsafe production Django settings, including missing
production hosts, missing public HTTPS `DJANGO_BASE_URL`, weak secrets, insecure
cookies and weak HSTS settings.

`aimer-web` persists security events in `SecurityAuditEvent` and emits the same
events as structured JSON through the `aimer.security.audit` logger. Production
must route those logs to immutable SIEM storage with retention and alerting.

`aimer-rag` and `MAGE` accept unauthenticated calls only in non-production
runtime modes when their service API key variables are unset. Production startup
fails unless these keys are set:

- `AIMER_RAG_API_KEY`
- `RAG_SERVICE_API_KEY`
- `MAGE_API_KEY`

## Operational Evidence Pack

For every production release, store the following artifacts with the release:

- Git commit SHA and image digests.
- SBOM for each runtime image.
- Vulnerability scan reports and accepted-risk records.
- Database migration plan and rollback notes.
- Backup restore evidence.
- RTO/RPO test result.
- Data protection impact assessment reference.
- Clinical/model validation reference, if recommendation output is exposed.
- Third-party ICT register update.
- Incident contact matrix and escalation rota.

## Residual Items Not Solved In Code

These controls require organizational implementation:

- SIEM routing and retention policy.
- MFA provider and access reviews.
- Formal supplier due diligence and DORA register of information.
- NIS2/DORA authority notification process.
- Clinical safety case, medical device classification and AI governance.
