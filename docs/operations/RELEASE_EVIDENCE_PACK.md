# Release Evidence Pack Template

Create one completed evidence pack for every production release. Store the
completed pack with the release record, image digest artifacts and approvals.

## Release Summary

| Field | Value |
| --- | --- |
| Release ID | TBD |
| Release date | TBD |
| Git commit SHA | TBD |
| Change owner | TBD |
| Production environment | TBD |
| Services changed | `aimer-web` / `aimer-rag` / `MAGE` / `FARM` / infrastructure |
| Data classes affected | None / operational / personal data / PHI / clinical |
| Critical functions affected | TBD |
| Rollback owner | TBD |

## Go / No-Go

| Gate | Evidence link | Owner | Status |
| --- | --- | --- | --- |
| Production readiness checklist complete | `docs/operations/PRODUCTION_READINESS.md` | TBD | TBD |
| Risk register reviewed | `docs/operations/RISK_REGISTER.md` | TBD | TBD |
| Asset and supplier register reviewed | `docs/operations/ASSET_AND_SUPPLIER_REGISTER.md` | TBD | TBD |
| Database migration plan reviewed | TBD | TBD | TBD |
| Rollback plan tested or reviewed | TBD | TBD | TBD |
| Backup restore evidence current | TBD | TBD | TBD |
| RTO/RPO targets still valid | `docs/operations/RESILIENCE_RUNBOOK.md` | TBD | TBD |
| Vulnerability exceptions reviewed | `docs/operations/VULNERABILITY_EXCEPTIONS.md` | TBD | TBD |
| Data protection impact reviewed | `docs/operations/DATA_PROTECTION_RETENTION.md` | TBD | TBD |
| Clinical/model validation reviewed | TBD | TBD | TBD |
| Incident rota and authority paths current | `docs/operations/INCIDENT_RESPONSE.md` | TBD | TBD |

## Runtime Image Evidence

| Service | Image reference by digest | SBOM artifact | Scan artifact | Cosign verification | Approved |
| --- | --- | --- | --- | --- | --- |
| `aimer-web` | TBD | TBD | TBD | TBD | TBD |
| `aimer-rag` | TBD | TBD | TBD | TBD | TBD |
| `MAGE` | TBD | TBD | TBD | TBD | TBD |
| `FARM` | TBD | TBD | TBD | TBD | TBD |

Every deployed image must use an immutable `name@sha256:<digest>` reference.
Mutable tags such as `latest` may exist in registries, but must not be the
production deployment input.

## Configuration Evidence

| Control | Evidence |
| --- | --- |
| Secrets come from a secret manager | TBD |
| `ENVIRONMENT=production` for `aimer-web` | TBD |
| `DJANGO_ENVIRONMENT=production` for `FARM` | TBD |
| `AIMER_RAG_ENVIRONMENT=production` for `aimer-rag` | TBD |
| `MAGE_ENVIRONMENT=production` for `MAGE` | TBD |
| `FARM_FEDERATED_ENVIRONMENT=production` for Flower traffic | TBD |
| HTTPS public base URLs configured | TBD |
| HSTS and secure cookies enabled | TBD |
| Internal service API keys configured | TBD |
| `RAG_STRICT_OPENRAG=true` | TBD |
| `RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS=false` | TBD |
| `RAG_ALLOW_EXTERNAL_PLUGINS=false` | TBD |
| `QDRANT_API_KEY` configured | TBD |
| FARM Flower TLS paths configured | TBD |
| SIEM log route tested | TBD |
| Backup encryption key custody checked | TBD |

## Approval

| Role | Name | Decision | Date |
| --- | --- | --- | --- |
| Engineering owner | TBD | Go / No-go | TBD |
| Operations owner | TBD | Go / No-go | TBD |
| Security owner | TBD | Go / No-go | TBD |
| Data protection owner | TBD | Go / No-go | TBD |
| Clinical/model owner, if applicable | TBD | Go / No-go | TBD |
| Executive risk owner | TBD | Go / No-go | TBD |

## Post-Release Evidence

| Check | Evidence | Due |
| --- | --- | --- |
| Deployment completed against approved digests | TBD | Same day |
| Smoke tests passed | TBD | Same day |
| Security logs visible in SIEM | TBD | Same day |
| Error budgets and alerts reviewed | TBD | Same day |
| Backup job succeeded after deployment | TBD | Within 24 hours |
| Release retrospective, if needed | TBD | Within 5 business days |
