# Production Readiness - AIMER

This checklist is the release gate for regulated medical deployments. A release
must be blocked while any `P0` item is open.

Use [RELEASE_EVIDENCE_PACK.md](RELEASE_EVIDENCE_PACK.md) to record the concrete
evidence and approvals for each production release.

## Scope

- Services: `aimer-web`, `aimer-rag`, `MAGE`, `FARM`.
- Infrastructure: PostgreSQL, vector store, RAG runtime, observability stack,
  workflow tooling and any clinical data stores.
- Production architecture: [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md)
  and `infra/prod/kubernetes/`.
- Regulatory drivers: NIS2 for healthcare/critical services and, when AIMER is
  used by a financial entity or an ICT provider to one, DORA.

## P0 Go/No-Go Controls

| Control | Evidence required |
| --- | --- |
| No development stack in production | Production deployment does not reference `infra/dev-stack`. |
| Strong secrets | All secrets come from a secret manager; no default or `dev-*` values. |
| HTTPS enforced | `ENVIRONMENT=production`, HTTPS `BASE_URL`, HSTS, secure cookies. |
| Internal API auth | `RAG_SERVICE_API_KEY`, `AIMER_RAG_API_KEY`, and `MAGE_API_KEY` set. |
| Grounded RAG output | `RAG_STRICT_OPENRAG=true`; ungrounded catalog recommendations disabled. |
| Safe RAG ingestion | Docling external plugins disabled; vector store writes authenticated. |
| Federated transport security | Flower servers use TLS certificates for all hospital node traffic. |
| MFA and privileged access | MFA enforced for staff/admin/ops accounts. |
| Audit logging | Auth, admin, RAG recommendations and privileged actions exported to SIEM. |
| Backup and restore | Encrypted backups tested with measured RTO/RPO. |
| Incident reporting | 24h/72h/1-month NIS2 workflow and DORA workflow, if applicable. |
| Supply chain evidence | SBOM, container scan, IaC scan, dependency audit and signed images. |
| Production evidence gate | `python scripts/validate_production_evidence.py` passes. |
| Deployment tests | `python scripts/validate_deployment_manifests.py --require-real-digests --require-real-domains --strict-network` passes for release promotion. |
| Clinical validation | Model/RAG output validated, drift monitored, human oversight defined. |
| ICT risk acceptance | Risk register reviewed and residual `High`/`Critical` risks approved. |
| Asset and supplier register | Critical assets, data classes and ICT suppliers reviewed. |
| Data protection | DPIA, retention, deletion and breach-notification paths approved. |

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

RAG recommendation output is now evidence-gated. Production must keep
`RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS=false`; requests cannot disable strict
OpenRAG retrieval in the `aimer-rag` service. Empty or irrelevant retrieval
results return no recommendation instead of catalog-only suggestions.

RAG ingestion disables Docling external plugins by default. Production ingestion
must keep `RAG_ALLOW_EXTERNAL_PLUGINS=false` and set `QDRANT_API_KEY`.

FARM federated Flower servers require TLS material in production:

- `tls_ca_cert_path`
- `tls_server_cert_path`
- `tls_server_key_path`

## Operational Evidence Pack

For every production release, store the following artifacts with the release:

- Git commit SHA and image digests.
- Completed [release evidence pack](RELEASE_EVIDENCE_PACK.md).
- Passing production evidence gate output.
- Cosign signature verification output for every runtime image digest.
- SBOM for each runtime image.
- Vulnerability scan reports and accepted-risk records.
- Database migration plan and rollback notes.
- Backup restore evidence.
- RTO/RPO test result.
- Data protection impact assessment reference.
- Clinical/model validation reference, if recommendation output is exposed.
- Third-party ICT register update.
- Incident contact matrix and escalation rota.
- Risk register review and residual-risk approval.

## Organizational Activation Required

These controls cannot be completed by repository code alone. They must be
activated by the production operator and evidenced in the release record:

- SIEM routing and retention policy, aligned with
  [DATA_PROTECTION_RETENTION.md](DATA_PROTECTION_RETENTION.md).
- MFA provider, privileged access reviews and break-glass controls.
- Formal supplier due diligence and, where applicable, DORA register of
  information using [ASSET_AND_SUPPLIER_REGISTER.md](ASSET_AND_SUPPLIER_REGISTER.md)
  as the local baseline.
- NIS2/DORA authority notification process and incident rota, maintained in
  [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md).
- Clinical safety case, medical device classification and AI governance for any
  user-facing recommendation workflow.
- Backup, restore and operational resilience drills from
  [RESILIENCE_RUNBOOK.md](RESILIENCE_RUNBOOK.md).
