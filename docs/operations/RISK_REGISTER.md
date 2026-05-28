# ICT Risk Register

The risk register is the working record for NIS2 cybersecurity risk management
and DORA ICT risk management. Review it before every production release, after
major architecture or supplier changes, and after every SEV-1 or SEV-2 incident.

## Review Rules

- Each risk must have an owner, treatment decision and review date.
- `P0` risks block production unless a named executive risk owner accepts the
  residual risk in the release evidence pack.
- Review dates may not exceed 90 days for `High` or `Critical` risks.
- Corrective actions must link to issues, pull requests, change records or
  operational tasks.

## Risk Register

| ID | Risk | Scenario | Impacted services | Current controls | Residual level | Treatment | Owner | Review by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-001 | Unsafe production configuration | A service starts with debug, weak secret, wildcard host or HTTP public URL. | `aimer-web`, `FARM` | Production fail-fast settings and checklist. | Medium | Validate environment evidence in every release. | Security owner | TBD |
| R-002 | Unauthenticated internal APIs | RAG or MAGE endpoints are reachable without service credentials. | `aimer-web`, `aimer-rag`, `MAGE` | Production API key fail-fast checks. | Medium | Enforce private networking and API key rotation. | Platform owner | TBD |
| R-003 | Ungrounded AI recommendation | RAG returns catalog-only or clinically unsafe output without retrieved evidence. | `aimer-rag`, `aimer-web` | Strict OpenRAG retrieval and ungrounded output disabled by default. | High | Maintain clinical validation, monitoring and human oversight. | Clinical/model owner | TBD |
| R-004 | Supply chain compromise | Compromised dependency or image is promoted to production. | All runtime services | Dependency audit, Bandit, Trivy, SBOM and Cosign signing. | High | Verify image digests and signatures before deploy. | Security owner | TBD |
| R-005 | Incomplete incident notification | A significant incident is not escalated to the right authority in time. | All services | Incident response playbook and evidence checklist. | High | Maintain rota, authority routes and tabletop exercises. | Compliance owner | TBD |
| R-006 | Inadequate log retention | Security events cannot be reconstructed during investigation. | `aimer-web`, infra | Structured audit log events and evidence requirements. | High | Route logs to immutable SIEM with retention policy. | Operations owner | TBD |
| R-007 | Backup restore failure | Production data cannot be restored within agreed RTO/RPO. | PostgreSQL, vector store, object stores | Resilience runbook and release gate. | Critical | Run restore drills at least every 90 days. | Operations owner | TBD |
| R-008 | Third-party ICT concentration | Critical functions depend on one cloud, model, registry or managed service without exit plan. | All services | Supplier register template. | High | Maintain DORA register of information and exit plans. | Procurement/compliance owner | TBD |
| R-009 | Personal or health data over-retention | Logs, RAG stores or backups retain PHI longer than the approved purpose. | `aimer-web`, `aimer-rag`, stores | Query hashes in audit metadata; retention policy template. | High | Approve DPIA, minimization and deletion controls. | DPO | TBD |
| R-010 | Federated traffic interception | Hospital node traffic is observed or modified in transit. | `FARM` | Flower TLS enforcement in production. | Medium | Validate certificates and rotation process. | Platform owner | TBD |

## Treatment Decisions

Use these values in the `Treatment` column:

- `Mitigate`: corrective action is planned or in progress.
- `Transfer`: risk is covered by contract, insurance or supplier obligation.
- `Avoid`: deployment or feature is blocked until the risk is removed.
- `Accept`: residual risk is approved by the executive risk owner.

## Release Review

Before production promotion, record:

- Risks changed by the release.
- New assets, suppliers, data flows or critical functions.
- Open `High` or `Critical` risks and the approving executive risk owner.
- Corrective actions created from this review.

