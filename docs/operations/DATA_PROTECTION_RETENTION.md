# Data Protection and Retention

This policy baseline supports production deployment where AIMER may process
personal data, health data, clinical metadata or security logs. Deployment
owners must complete a DPIA and align retention with local law, contracts and
clinical safety requirements before go-live.

## Data Classes

| Data class | Examples | Default handling | Owner |
| --- | --- | --- | --- |
| Account data | Usernames, emails, auth metadata | Encrypt in transit and at rest; restrict admin access | DPO / product owner |
| Security audit data | Login failures, admin mutations, RAG/admin audit metadata | Send to immutable SIEM; restrict read access | Security owner |
| RAG query data | Prompts, query hashes, retrieved document metadata | Minimize storage; avoid PHI in logs; protect vector stores | Data owner |
| Embeddings/vector data | Document embeddings and metadata | Treat as sensitive when source data is sensitive | Data owner |
| Clinical/model data | Clinical payloads, model inputs/outputs, validation evidence | Require clinical validation and DPIA controls | Clinical/model owner |
| Backup data | Database, vector and artifact backups | Encrypt, access-control and retention-limit | Operations owner |

## Retention Baseline

Set explicit production values before go-live.

| Data set | Default target | Deletion or archive rule | Evidence |
| --- | --- | --- | --- |
| Application account records | Contract/legal requirement | Delete or anonymize after account closure where allowed | Data retention job or manual evidence |
| Security audit records in application DB | Short operational window, TBD | Export to SIEM before pruning local DB records | SIEM export and pruning record |
| SIEM security logs | Regulatory/security requirement, TBD | Immutable retention with access review | SIEM retention policy |
| RAG prompts and query text | Minimize; avoid persistent PHI unless approved | Do not log raw PHI; redact or hash where feasible | Log sampling evidence |
| Vector store documents/embeddings | Source-data retention period | Delete derived vectors when source data expires | Collection deletion record |
| Backups | RTO/RPO and legal hold requirements | Expire encrypted backups after approved retention | Backup lifecycle policy |
| Release evidence | Audit requirement, TBD | Archive with access control | Release evidence pack |

## DPIA Checklist

Complete before production if personal data, PHI or clinical data may be
processed:

- Processing purpose, legal basis and data subjects identified.
- Data categories, sources, recipients and storage locations mapped.
- RAG, model and vector-store data flows documented.
- Data minimization and pseudonymization controls defined.
- Cross-border transfers and subprocessors reviewed.
- Access controls, MFA and privileged access reviews defined.
- Retention and deletion mechanisms approved.
- Data subject request process documented.
- Breach notification route aligned with incident response.
- Clinical safety and human oversight documented when recommendations are
  exposed to users.

## Production Controls

- Do not use development fixtures or local demo data in production.
- Do not expose PostgreSQL, vector stores, MLflow, object storage or workflow
  tools directly to the public internet.
- Do not log raw passwords, API keys, tokens or full clinical prompts.
- Send structured security audit events to SIEM before pruning local records.
- Require explicit approval before using external plugins or unmanaged model
  endpoints with sensitive data.

