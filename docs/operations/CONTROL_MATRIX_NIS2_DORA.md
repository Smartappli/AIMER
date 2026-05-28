# Control Matrix - NIS2 / DORA

This matrix maps repository controls to regulated production expectations.
Legal applicability depends on the deploying entity and country transposition.

| Area | NIS2 / DORA expectation | Current repository control | Required production action |
| --- | --- | --- | --- |
| Governance | Management accountability and ICT risk framework | Production readiness checklist, release evidence template and risk register | Assign accountable owner and approve residual risk per release. |
| Asset inventory | Know critical systems and dependencies | `docs/MICROSERVICES.md` service boundaries and asset/supplier register | Maintain deployment CMDB with owners, data class and criticality. |
| Access control | Least privilege and strong auth | Django auth, staff gating for RAG health, internal API key fail-fast | Enforce MFA, SSO, admin reviews and break-glass controls. |
| Incident reporting | Significant incident notification workflows | Incident playbook with NIS2 and DORA timing references | Wire SOC/CSIRT process to national authority contacts and templates. |
| ICT risk management | Policies, procedures and controls | Production settings, internal service auth, RAG strict retrieval, Flower TLS enforcement and ICT risk register | Review risks monthly, after incidents and before every release. |
| Third-party risk | Register and monitor ICT providers | Asset/supplier register, dependency automation and supply-chain docs | Maintain formal DORA register of information where applicable. |
| Supply chain | Vulnerability and dependency management | `pip-audit`, Bandit, secret scan, Trivy, SBOM and Cosign signing | Verify digests, signatures and scan evidence before deploy. |
| Resilience testing | Test backup, restore and operational resilience | Resilience runbook and Docker smoke tests | Run restore drills, failover tests and tabletop exercises on cadence. |
| Logging | Detect and investigate security events | `SecurityAuditEvent` model, structured `aimer.security.audit` logs, RAG/admin action audit | Route logs to immutable SIEM storage with retention and alerting. |
| Data protection | Protect medical data and PHI | RAG query hashes in audit metadata and data protection/retention baseline | Complete DPIA, encryption, minimization and deletion controls. |
| AI/model safety | Human oversight and validation | RAG safety notice, strict retrieval, ungrounded recommendations blocked by default | Add clinical validation, drift monitoring and escalation path. |

## Minimum Release Evidence

- Signed risk acceptance for open vulnerabilities.
- SBOM and vulnerability scan per image.
- Restore drill result less than 90 days old.
- Access review less than 90 days old.
- Incident tabletop less than 180 days old.
- Supplier register updated for all critical ICT services.
- Release evidence pack approved by engineering, operations, security and
  data-protection owners.
