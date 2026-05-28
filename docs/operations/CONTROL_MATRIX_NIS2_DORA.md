# Control Matrix - NIS2 / DORA

This matrix maps repository controls to regulated production expectations.
Legal applicability depends on the deploying entity and country transposition.

| Area | NIS2 / DORA expectation | Current repository control | Required production action |
| --- | --- | --- | --- |
| Governance | Management accountability and ICT risk framework | Production readiness checklist | Assign accountable owner and approve residual risk. |
| Asset inventory | Know critical systems and dependencies | `docs/MICROSERVICES.md` service boundaries | Maintain CMDB with owners, data class and criticality. |
| Access control | Least privilege and strong auth | Django auth, staff gating for RAG health | Enforce MFA, SSO, admin reviews and break-glass controls. |
| Incident reporting | Significant incident notification workflows | Incident playbook template | Wire SOC/CSIRT process to national authority timelines. |
| ICT risk management | Policies, procedures and controls | Production settings fail fast | Add risk register, threat model and recurring review cadence. |
| Third-party risk | Register and monitor ICT providers | Dependency automation and docs | Maintain DORA register of information where applicable. |
| Supply chain | Vulnerability and dependency management | `pip-audit`, Bandit, secret scan | Add SBOM, image signing, IaC/container scan gates. |
| Resilience testing | Test backup, restore and operational resilience | Docker smoke tests | Add restore drills, failover tests and tabletop exercises. |
| Logging | Detect and investigate security events | `SecurityAuditEvent` model | Export immutable logs to SIEM with retention and alerting. |
| Data protection | Protect medical data and PHI | No PHI-specific control in code | Add encryption, minimization, DPIA and data retention rules. |
| AI/model safety | Human oversight and validation | RAG safety notice | Add clinical validation, drift monitoring and escalation path. |

## Minimum Release Evidence

- Signed risk acceptance for open vulnerabilities.
- SBOM and vulnerability scan per image.
- Restore drill result less than 90 days old.
- Access review less than 90 days old.
- Incident tabletop less than 180 days old.
- Supplier register updated for all critical ICT services.
