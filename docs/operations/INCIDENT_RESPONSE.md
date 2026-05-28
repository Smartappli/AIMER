# Incident Response Playbook

## Severity Classes

| Severity | Description | Examples |
| --- | --- | --- |
| SEV-1 | Patient safety, PHI breach, major outage or active compromise | Ransomware, data exfiltration, production outage. |
| SEV-2 | Material degradation or contained security incident | RAG unavailable, suspicious admin activity. |
| SEV-3 | Limited impact, no sensitive data exposure | Failed deployment, isolated dependency alert. |

## First Hour

1. Open an incident record and assign incident commander.
2. Freeze risky deployments unless needed for containment.
3. Preserve logs, image digests and affected data snapshots.
4. Triage impact: confidentiality, integrity, availability, patient safety.
5. Contain: revoke tokens, isolate service, rotate secrets, block ingress.
6. Notify DPO, security owner, clinical owner and legal/compliance.

## Regulated Notification Timelines

- NIS2: prepare early warning within 24 hours where the national law requires
  it, follow-up notification around 72 hours, and final report around one month.
- DORA, when applicable: use the competent authority templates and timelines
  for major ICT-related incidents and significant cyber threats.

## Evidence To Capture

- Incident timeline in UTC.
- Affected services, versions and image digests.
- Users, data classes and patient/clinical workflows affected.
- Indicators of compromise.
- Logs from Django audit, reverse proxy, database, RAG, MAGE and infrastructure.
- Containment actions and approvers.
- Recovery validation and residual risks.

## Recovery Exit Criteria

- Root cause known or compensating control approved.
- Secrets rotated where exposure is plausible.
- Backlog item created for every corrective action.
- Monitoring confirms stable service and no continuing compromise.
- Compliance owner approves external notification status.
