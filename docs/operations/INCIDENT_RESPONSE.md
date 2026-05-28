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

- NIS2: for a significant incident, prepare an early warning without undue
  delay and within 24 hours after awareness, an incident notification within
  72 hours after awareness, and a final report no later than one month after
  the incident notification. If the incident is still ongoing at final-report
  time, submit a progress report and then a final report within one month after
  the incident is handled.
- DORA, when applicable: for a major ICT-related incident, submit the initial
  notification as early as possible, within four hours after classifying it as
  major and no later than 24 hours after awareness. Submit the intermediate
  report no later than 72 hours after the initial notification, and the final
  report no later than one month after the intermediate report or latest updated
  intermediate report. Use the competent authority templates and delay notice
  process when a deadline cannot be met.

## Classification Checklist

Record the first assessment using these factors:

- Operational disruption, service downtime and critical functions affected.
- Confidentiality, integrity, availability or authenticity impact on data.
- Patient, user, client or counterparty impact.
- Financial, legal, contractual and reputational impact.
- Geographic or cross-border impact.
- Indicators of compromise and suspected malicious activity.
- Third-party ICT supplier involvement.

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

## Regulatory References

- NIS2 Directive (EU) 2022/2555, Article 23 incident reporting:
  https://eur-lex.europa.eu/eli/dir/2022/2555/oj
- European Commission NIS2 FAQ:
  https://digital-strategy.ec.europa.eu/en/faqs/directive-measures-high-common-level-cybersecurity-across-union-nis2-directive-faqs
- DORA Regulation (EU) 2022/2554, Articles 17 to 20 incident management and
  reporting: https://eur-lex.europa.eu/eli/reg/2022/2554/oj
- Commission Delegated Regulation (EU) 2025/301, Article 5 DORA reporting
  time limits:
  https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32025R0301
