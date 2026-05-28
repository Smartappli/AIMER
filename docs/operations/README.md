# Operations Evidence Index

This directory contains the operating evidence expected before a regulated
production deployment of AIMER. Treat the files as release gates: missing,
stale or unapproved evidence should block production promotion.

## Core Release Gates

| Evidence | File | Cadence |
| --- | --- | --- |
| Production go/no-go checklist | [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) | Every release |
| Release evidence pack template | [RELEASE_EVIDENCE_PACK.md](RELEASE_EVIDENCE_PACK.md) | Every release |
| NIS2 / DORA control matrix | [CONTROL_MATRIX_NIS2_DORA.md](CONTROL_MATRIX_NIS2_DORA.md) | Quarterly and every major change |
| ICT risk register | [RISK_REGISTER.md](RISK_REGISTER.md) | Monthly and every major incident/change |
| Asset and ICT supplier register | [ASSET_AND_SUPPLIER_REGISTER.md](ASSET_AND_SUPPLIER_REGISTER.md) | Monthly and before supplier changes |
| Incident response playbook | [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) | Every tabletop or incident |
| Resilience runbook | [RESILIENCE_RUNBOOK.md](RESILIENCE_RUNBOOK.md) | Restore drill at least every 90 days |
| Supply chain evidence | [SUPPLY_CHAIN.md](SUPPLY_CHAIN.md) | Every image build and release |
| Vulnerability exceptions | [VULNERABILITY_EXCEPTIONS.md](VULNERABILITY_EXCEPTIONS.md) | Every security scan and release |
| Data protection and retention | [DATA_PROTECTION_RETENTION.md](DATA_PROTECTION_RETENTION.md) | Quarterly and every data-flow change |

## Minimum Production Rule

A release is not production-ready until:

- Every `P0` item in the production checklist has evidence.
- The release evidence pack has named approvers for engineering, security,
  operations, data protection and clinical/model ownership when applicable.
- Open vulnerabilities have a dated risk acceptance in
  [VULNERABILITY_EXCEPTIONS.md](VULNERABILITY_EXCEPTIONS.md).
- The risk register and supplier register have been reviewed for the release.
- Backup restore evidence is less than 90 days old.
- Incident notification contacts and authority routes are current.

