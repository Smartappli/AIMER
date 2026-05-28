# Resilience Runbook

This runbook defines the minimum operational resilience evidence for production
promotion. It covers backup, restore, failover and tabletop exercises for the
AIMER services and their data stores.

## Targets

Set environment-specific targets before go-live.

| Service/data store | RTO | RPO | Backup frequency | Restore drill cadence | Owner |
| --- | --- | --- | --- | --- | --- |
| `aimer-web` PostgreSQL | TBD | TBD | TBD | 90 days | TBD |
| `FARM` PostgreSQL | TBD | TBD | TBD | 90 days | TBD |
| Qdrant/vector store | TBD | TBD | TBD | 90 days | TBD |
| Object/model artifact storage | TBD | TBD | TBD | 90 days | TBD |
| SIEM/audit log archive | TBD | TBD | TBD | 180 days | TBD |

## Backup Requirements

- Backups must be encrypted at rest with keys managed outside the workload.
- Backup credentials must be separate from runtime application credentials.
- Backups must cover databases, vector stores, object/model artifacts and
  configuration needed for recovery.
- Production backup jobs must alert on failure.
- Backup deletion and retention must follow
  [DATA_PROTECTION_RETENTION.md](DATA_PROTECTION_RETENTION.md).

## Restore Drill Procedure

Run the drill in an isolated environment that cannot write to production.

1. Select the backup set and record timestamp, source and data classes.
2. Validate backup integrity and encryption key access.
3. Restore PostgreSQL databases.
4. Restore vector store collections and object/model artifacts.
5. Apply database migrations required by the target release.
6. Start services with production-equivalent safe configuration.
7. Run smoke tests for login, RAG health, MAGE health and FARM health.
8. Verify security audit logs are produced and collected.
9. Measure restore duration and data loss window.
10. Record gaps, corrective actions and owner.

## Failover and Degradation Scenarios

Test at least the following scenarios before go-live and then on the stated
cadence:

| Scenario | Cadence | Expected evidence |
| --- | --- | --- |
| PostgreSQL restore from encrypted backup | 90 days | Restore log, RTO/RPO measurement |
| Qdrant/vector store restore | 90 days | Collection count and sample query validation |
| Loss of OpenRAG endpoint | 180 days | RAG fails closed or returns no ungrounded recommendation |
| Loss of MAGE service | 180 days | Caller timeout and user-safe degradation |
| FARM Flower certificate expiry or mismatch | 180 days | Production server refuses insecure traffic |
| Container registry unavailable | 180 days | Deployment can use approved mirrored digests |
| SIEM ingestion interruption | 180 days | Alert, backfill plan and no data loss statement |

## Tabletop Exercises

Run a tabletop at least every 180 days and after major architecture changes.

Minimum scenarios:

- Ransomware or destructive compromise of an application host.
- PHI or personal data exposure through logs, RAG prompts or backups.
- Compromised dependency or container image after deployment.
- Major cloud or supplier outage affecting a critical function.
- Significant incident requiring NIS2 and/or DORA reporting.

Record participants, timeline, decisions, missed evidence, authority
notification route, customer/patient communication route and corrective actions.

