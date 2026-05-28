# Production Architecture

This document describes the target production architecture for AIMER. The
reference Kubernetes baseline lives in `infra/prod/kubernetes/`.

## Boundary Model

| Zone | Components | Exposure |
| --- | --- | --- |
| Public ingress | Reverse proxy / API gateway / Kubernetes Ingress | HTTPS only |
| Application tier | `aimer-web` | Public service behind ingress |
| Private service tier | `aimer-rag`, `MAGE`, `FARM` | Cluster-internal only |
| Data tier | PostgreSQL, Qdrant/vector store, backup storage | Private network only |
| Control plane | CI/CD, registry, SIEM, secret manager | Restricted operations access |
| External ICT providers | OpenRAG, email, managed databases, observability | Controlled egress only |

Only `aimer-web` should receive public traffic. `aimer-rag`, `MAGE`, `FARM`,
databases, vector stores, model stores and observability services must remain
private.

## Request Flow

```text
User -> HTTPS ingress -> aimer-web
                         |-> aimer-rag over private service DNS
                         |   |-> OpenRAG over controlled HTTPS egress
                         |   |-> Qdrant over private/controlled endpoint
                         |-> PostgreSQL over private endpoint

Internal clients -> MAGE/FARM over private service DNS only
```

## Runtime Units

| Workload | Replicas | Public | Readiness | Notes |
| --- | --- | --- | --- | --- |
| `aimer-web` | 2+ | Yes | `/readyz/` with DB check | Runs behind ingress. |
| `aimer-rag` | 2+ | No | `/readyz` with internal API key | Fails closed on OpenRAG issues. |
| `MAGE` | 1+ | No | `/healthz` | Size nodes for ML memory/CPU. |
| `FARM` | 2+ | No | `/readyz/` with DB check | Flower traffic requires TLS. |

## Deployment Rules

- Deploy images by digest, not mutable tags.
- Verify Cosign signatures before admitting images.
- Run migrations as a controlled deployment step, not as a race between pods.
- Keep service API keys in a secret manager-backed Kubernetes Secret.
- Use `NetworkPolicy` default-deny ingress and egress.
- Expose only `aimer-web` through ingress.
- Route security audit logs to SIEM from every pod.
- Set resource requests/limits and PodDisruptionBudgets for every serving
  workload.
- Use `readOnlyRootFilesystem` with writable `emptyDir` mounts only for `/tmp`,
  runtime cache and Django `staticfiles`.

## Readiness And Liveness

- Liveness checks should only prove that the process can answer.
- Readiness checks should include required serving dependencies such as the
  database or strict RAG backend.
- If readiness fails, the orchestrator must stop routing traffic to the pod.
- If liveness fails repeatedly, the orchestrator may restart the pod.

## Data And Secrets

- PostgreSQL, Qdrant/vector store and backup storage must use encryption at
  rest and private connectivity.
- Runtime credentials must be injected from the secret manager and rotated
  without code changes.
- RAG prompts, vectors and audit logs inherit the sensitivity of the underlying
  clinical or personal data.

## Scaling And Resilience

- Use at least two replicas for `aimer-web`, `aimer-rag` and `FARM` where the
  environment supports it.
- Use node pools sized for `MAGE` memory and CPU requirements.
- Keep restore drills, failover exercises and incident tabletops aligned with
  [RESILIENCE_RUNBOOK.md](RESILIENCE_RUNBOOK.md).

