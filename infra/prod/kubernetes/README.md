# Kubernetes Production Baseline

This baseline is a hardened reference for production orchestration. It is not a
complete environment: operators must replace domains, image digests, private
CIDR ranges, secret provisioning and ingress labels for the target platform.

## Apply Order

1. Create or connect the secret manager integration that materializes the
   `aimer-prod-secrets` Kubernetes Secret.
2. Replace image digests in `kustomization.yaml` with approved release digests.
3. Replace `aimer.example.org`, `farm.example.org` and external endpoints in
   `configmap.yaml`.
4. Label the ingress-controller namespace with
   `network.aimer.io/ingress=true`.
5. Adjust private egress CIDR ranges in `network-policies.yaml`.
6. Run the production evidence gate from the repository root:

```sh
python scripts/validate_production_evidence.py
```

## Runtime Architecture

- `aimer-web` is the only public service and is exposed through Ingress.
- `aimer-rag`, `MAGE` and `FARM` are private `ClusterIP` services.
- Network policies default-deny ingress and egress in the namespace.
- Workloads run as non-root, drop Linux capabilities, use the runtime default
  seccomp profile and mount writable `emptyDir` volumes only for `/tmp`,
  runtime cache and Django `staticfiles`.
- Django pods run migrations outside the serving deployment; do not run multiple
  concurrent migration jobs during deployment.
- RAG readiness checks `/readyz` using the configured internal API key from the
  pod environment, while `/healthz` remains unauthenticated liveness.

## Secret Contract

The `aimer-prod-secrets` Secret must provide at least these keys:

- `SECRET_KEY`
- `DJANGO_SECRET_KEY`
- `DATABASE_URL`
- `DJANGO_DATABASE_URL`
- `EMAIL_HOST_USER`
- `EMAIL_HOST_PASSWORD`
- `RAG_SERVICE_API_KEY`
- `AIMER_RAG_API_KEY`
- `MAGE_API_KEY`
- `OPENRAG_API_KEY`
- `QDRANT_API_KEY`

Use an external secret manager or sealed-secret workflow. Do not commit real
secret values to this directory.

