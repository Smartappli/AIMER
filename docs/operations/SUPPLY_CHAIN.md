# Supply Chain Release Evidence

Production releases must deploy immutable image digests, not mutable tags.
The Docker publish workflows generate SBOM/provenance metadata, sign each pushed
image digest with Sigstore Cosign keyless signing, and upload a digest evidence
artifact.

## Signed Images

The publish workflows sign these images after a successful push:

- `smartappli/aimer`
- `smartappli/aimer-rag`
- `smartappli/mage-api`
- `smartappli/farm`

Each workflow grants `id-token: write` only so Cosign can obtain a GitHub OIDC
identity for keyless signing. No long-lived signing key should be stored in the
repository or CI secrets.

## Release Gate

Before production deployment, store the following in the
[release evidence pack](RELEASE_EVIDENCE_PACK.md):

- Git commit SHA.
- Image digest artifact from the Docker workflow.
- SBOM artifact for each runtime image.
- Trivy scan result or accepted-risk record.
- Cosign verification output for each image digest.

Use the digest artifact as the deployable reference, for example:

```sh
cosign verify \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp 'https://github.com/Smartappli/AIMER/.github/workflows/.*@refs/heads/master' \
  smartappli/aimer@sha256:<digest>
```

Deployments must fail closed when signature verification, SBOM retrieval or
vulnerability review evidence is missing.

## Supplier and Registry Controls

Container registries, CI/CD systems, hosted scanners and artifact stores are ICT
suppliers for production purposes. Record them in
[ASSET_AND_SUPPLIER_REGISTER.md](ASSET_AND_SUPPLIER_REGISTER.md), including
contract owner, jurisdiction, exit plan and incident contact route.
