# OpenRAG integration guide (strict mode)

This project is configured to prefer **strict OpenRAG retrieval** by default.

## Runtime prerequisites

Set and verify the following before using `/api/rag/recommend/`:

- `OPENRAG_ENDPOINT` (required)
- `OPENRAG_API_KEY` (optional, depends on your OpenRAG deployment)
- `RAG_COLLECTION_NAME` (optional, defaults to `rag_docs`)
- `RAG_STRICT_OPENRAG` (optional, defaults to `1`)

## Strictness policy

- Default behavior: `RAG_STRICT_OPENRAG=1` (strict)
- Optional dev fallback: `RAG_STRICT_OPENRAG=0`

When strict mode is enabled and retrieval runtime is unavailable, the API returns HTTP `503`.

## Verification commands

```bash
python -m RAG.verify_openrag
# or, once the project is installed
verify-openrag
```

- Exit code `0`: runtime ready
- Exit code `1`: runtime not ready

## Quick smoke check

```bash
curl -i "http://localhost:8000/api/rag/recommend/?q=classification+mri&top_k=2"
```

Expected behavior:
- `200` with recommendations if runtime is ready
- `503` with JSON error payload if strict OpenRAG runtime is unavailable


## Startup preflight (optional)

Set `RAG_VERIFY_ON_START=1` to run `verify-openrag` automatically at container startup before migrations/service launch.


## Runtime health endpoint

`GET /api/rag/health/` exposes OpenRAG readiness for operations tooling.

Access policy:
- `401` for unauthenticated requests
- `403` for authenticated non-staff users
- `200` for staff users with payload:
  - `ready` (boolean)
  - `status` (dependency/config flags)
