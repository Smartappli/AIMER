# Copyright (c) 2026 AIMER contributors.
"""Runtime verifier for mandatory OpenRAG integration."""

from __future__ import annotations

from typing import TypedDict

from RAG.healthcheck import is_rag_runtime_ready, rag_runtime_health


class RuntimeStatusPayload(TypedDict):
    """Structured runtime readiness payload."""

    ready: bool
    status: dict[str, bool]


REQUIRED_KEYS = (
    "openrag_installed",
    "langchain_ollama_installed",
    "langchain_core_installed",
    "dotenv_installed",
    "openrag_endpoint_set",
    "openrag_endpoint_valid",
)


def runtime_status() -> RuntimeStatusPayload:
    """Return machine-readable OpenRAG runtime readiness payload."""
    status = rag_runtime_health()
    ready = bool(all(status[name] for name in REQUIRED_KEYS))
    return {"ready": ready, "status": status}


def format_report() -> str:
    """Build a human-readable OpenRAG readiness report."""
    payload = runtime_status()
    status = payload["status"]
    lines = ["OpenRAG runtime verification:"]
    for key in sorted(status):
        mark = "OK" if status[key] else "MISSING"
        lines.append(f"- {key}: {mark}")
    all_ready = bool(payload["ready"])
    lines.append(f"- runtime_ready: {'YES' if all_ready else 'NO'}")
    if not all_ready:
        lines.append(
            "- action: configure OPENRAG_ENDPOINT and install missing dependencies "
            "before using /api/rag/recommend/ "
            "(or set RAG_STRICT_OPENRAG=0 for non-strict dev fallback)."
        )
    return "\n".join(lines)


def main() -> int:
    """Print readiness report and return shell status code."""
    report = format_report()
    print(report)
    return 0 if is_rag_runtime_ready() else 1


if __name__ == "__main__":
    raise SystemExit(main())
