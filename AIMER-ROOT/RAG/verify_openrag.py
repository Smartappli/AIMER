# Copyright (c) 2026 AIMER contributors.
"""Runtime verifier for mandatory OpenRAG integration."""

from __future__ import annotations

from RAG.healthcheck import is_rag_runtime_ready, rag_runtime_health


REQUIRED_KEYS = (
    "openrag_installed",
    "langchain_ollama_installed",
    "langchain_core_installed",
    "dotenv_installed",
    "openrag_endpoint_set",
)


def format_report() -> str:
    """Build a human-readable OpenRAG readiness report."""
    status = rag_runtime_health()
    lines = ["OpenRAG runtime verification:"]
    for key in sorted(status):
        mark = "OK" if status[key] else "MISSING"
        lines.append(f"- {key}: {mark}")
    all_ready = all(status[name] for name in REQUIRED_KEYS)
    lines.append(f"- runtime_ready: {'YES' if all_ready else 'NO'}")
    if not all_ready:
        lines.append(
            "- action: configure OPENRAG_ENDPOINT and install missing dependencies "
            "before using /api/rag/recommend/ (or set RAG_STRICT_OPENRAG=0 for non-strict dev fallback)."
        )
    return "\n".join(lines)


def main() -> int:
    """Print readiness report and return shell status code."""
    report = format_report()
    print(report)
    return 0 if is_rag_runtime_ready() else 1


if __name__ == "__main__":
    raise SystemExit(main())
