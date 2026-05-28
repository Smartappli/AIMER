#!/usr/bin/env python3
# Copyright (c) 2026 AIMER contributors.
"""Validate production evidence gates for regulated releases."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TODAY = datetime.now(UTC).date()

OPERATIONS_DOCS = [
    "README.md",
    "PRODUCTION_READINESS.md",
    "PRODUCTION_ARCHITECTURE.md",
    "RELEASE_EVIDENCE_PACK.md",
    "CONTROL_MATRIX_NIS2_DORA.md",
    "RISK_REGISTER.md",
    "ASSET_AND_SUPPLIER_REGISTER.md",
    "INCIDENT_RESPONSE.md",
    "RESILIENCE_RUNBOOK.md",
    "SUPPLY_CHAIN.md",
    "VULNERABILITY_EXCEPTIONS.md",
    "DATA_PROTECTION_RETENTION.md",
]

PRODUCTION_ENV_VARS = {
    "ENVIRONMENT": "production",
    "DJANGO_ENVIRONMENT": "production",
    "AIMER_RAG_ENVIRONMENT": "production",
    "MAGE_ENVIRONMENT": "production",
    "FARM_FEDERATED_ENVIRONMENT": "production",
    "DEBUG": "false",
    "DJANGO_DEBUG": "false",
    "RAG_STRICT_OPENRAG": "true",
    "RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS": "false",
    "RAG_ALLOW_EXTERNAL_PLUGINS": "false",
}

REQUIRED_SECRET_PLACEHOLDERS = {
    "SECRET_KEY",
    "DJANGO_SECRET_KEY",
    "RAG_SERVICE_API_KEY",
    "AIMER_RAG_API_KEY",
    "MAGE_API_KEY",
    "OPENRAG_API_KEY",
    "QDRANT_API_KEY",
}

REQUIRED_RISK_COLUMNS = {
    "ID",
    "Risk",
    "Scenario",
    "Impacted services",
    "Current controls",
    "Residual level",
    "Treatment",
    "Owner",
    "Review by",
}

REQUIRED_SUPPLIER_COLUMNS = {
    "Supplier/service",
    "Provides",
    "Critical or important function",
    "Data handled",
    "Contract owner",
    "Location/jurisdiction",
    "Exit plan",
    "Review by",
}

BUILDER_WORKFLOWS = [
    ".github/workflows/aimer_docker_builder.yml",
    ".github/workflows/mage_docker_builder.yml",
    ".github/workflows/farm_docker_builder.yml",
]

DOCKERFILES = [
    "services/aimer-web/Dockerfile",
    "services/aimer-web/Dockerfile.rag",
    "services/MAGE/Dockerfile",
    "services/FARM/Dockerfile",
]

KUBERNETES_FILES = [
    "README.md",
    "kustomization.yaml",
    "namespace.yaml",
    "serviceaccounts.yaml",
    "configmap.yaml",
    "services.yaml",
    "deployments.yaml",
    "ingress.yaml",
    "network-policies.yaml",
    "pod-disruption-budgets.yaml",
    "hpa.yaml",
]


@dataclass(frozen=True)
class Table:
    """Minimal Markdown table representation."""

    header: list[str]
    rows: list[list[str]]


def _read(path: str | Path) -> str:
    """Read a repository file."""
    return (ROOT / path).read_text(encoding="utf-8")


def _check(condition: bool, message: str, errors: list[str]) -> None:
    """Append an error when a condition is false."""
    if not condition:
        errors.append(message)


def _parse_env(path: str) -> dict[str, str]:
    """Parse simple KEY=value lines from an env example file."""
    values: dict[str, str] = {}
    for line in _read(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def _markdown_tables(text: str) -> list[Table]:
    """Extract simple pipe-delimited Markdown tables."""
    lines = text.splitlines()
    tables: list[Table] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("|"):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("|"):
            index += 1
            continue

        header = _split_table_row(lines[index])
        separator = _split_table_row(lines[index + 1])
        if not header or not all(set(cell) <= {"-", ":"} for cell in separator):
            index += 1
            continue

        index += 2
        rows: list[list[str]] = []
        while index < len(lines) and lines[index].startswith("|"):
            rows.append(_split_table_row(lines[index]))
            index += 1
        tables.append(Table(header=header, rows=rows))
    return tables


def _split_table_row(line: str) -> list[str]:
    """Split a Markdown table row into cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_with_header(text: str, required_columns: set[str]) -> Table | None:
    """Return the first table containing the required columns."""
    for table in _markdown_tables(text):
        if required_columns.issubset(set(table.header)):
            return table
    return None


def _vulnerability_ids_from_workflow() -> set[str]:
    """Return vulnerability IDs ignored by the global security workflow."""
    workflow = _read(".github/workflows/global_security.yml")
    pattern = r"--ignore-vuln\s+([A-Z]+-\d{4}-\d+|PYSEC-\d{4}-\d+)"
    return set(re.findall(pattern, workflow))


def _vulnerability_exception_rows() -> dict[str, dict[str, str]]:
    """Return vulnerability exception rows keyed by ID."""
    text = _read("docs/operations/VULNERABILITY_EXCEPTIONS.md")
    table = _table_with_header(text, {"ID", "Scope", "Reason", "Owner", "Review by"})
    if table is None:
        return {}
    rows: dict[str, dict[str, str]] = {}
    for row in table.rows:
        if len(row) < len(table.header):
            continue
        item = dict(zip(table.header, row, strict=False))
        rows[item["ID"]] = item
    return rows


def _validate_required_files(errors: list[str]) -> None:
    """Validate the operations evidence file set."""
    for filename in OPERATIONS_DOCS:
        path = ROOT / "docs" / "operations" / filename
        _check(path.is_file(), f"Missing operations document: {path}", errors)
        if path.is_file():
            _check(
                path.stat().st_size > 0,
                f"Empty operations document: {path}",
                errors,
            )

    _check(
        (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").is_file(),
        "Missing .github/PULL_REQUEST_TEMPLATE.md.",
        errors,
    )
    _check(
        (ROOT / "infra" / "prod" / ".env.example").is_file(),
        "Missing infra/prod/.env.example.",
        errors,
    )


def _validate_markdown_links(errors: list[str]) -> None:
    """Validate local Markdown links in production evidence files."""
    files = [
        ROOT / "README.md",
        ROOT / "infra" / "prod" / "README.md",
        *(ROOT / "docs" / "operations").glob("*.md"),
    ]
    pattern = re.compile(r"\[[^\]]+\]\(([^)#]+\.md)(?:#[^)]*)?\)")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            target = match.group(1)
            if re.match(r"^[a-z]+://", target):
                continue
            resolved = (path.parent / target).resolve()
            _check(
                resolved.is_file(),
                f"Broken Markdown link in {path.relative_to(ROOT)}: {target}",
                errors,
            )


def _validate_production_env(errors: list[str]) -> None:
    """Validate the production environment inventory."""
    values = _parse_env("infra/prod/.env.example")
    for key, expected in PRODUCTION_ENV_VARS.items():
        actual = values.get(key)
        _check(
            actual == expected,
            f"infra/prod/.env.example must set {key}={expected}, found {actual!r}.",
            errors,
        )

    for key in REQUIRED_SECRET_PLACEHOLDERS:
        actual = values.get(key, "")
        _check(
            actual.startswith("<vault:"),
            f"{key} must use a vault placeholder.",
            errors,
        )

    for url_key in (
        "BASE_URL",
        "DJANGO_BASE_URL",
        "OPENRAG_ENDPOINT",
        "QDRANT_URL",
    ):
        actual = values.get(url_key, "")
        _check(
            actual.startswith("https://"),
            f"{url_key} must use HTTPS in production.",
            errors,
        )


def _validate_vulnerability_exceptions(errors: list[str]) -> None:
    """Validate ignored vulnerabilities against accepted-risk records."""
    ignored = _vulnerability_ids_from_workflow()
    rows = _vulnerability_exception_rows()
    recorded = set(rows)

    _check(
        bool(ignored),
        "No ignored vulnerability IDs found in global_security.yml.",
        errors,
    )
    for vuln_id in sorted(ignored - recorded):
        errors.append(f"{vuln_id} is ignored in CI but missing from exceptions.")
    for vuln_id in sorted(recorded - ignored):
        errors.append(f"{vuln_id} is recorded as an exception but not ignored in CI.")

    for vuln_id, row in sorted(rows.items()):
        owner = row.get("Owner", "").strip()
        reason = row.get("Reason", "").strip()
        review_raw = row.get("Review by", "").strip()
        _check(owner not in {"", "TBD"}, f"{vuln_id} must have an owner.", errors)
        _check(reason not in {"", "TBD"}, f"{vuln_id} must have a reason.", errors)
        try:
            review_by = date.fromisoformat(review_raw)
        except ValueError:
            errors.append(f"{vuln_id} has invalid review date: {review_raw!r}.")
            continue
        _check(
            review_by >= TODAY,
            f"{vuln_id} exception expired on {review_by}.",
            errors,
        )


def _validate_register_tables(errors: list[str]) -> None:
    """Validate risk and supplier register structures."""
    risk_text = _read("docs/operations/RISK_REGISTER.md")
    risk_table = _table_with_header(risk_text, REQUIRED_RISK_COLUMNS)
    _check(
        risk_table is not None,
        "Risk register table is missing required columns.",
        errors,
    )
    if risk_table is not None:
        _check(
            len(risk_table.rows) >= 5,
            "Risk register must contain baseline risks.",
            errors,
        )
        ids = [row[0] for row in risk_table.rows if row]
        _check(len(ids) == len(set(ids)), "Risk register IDs must be unique.", errors)

    supplier_text = _read("docs/operations/ASSET_AND_SUPPLIER_REGISTER.md")
    supplier_table = _table_with_header(supplier_text, REQUIRED_SUPPLIER_COLUMNS)
    _check(
        supplier_table is not None,
        "ICT supplier register table is missing required columns.",
        errors,
    )
    if supplier_table is not None:
        _check(
            len(supplier_table.rows) >= 5,
            "ICT supplier register must contain baseline supplier categories.",
            errors,
        )


def _validate_supply_chain_workflows(errors: list[str]) -> None:
    """Validate supply-chain workflow guardrails."""
    supply_chain = _read(".github/workflows/supply_chain_security.yml")
    for required in (
        "scan-type: fs",
        "scan-type: config",
        "format: cyclonedx",
        "scan-type: image",
        "github/codeql-action/upload-sarif",
    ):
        _check(
            required in supply_chain,
            f"supply_chain_security.yml missing required guardrail: {required}",
            errors,
        )

    for workflow_path in BUILDER_WORKFLOWS:
        workflow = _read(workflow_path)
        for required in (
            "provenance: true",
            "sbom: true",
            "cosign sign",
            "image-digest",
        ):
            _check(
                required in workflow,
                f"{workflow_path} missing required release evidence step: {required}",
                errors,
            )


def _validate_runtime_architecture(errors: list[str]) -> None:
    """Validate production runtime architecture guardrails."""
    for dockerfile in DOCKERFILES:
        text = _read(dockerfile)
        _check(
            "HEALTHCHECK" in text,
            f"{dockerfile} must define a container HEALTHCHECK.",
            errors,
        )
        _check(
            "USER appuser" in text,
            f"{dockerfile} must run as the non-root appuser.",
            errors,
        )

    k8s_root = ROOT / "infra" / "prod" / "kubernetes"
    for filename in KUBERNETES_FILES:
        path = k8s_root / filename
        _check(path.is_file(), f"Missing Kubernetes baseline file: {path}", errors)
        if path.is_file():
            _check(path.stat().st_size > 0, f"Empty Kubernetes file: {path}", errors)

    deployments = _read("infra/prod/kubernetes/deployments.yaml")
    for required in (
        "livenessProbe:",
        "readinessProbe:",
        "resources:",
        "name: aimer-web-secrets",
        "name: aimer-rag-secrets",
        "name: mage-secrets",
        "name: farm-secrets",
        "runAsNonRoot: true",
        "allowPrivilegeEscalation: false",
        "readOnlyRootFilesystem: true",
        'drop: ["ALL"]',
    ):
        _check(
            required in deployments,
            f"deployments.yaml missing workload hardening: {required}",
            errors,
        )

    network_policies = _read("infra/prod/kubernetes/network-policies.yaml")
    for required in (
        "name: default-deny",
        "policyTypes:",
        "network.aimer.io/ingress",
        "app.kubernetes.io/name: aimer-rag",
    ):
        _check(
            required in network_policies,
            f"network-policies.yaml missing required boundary: {required}",
            errors,
        )

    kustomization = _read("infra/prod/kubernetes/kustomization.yaml")
    for image in (
        "smartappli/aimer",
        "smartappli/aimer-rag",
        "smartappli/mage-api",
        "smartappli/farm",
    ):
        _check(image in kustomization, f"kustomization.yaml missing {image}.", errors)
    _check(
        "digest: sha256:" in kustomization,
        "kustomization.yaml must pin images by digest.",
        errors,
    )


def _validate_pr_template(errors: list[str]) -> None:
    """Validate the pull request template includes production review prompts."""
    text = _read(".github/PULL_REQUEST_TEMPLATE.md")
    for required in (
        "Production and Compliance Impact",
        "Risk register reviewed",
        "Asset/supplier register reviewed",
        "Data protection and retention impact reviewed",
        "Rollback or recovery notes updated",
    ):
        _check(required in text, f"PR template missing prompt: {required}", errors)


def main() -> int:
    """Run all production evidence checks."""
    errors: list[str] = []

    _validate_required_files(errors)
    _validate_markdown_links(errors)
    _validate_production_env(errors)
    _validate_vulnerability_exceptions(errors)
    _validate_register_tables(errors)
    _validate_supply_chain_workflows(errors)
    _validate_runtime_architecture(errors)
    _validate_pr_template(errors)

    if errors:
        print("Production evidence validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Production evidence validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
