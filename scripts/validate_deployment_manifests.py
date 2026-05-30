#!/usr/bin/env python3
# Copyright (c) 2026 AIMER contributors.
"""Validate production deployment manifests without external dependencies."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K8S_DIR = ROOT / "infra" / "prod" / "kubernetes"
ZERO_DIGEST = "0" * 64

EXPECTED_RESOURCES = {
    "namespace.yaml",
    "serviceaccounts.yaml",
    "configmap.yaml",
    "services.yaml",
    "deployments.yaml",
    "ingress.yaml",
    "network-policies.yaml",
    "pod-disruption-budgets.yaml",
    "hpa.yaml",
}

EXPECTED_IMAGES = {
    "aimer-web": "docker.io/smartappli/aimer",
    "aimer-rag": "docker.io/smartappli/aimer-rag",
    "mage": "docker.io/smartappli/mage-api",
    "farm": "docker.io/smartappli/farm",
}

EXPECTED_SECRET_REFS = {
    "aimer-web": "aimer-web-secrets",
    "aimer-rag": "aimer-rag-secrets",
    "mage": "mage-secrets",
    "farm": "farm-secrets",
}

EXPECTED_SERVICE_PORTS = {
    "aimer-web": "8000",
    "aimer-rag": "8000",
    "mage": "10000",
    "farm": "8000",
}

SENSITIVE_MESSAGE_PATTERN = re.compile(
    r"(?i)(password|secret|token|credential|api[_-]?key)([^,\n]*)",
)


@dataclass(slots=True)
class CheckResult:
    """Deployment validation result."""

    errors: list[str]
    warnings: list[str]

    def error(self, message: str) -> None:
        """Record a blocking validation error."""
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Record a non-blocking warning."""
        self.warnings.append(message)


def _read(path: Path) -> str:
    """Read text from a repository file."""
    return path.read_text(encoding="utf-8")


def _yaml_docs(path: Path) -> list[str]:
    """Split a simple multi-document YAML file."""
    text = _read(path)
    return [doc.strip() for doc in re.split(r"(?m)^---\s*$", text) if doc.strip()]


def _field(doc: str, name: str) -> str | None:
    """Return the first scalar field value with a given YAML key."""
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*\"?([^\"\n]+)\"?\s*$", doc)
    return match.group(1).strip() if match else None


def _kind(doc: str) -> str | None:
    """Return a Kubernetes document kind."""
    return _field(doc, "kind")


def _metadata_name(doc: str) -> str | None:
    """Return a Kubernetes document metadata.name value."""
    pattern = r"(?ms)^metadata:\s*\n(?:^[ ]{2,}.*\n)*?^[ ]{2}name:\s*([^\n]+)"
    match = re.search(pattern, doc)
    return match.group(1).strip().strip('"') if match else None


def _safe_console_message(message: str) -> str:
    """Return a redacted single-line message for logs and CI summaries."""
    redacted = SENSITIVE_MESSAGE_PATTERN.sub(r"\1=[REDACTED]", message)
    return redacted.replace("\r", "\\r").replace("\n", "\\n")


def _ingress_hosts(doc: str) -> set[str]:
    """Return hostnames declared in an Ingress manifest."""
    hosts: set[str] = set()
    for match in re.finditer(r"(?m)^\s*host:\s*\"?([^\"\n]+)\"?\s*$", doc):
        hosts.add(match.group(1).strip().lower().rstrip("."))
    return hosts


def _documents_by_kind_name(path: Path) -> dict[tuple[str, str], str]:
    """Return Kubernetes docs keyed by kind/name."""
    docs: dict[tuple[str, str], str] = {}
    for doc in _yaml_docs(path):
        kind = _kind(doc)
        name = _metadata_name(doc)
        if kind and name:
            docs[(kind, name)] = doc
    return docs


def _kustomization_resources() -> set[str]:
    """Return resource filenames listed in kustomization.yaml."""
    lines = _read(K8S_DIR / "kustomization.yaml").splitlines()
    resources: set[str] = set()
    in_resources = False
    for line in lines:
        if line == "resources:":
            in_resources = True
            continue
        if in_resources and line and not line.startswith("  "):
            break
        if in_resources and line.strip().startswith("-"):
            resources.add(line.strip().removeprefix("-").strip())
    return resources


def _kustomization_images() -> dict[str, str]:
    """Return image digests from kustomization.yaml keyed by image name."""
    lines = _read(K8S_DIR / "kustomization.yaml").splitlines()
    images: dict[str, str] = {}
    current_name: str | None = None
    current_digest: str | None = None
    in_images = False

    def flush_current() -> None:
        if current_name and current_digest:
            images[current_name] = current_digest

    for line in lines:
        if line == "images:":
            in_images = True
            continue
        if in_images and line and not line.startswith("  "):
            break
        if not in_images:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            flush_current()
            current_name = None
            current_digest = None
            key, _, value = stripped.removeprefix("- ").partition(":")
        elif stripped:
            key, _, value = stripped.partition(":")
        else:
            continue

        if key == "name":
            current_name = value.strip()
        elif key == "digest":
            current_digest = value.strip()
    flush_current()
    return images


def _validate_required_files(result: CheckResult) -> None:
    """Validate required Kubernetes baseline files."""
    if not K8S_DIR.is_dir():
        result.error(f"Missing Kubernetes directory: {K8S_DIR}")
        return
    for filename in sorted(EXPECTED_RESOURCES | {"README.md", "kustomization.yaml"}):
        path = K8S_DIR / filename
        if not path.is_file():
            result.error(f"Missing Kubernetes baseline file: {path}")
        elif path.stat().st_size == 0:
            result.error(f"Empty Kubernetes baseline file: {path}")


def _validate_kustomization(
    result: CheckResult,
    *,
    require_real_digests: bool,
) -> None:
    """Validate kustomization resources and image pinning."""
    resources = _kustomization_resources()
    missing = EXPECTED_RESOURCES - resources
    extra = resources - EXPECTED_RESOURCES
    for filename in sorted(missing):
        result.error(f"kustomization.yaml does not include {filename}.")
    for filename in sorted(extra):
        result.warn(f"kustomization.yaml includes unexpected resource {filename}.")

    images = _kustomization_images()
    for image in EXPECTED_IMAGES.values():
        digest = images.get(image)
        if digest is None:
            result.error(f"kustomization.yaml missing image digest for {image}.")
            continue
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            result.error(f"{image} digest is not a sha256 pin: {digest}")
        elif digest.endswith(ZERO_DIGEST):
            message = f"{image} uses the placeholder zero digest."
            if require_real_digests:
                result.error(message)
            else:
                result.warn(message)


def _validate_deployments(result: CheckResult) -> None:
    """Validate deployment hardening contracts."""
    docs = _documents_by_kind_name(K8S_DIR / "deployments.yaml")
    for service, image in EXPECTED_IMAGES.items():
        doc = docs.get(("Deployment", service))
        if doc is None:
            result.error(f"deployments.yaml missing Deployment/{service}.")
            continue
        for required in (
            "runAsNonRoot: true",
            "type: RuntimeDefault",
            "allowPrivilegeEscalation: false",
            "readOnlyRootFilesystem: true",
            'drop: ["ALL"]',
            "livenessProbe:",
            "readinessProbe:",
            "resources:",
            f"serviceAccountName: {service}",
            f"name: {EXPECTED_SECRET_REFS[service]}",
            f"image: {image}",
        ):
            if required not in doc:
                result.error(f"Deployment/{service} missing required text: {required}")
        if service in {"aimer-web", "farm"}:
            if "RUN_DJANGO_MIGRATIONS" not in doc or 'value: "0"' not in doc:
                result.error(
                    f"Deployment/{service} must disable serving-pod migrations."
                )
        if service == "aimer-rag" and "RAG_VERIFY_ON_START" not in doc:
            result.error("Deployment/aimer-rag must verify RAG runtime on start.")


def _validate_services(result: CheckResult) -> None:
    """Validate private service exposure contracts."""
    docs = _documents_by_kind_name(K8S_DIR / "services.yaml")
    for service, port in EXPECTED_SERVICE_PORTS.items():
        doc = docs.get(("Service", service))
        if doc is None:
            result.error(f"services.yaml missing Service/{service}.")
            continue
        for required in (
            "type: ClusterIP",
            f"app.kubernetes.io/name: {service}",
            f"port: {port}",
        ):
            if required not in doc:
                result.error(f"Service/{service} missing required text: {required}")


def _validate_ingress(result: CheckResult, *, require_real_domains: bool) -> None:
    """Validate public ingress scope."""
    docs = _documents_by_kind_name(K8S_DIR / "ingress.yaml")
    doc = docs.get(("Ingress", "aimer-web"))
    if doc is None:
        result.error("ingress.yaml must expose only Ingress/aimer-web.")
        return
    if "name: aimer-web" not in doc:
        result.error("Ingress must route to the aimer-web service.")
    for private_service in ("aimer-rag", "mage", "farm"):
        if private_service in doc:
            result.error(
                f"Ingress must not route to private service {private_service}."
            )
    hosts = _ingress_hosts(doc)
    if any(host == "example.org" or host.endswith(".example.org") for host in hosts):
        message = "Ingress still uses example.org placeholder domains."
        if require_real_domains:
            result.error(message)
        else:
            result.warn(message)


def _validate_network_policies(result: CheckResult, *, strict_network: bool) -> None:
    """Validate network policy boundary contracts."""
    docs = _documents_by_kind_name(K8S_DIR / "network-policies.yaml")
    for policy in (
        "default-deny",
        "allow-dns-egress",
        "allow-public-ingress-to-aimer-web",
        "allow-web-to-rag",
        "allow-private-service-ingress",
        "allow-private-egress",
    ):
        if ("NetworkPolicy", policy) not in docs:
            result.error(f"network-policies.yaml missing NetworkPolicy/{policy}.")
    default_deny = docs.get(("NetworkPolicy", "default-deny"), "")
    if "policyTypes:" not in default_deny or "- Ingress" not in default_deny:
        result.error("NetworkPolicy/default-deny must deny ingress.")
    if "- Egress" not in default_deny:
        result.error("NetworkPolicy/default-deny must deny egress.")

    egress = docs.get(("NetworkPolicy", "allow-private-egress"), "")
    broad_cidrs = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    present_broad_cidrs = [cidr for cidr in broad_cidrs if cidr in egress]
    if present_broad_cidrs:
        message = "allow-private-egress permits broad private CIDRs: "
        message += ", ".join(present_broad_cidrs)
        if strict_network:
            result.error(message)
        else:
            result.warn(message)


def _validate_no_dev_stack_references(result: CheckResult) -> None:
    """Ensure production manifests do not depend on the dev stack."""
    for path in K8S_DIR.glob("*.yaml"):
        text = _read(path)
        if "infra/dev-stack" in text or "dev-stack" in text:
            result.error(f"{path.relative_to(ROOT)} references the development stack.")


def _write_summary(path: Path, result: CheckResult) -> None:
    """Write a GitHub job summary compatible Markdown file."""
    lines = [
        "## Deployment Manifest Tests",
        "",
        f"- Errors: **{len(result.errors)}**",
        f"- Warnings: **{len(result.warnings)}**",
        "",
    ]
    if result.errors:
        lines.extend(["### Errors", ""])
        lines.extend(f"- {_safe_console_message(error)}" for error in result.errors)
        lines.append("")
    if result.warnings:
        lines.extend(["### Warnings", ""])
        lines.extend(
            f"- {_safe_console_message(warning)}" for warning in result.warnings
        )
        lines.append("")
    if not result.errors:
        lines.append("Deployment manifest validation passed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(args: argparse.Namespace) -> CheckResult:
    """Run deployment manifest validation."""
    result = CheckResult(errors=[], warnings=[])
    _validate_required_files(result)
    if result.errors:
        return result
    _validate_kustomization(result, require_real_digests=args.require_real_digests)
    _validate_deployments(result)
    _validate_services(result)
    _validate_ingress(result, require_real_domains=args.require_real_domains)
    _validate_network_policies(result, strict_network=args.strict_network)
    _validate_no_dev_stack_references(result)
    return result


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-real-digests",
        action="store_true",
        help="Reject placeholder image digests.",
    )
    parser.add_argument(
        "--require-real-domains",
        action="store_true",
        help="Reject example.org placeholder ingress domains.",
    )
    parser.add_argument(
        "--strict-network",
        action="store_true",
        help="Reject broad private CIDR egress rules.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional Markdown summary output path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run deployment tests."""
    args = _parse_args()
    result = validate(args)
    if args.summary_file is not None:
        _write_summary(args.summary_file, result)
    for warning in result.warnings:
        print(f"WARNING: {_safe_console_message(warning)}")
    if result.errors:
        print("Deployment manifest validation failed:", file=sys.stderr)
        for error in result.errors:
            print(f"- {_safe_console_message(error)}", file=sys.stderr)
        return 1
    print("Deployment manifest validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
