# Copyright (C) 2026 AIMER contributors.

"""
Runtime environment information utilities.

This module gathers Python/PyTorch and accelerator backend details (CUDA/ROCm,
Apple MPS, Intel XPU) for debugging and reproducibility.

It is import-safe: nothing is written to stdout on import. Run as a script to
print a report, e.g. ``python info.py``.
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def _safe[T](label: str, fn: Callable[[], T], default: T | str = "N/A") -> T | str:
    """
    Call a function and return its result, or a safe fallback on failure.

    Args:
        label: Human-friendly label included in the fallback message.
        fn: Callable to execute.
        default: Default value (or default label) used when an exception occurs.

    Returns:
        The callable result if successful, otherwise a string describing the
        failure along with the provided default.

    """
    try:
        return fn()
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        return f"{default} ({label}: {type(exc).__name__}: {exc})"


def _base_info_lines() -> list[str]:
    """Return static Python and PyTorch runtime details."""
    return [
        f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})",
        f"PyTorch version: {torch.__version__}",
        "",
        "--- Build info (CUDA / HIP (ROCm) / etc.)",
        f"torch.version.cuda: {torch.version.cuda}",
        f"torch.version.hip:  {getattr(torch.version, 'hip', None)}",
        f"torch.version.git:  {getattr(torch.version, 'git_version', None)}",
        "",
        "=== CUDA / ROCm ===",
    ]


def _append_cuda_info(lines: list[str]) -> None:
    """Append CUDA and ROCm device details to a report."""
    cuda_available = torch.cuda.is_available()
    lines.append(f"CUDA available: {cuda_available}")

    cuda_count = _safe("cuda_count", torch.cuda.device_count, default=0)
    lines.append(f"CUDA device count: {cuda_count}")

    if cuda_available is True and isinstance(cuda_count, int) and cuda_count > 0:
        cur = _safe("cuda_current", torch.cuda.current_device, default="N/A")
        lines.append(f"CUDA current device: {cur}")

        for i in range(cuda_count):
            name = _safe(
                "cuda_name",
                lambda i=i: torch.cuda.get_device_name(i),
                default="Unknown",
            )
            props = _safe(
                "cuda_props",
                lambda i=i: torch.cuda.get_device_properties(i),
                default=None,
            )

            if props is None or isinstance(props, str):
                lines.append(f" - [{i}] {name} | props={props}")
                continue

            cap_major = getattr(props, "major", None)
            cap_minor = getattr(props, "minor", None)
            mem_bytes = getattr(props, "total_memory", 0)
            mem_gb = float(mem_bytes) / (1024**3) if mem_bytes else 0.0
            lines.append(
                (
                    f" - [{i}] {name} | capability={cap_major}.{cap_minor} "
                    f"| VRAM={mem_gb:.2f} GB"
                ),
            )


def _append_mps_info(lines: list[str]) -> None:
    """Append Apple MPS backend details to a report."""
    lines.extend(("", "=== MPS (Apple Silicon) ==="))
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        lines.append("MPS backend: not present in this build")
    else:
        lines.extend(
            (
                (
                    "MPS built:     "
                    f"{_safe('mps_built', mps_backend.is_built, default=False)}"
                ),
                (
                    "MPS available: "
                    f"{_safe('mps_avail', mps_backend.is_available, default=False)}"
                ),
            ),
        )


def _append_xpu_info(lines: list[str]) -> None:
    """Append Intel XPU backend details to a report."""
    lines.extend(("", "=== XPU (Intel) ==="))
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        lines.append("XPU backend: not present in this build (no torch.xpu)")
    else:
        xpu_available = _safe("xpu_avail", xpu.is_available, default=False)
        lines.append(f"XPU available: {xpu_available}")

        xpu_count = _safe("xpu_count", xpu.device_count, default=0)
        lines.append(f"XPU device count: {xpu_count}")

        if xpu_available is True and isinstance(xpu_count, int) and xpu_count > 0:
            cur = _safe("xpu_current", xpu.current_device, default="N/A")
            lines.append(f"XPU current device: {cur}")

            for i in range(xpu_count):
                name = _safe(
                    "xpu_name",
                    lambda i=i: xpu.get_device_name(i),
                    default="Unknown",
                )
                lines.append(f" - [{i}] {name}")


def _append_cpu_info(lines: list[str]) -> None:
    """Append CPU details to a report."""
    lines.extend(
        (
            "",
            "=== CPU ===",
            f"Num threads: {_safe('threads', torch.get_num_threads, default='N/A')}",
            f"Default dtype: {torch.get_default_dtype()}",
        ),
    )


def build_info_report() -> str:
    """
    Build a multi-line report describing the current runtime environment.

    Returns:
        A formatted string with Python, PyTorch and backend capability details.

    """
    lines = _base_info_lines()
    _append_cuda_info(lines)
    _append_mps_info(lines)
    _append_xpu_info(lines)
    _append_cpu_info(lines)
    return "\n".join(lines)


def main() -> None:
    """Script entry point: write the info report to stdout."""
    sys.stdout.write(build_info_report() + "\n")


if __name__ == "__main__":
    main()
