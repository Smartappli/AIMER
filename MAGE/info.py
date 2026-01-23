"""Runtime environment information utilities.

This module gathers Python/PyTorch and accelerator backend details (CUDA/ROCm,
Apple MPS, Intel XPU) for debugging and reproducibility.

It is import-safe: nothing is written to stdout on import. Run as a script to
print a report, e.g. ``python -m MAGE.info``.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def _safe(label: str, fn: Callable[[], T], default: T | str = "N/A") -> T | str:
    """Call a function and return its result, or a safe fallback on failure.

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
    except Exception as exc:  # noqa: BLE001
        return f"{default} ({label}: {type(exc).__name__}: {exc})"


def build_info_report() -> str:
    """Build a multi-line report describing the current runtime environment.

    Returns:
        A formatted string with Python, PyTorch and backend capability details.

    """
    lines: list[str] = []

    lines.append(
        f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})"
    )
    lines.append(f"PyTorch version: {torch.__version__}")
    lines.append("")
    lines.append("--- Build info (CUDA / HIP (ROCm) / etc.)")
    lines.append(f"torch.version.cuda: {torch.version.cuda}")
    lines.append(f"torch.version.hip:  {getattr(torch.version, 'hip', None)}")
    lines.append(f"torch.version.git:  {getattr(torch.version, 'git_version', None)}")

    # --- CUDA / ROCm (AMD uses torch.cuda API too)
    lines.append("")
    lines.append("=== CUDA / ROCm ===")
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
                f" - [{i}] {name} | capability={cap_major}.{cap_minor} | VRAM={mem_gb:.2f} GB"
            )

    # --- Apple MPS (Mac)
    lines.append("")
    lines.append("=== MPS (Apple Silicon) ===")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        lines.append("MPS backend: not present in this build")
    else:
        lines.append(
            f"MPS built:     {_safe('mps_built', mps_backend.is_built, default=False)}"
        )
        lines.append(
            f"MPS available: {_safe('mps_avail', mps_backend.is_available, default=False)}"
        )

    # --- Intel XPU (oneAPI / Intel GPU)
    lines.append("")
    lines.append("=== XPU (Intel) ===")
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
                    "xpu_name", lambda i=i: xpu.get_device_name(i), default="Unknown"
                )
                lines.append(f" - [{i}] {name}")

    # --- CPU basics
    lines.append("")
    lines.append("=== CPU ===")
    lines.append(
        f"Num threads: {_safe('threads', torch.get_num_threads, default='N/A')}"
    )
    lines.append(f"Default dtype: {torch.get_default_dtype()}")

    return "\n".join(lines)


def main() -> None:
    """Script entry point: write the info report to stdout."""
    sys.stdout.write(build_info_report() + "\n")


if __name__ == "__main__":
    main()
