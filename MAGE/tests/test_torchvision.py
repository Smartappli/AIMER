# Copyright (C) 2026 AIMER contributors.

"""TorchVision model registry smoke tests.

The goal is to validate that TorchVision models can be constructed across
multiple submodules (classification, detection, segmentation, etc.) without
crashing, while remaining compatible with older TorchVision APIs.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
import time
from types import ModuleType
from typing import TYPE_CHECKING
from urllib.error import URLError

import pytest

TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
TQDM_AVAILABLE = importlib.util.find_spec("tqdm") is not None

pytestmark = pytest.mark.skipif(
    not (TORCHVISION_AVAILABLE and TQDM_AVAILABLE),
    reason="Optional dependencies `torchvision` and `tqdm` are required for this test.",
)

if TORCHVISION_AVAILABLE and TQDM_AVAILABLE:
    import torchvision.models as tvm
    from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn

logger = logging.getLogger(__name__)


def _safe_list_models(module: ModuleType) -> list[str]:
    """Return available model builder names for a TorchVision module.

    Prefers the official registry (TorchVision >= 0.14 via ``tvm.list_models``).
    Falls back to introspection for older versions (less reliable, but avoids
    breaking on legacy installs).

    Returns:
        List of detected model-builder names for ``module``.

    """
    if hasattr(tvm, "list_models"):
        return list(tvm.list_models(module=module))

    names: list[str] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        if name.lower() != name:
            continue

        try:
            inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            logger.debug(
                "Skipping %s.%s: %s",
                module.__name__,
                name,
                exc,
                exc_info=True,
            )
            continue

        names.append(name)

    return sorted(set(names))


def _safe_get_model_weights_default(model_name: str) -> object | None:
    """Return ``weights_enum.DEFAULT`` for a model if available, else ``None``.

    Returns:
        The default weights enum member when available, otherwise ``None``.

    """
    if not hasattr(tvm, "get_model_weights"):
        return None

    try:
        weights_enum = tvm.get_model_weights(model_name)
        return getattr(weights_enum, "DEFAULT", None)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.debug(
            "No DEFAULT weights for %s: %s",
            model_name,
            exc,
            exc_info=True,
        )
        return None


def _safe_get_model(
    model_name: str,
    *,
    weights: object | None,
    num_classes: int,
) -> nn.Module:
    """Create a model using the best available TorchVision API.

    - If ``tvm.get_model`` exists:
      - When ``weights`` is not None, call ``get_model(name, weights=...)``.
      - Otherwise, try ``get_model(name, weights=None, num_classes=...)``,
        then fall back to without ``num_classes`` if unsupported.
    - Otherwise, fall back to calling the builder function directly on
      ``torchvision.models`` (very old TorchVision).

    Returns:
        Instantiated TorchVision model.

    Raises:
        RuntimeError: If no legacy builder exists for ``model_name``.

    """
    if hasattr(tvm, "get_model"):
        get_model = tvm.get_model

        if weights is not None:
            return get_model(model_name, weights=weights)

        try:
            return get_model(model_name, weights=None, num_classes=num_classes)
        except TypeError:
            return get_model(model_name, weights=None)

    builder = getattr(tvm, model_name, None)
    if builder is None:
        msg = f"Model builder not found for old torchvision fallback: {model_name!r}"
        raise RuntimeError(msg)

    try:
        return builder(pretrained=True)
    except (TypeError, OSError, RuntimeError, ValueError) as exc:
        logger.debug(
            "pretrained=True failed for %s, retrying pretrained=False: %s",
            model_name,
            exc,
            exc_info=True,
        )
        return builder(pretrained=False)


def _create_one(
    model_name: str,
    num_classes: int,
) -> tuple[nn.Module, str, float]:
    """Create one model, trying DEFAULT weights first, then random init.

    Returns:
        A tuple ``(model, status, elapsed_seconds)`` where ``status`` is either
        ``"weights"`` or ``"no-weights"``.

    """
    start_time = time.time()
    weights = _safe_get_model_weights_default(model_name)

    if weights is not None:
        try:
            model = _safe_get_model(
                model_name,
                weights=weights,
                num_classes=num_classes,
            )
            return model, "weights", time.time() - start_time
        except (URLError, OSError, RuntimeError, ValueError, TypeError):
            # Download/mismatch/old API quirks -> fall back to random.
            pass

    start_time = time.time()
    model = _safe_get_model(model_name, weights=None, num_classes=num_classes)
    return model, "no-weights", time.time() - start_time


def _iter_models(tv_models: dict[str, list[str]]) -> Iterator[tuple[str, str]]:
    """Yield ``(module_name, model_name)`` pairs.

    Yields:
        Pairs combining a submodule name and one model name from that submodule.

    """
    for module_name, model_list in tv_models.items():
        for model_name in model_list:
            yield module_name, model_name


def _run_creation_with_progress(
    tv_models: dict[str, list[str]],
    *,
    num_classes: int,
) -> list[tuple[str, str, str]]:
    """Run model creation checks and return a list of failures.

    Returns:
        List of model-creation failures as ``(module, model, error_repr)``.

    """
    total_models = sum(len(model_list) for model_list in tv_models.values())
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    bar_fmt = (
        "{l_bar}{bar} | {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    failures: list[tuple[str, str, str]] = []

    if is_tty:
        return _run_creation_with_tty_progress(
            tv_models=tv_models,
            total_models=total_models,
            bar_fmt=bar_fmt,
            num_classes=num_classes,
        )

    with tqdm(
        total=total_models,
        desc="torchvision get_model",
        unit="model",
        dynamic_ncols=False,
        bar_format=bar_fmt,
        mininterval=0.2,
        smoothing=0.1,
        leave=True,
        disable=False,
        file=sys.stdout,
    ) as pbar:
        for module_name, model_name in _iter_models(tv_models):
            pbar.set_postfix_str(f"{module_name} • {model_name}")
            try:
                model, _, _ = _create_one(model_name, num_classes)
                del model
            except (RuntimeError, OSError, ValueError, TypeError, URLError) as exc:
                failures.append((module_name, model_name, repr(exc)))
            pbar.update(1)

    return failures


def _run_creation_with_tty_progress(
    *,
    tv_models: dict[str, list[str]],
    total_models: int,
    bar_fmt: str,
    num_classes: int,
) -> list[tuple[str, str, str]]:
    """Run model creation checks with nested TTY progress bars.

    Returns:
        List of model-creation failures as ``(module, model, error_repr)``.

    """
    failures: list[tuple[str, str, str]] = []
    with tqdm(
        total=total_models,
        desc="All models",
        unit="model",
        position=0,
        dynamic_ncols=True,
        bar_format=bar_fmt,
        mininterval=0.2,
        smoothing=0.1,
        leave=True,
        disable=False,
        file=sys.stdout,
    ) as p_global:
        for module_name, model_list in tv_models.items():
            with tqdm(
                total=len(model_list),
                desc=f"Module: {module_name}",
                unit="model",
                position=1,
                dynamic_ncols=True,
                bar_format=bar_fmt,
                mininterval=0.2,
                smoothing=0.1,
                leave=False,
                disable=False,
                file=sys.stdout,
            ) as p_mod:
                _run_module_creation(
                    module_name=module_name,
                    tv_models=tv_models,
                    progress_bars=(p_mod, p_global),
                    num_classes=num_classes,
                    failures=failures,
                )

    return failures


def _run_module_creation(
    *,
    module_name: str,
    tv_models: dict[str, list[str]],
    progress_bars: tuple[object, object],
    num_classes: int,
    failures: list[tuple[str, str, str]],
) -> None:
    """Create all models for a module and update progress bars in-place."""
    p_mod, p_global = progress_bars
    model_list = tv_models[module_name]
    for model_name in model_list:
        p_mod.set_postfix_str(model_name)
        p_global.set_postfix_str(f"{module_name} • {model_name}")

        try:
            model, status, elapsed = _create_one(model_name, num_classes)
            del model

            p_mod.set_postfix_str(
                f"{model_name} • {status} • {elapsed:.2f}s",
            )
            postfix = f"{module_name} • {model_name} • {status} • {elapsed:.2f}s"
            p_global.set_postfix_str(postfix)
        except (
            RuntimeError,
            OSError,
            ValueError,
            TypeError,
            URLError,
        ) as exc:
            failures.append((module_name, model_name, repr(exc)))

        p_mod.update(1)
        p_global.update(1)


@pytest.fixture(scope="session")
def tv_modules() -> dict[str, ModuleType]:
    """Return TorchVision submodules to probe for model builders.

    Returns:
        Mapping from logical module labels to TorchVision submodule objects.

    """
    modules: dict[str, ModuleType] = {"classification": tvm}

    for sub in (
        "detection",
        "segmentation",
        "video",
        "optical_flow",
        "quantization",
    ):
        mod = getattr(tvm, sub, None)
        if isinstance(mod, ModuleType):
            modules[sub] = mod

    wanted = os.getenv("TV_TEST_MODULES")
    if wanted:
        keep = {m.strip() for m in wanted.split(",") if m.strip()}
        modules = {k: v for k, v in modules.items() if k in keep}

    return modules


@pytest.fixture(scope="session")
def tv_models(tv_modules: dict[str, ModuleType]) -> dict[str, list[str]]:
    """Return model-name lists per TorchVision submodule.

    Returns:
        Dictionary keyed by module label with corresponding model-builder names.

    """
    models = {name: _safe_list_models(mod) for name, mod in tv_modules.items()}

    lim = os.getenv("TV_TEST_LIMIT_PER_MODULE")
    limit = int(lim) if (lim and lim.isdigit()) else None
    if limit is not None:
        models = {k: v[:limit] for k, v in models.items()}

    return models


@pytest.fixture(scope="session")
def num_classes() -> int:
    """Return a small class count for classifier heads.

    Returns:
        Number of classes used to build classifier heads during tests.

    """
    return 10


@pytest.mark.slow
def test_torchvision_model_creation(
    tv_models: dict[str, list[str]],
    num_classes: int,
) -> None:
    """Smoke-test that TorchVision models can be instantiated without crashing."""
    failures = _run_creation_with_progress(tv_models, num_classes=num_classes)
    if failures:
        msg = "Certaines créations de modèles TorchVision ont échoué:\n" + "\n".join(
            f"- {m} / {n}: {err}" for m, n, err in failures
        )
        pytest.fail(msg)
