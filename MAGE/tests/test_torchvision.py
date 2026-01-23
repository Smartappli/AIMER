"""TorchVision model registry smoke tests.

The goal is to validate that TorchVision models can be constructed across
multiple submodules (classification, detection, segmentation, etc.) without
crashing, while remaining compatible with older TorchVision APIs.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import time
from types import ModuleType
from typing import TYPE_CHECKING
from urllib.error import URLError

import pytest
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
                "Skipping %s.%s: %s", module.__name__, name, exc, exc_info=True,
            )
            continue

        names.append(name)

    return sorted(set(names))


def _safe_get_model_weights_default(model_name: str) -> object | None:
    """Return ``weights_enum.DEFAULT`` for a model if available, else ``None``."""
    if not hasattr(tvm, "get_model_weights"):
        return None

    try:
        weights_enum = tvm.get_model_weights(model_name)
        return getattr(weights_enum, "DEFAULT", None)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug("No DEFAULT weights for %s: %s", model_name, exc, exc_info=True)
        return None


def _safe_get_model(
    model_name: str, *, weights: object | None, num_classes: int,
) -> nn.Module:
    """Create a model using the best available TorchVision API.

    - If ``tvm.get_model`` exists:
      - When ``weights`` is not None, call ``get_model(name, weights=...)``.
      - Otherwise, try ``get_model(name, weights=None, num_classes=...)``,
        then fall back to without ``num_classes`` if unsupported.
    - Otherwise, fall back to calling the builder function directly on
      ``torchvision.models`` (very old TorchVision).
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


def _create_one(model_name: str, num_classes: int) -> tuple[nn.Module, str, float]:
    """Create one model, trying DEFAULT weights first, then random init."""
    start_time = time.time()
    weights = _safe_get_model_weights_default(model_name)

    if weights is not None:
        try:
            model = _safe_get_model(
                model_name, weights=weights, num_classes=num_classes,
            )
            return model, "weights", time.time() - start_time
        except (URLError, OSError, RuntimeError, ValueError, TypeError):
            # Download/mismatch/old API quirks -> fall back to random.
            pass

    start_time = time.time()
    model = _safe_get_model(model_name, weights=None, num_classes=num_classes)
    return model, "no-weights", time.time() - start_time


def _iter_models(tv_models: dict[str, list[str]]) -> Iterator[tuple[str, str]]:
    """Yield (module_name, model_name) pairs."""
    for module_name, model_list in tv_models.items():
        for model_name in model_list:
            yield module_name, model_name


def _run_creation_with_progress(
    tv_models: dict[str, list[str]], *, num_classes: int,
) -> list[tuple[str, str, str]]:
    """Run model creation checks and return a list of failures."""
    total_models = sum(len(model_list) for model_list in tv_models.values())
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    bar_fmt = (
        "{l_bar}{bar} | {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    failures: list[tuple[str, str, str]] = []

    if is_tty:
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
                    for model_name in model_list:
                        p_mod.set_postfix_str(model_name)
                        p_global.set_postfix_str(f"{module_name} • {model_name}")

                        try:
                            model, status, elapsed = _create_one(
                                model_name, num_classes,
                            )
                            del model

                            p_mod.set_postfix_str(
                                f"{model_name} • {status} • {elapsed:.2f}s",
                            )
                            postfix = (
                                f"{module_name} • {model_name} • "
                                f"{status} • {elapsed:.2f}s"
                            )
                            p_global.set_postfix_str(postfix)
                        except Exception as exc:  # noqa: BLE001
                            failures.append((module_name, model_name, repr(exc)))

                        p_mod.update(1)
                        p_global.update(1)

        return failures

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
            except Exception as exc:  # noqa: BLE001
                failures.append((module_name, model_name, repr(exc)))
            pbar.update(1)

    return failures


@pytest.fixture(scope="session")
def tv_modules() -> dict[str, ModuleType]:
    """Return TorchVision submodules to probe for model builders."""
    modules: dict[str, ModuleType] = {"classification": tvm}

    for sub in ("detection", "segmentation", "video", "optical_flow", "quantization"):
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
    """Return model name lists per TorchVision submodule."""
    models = {name: _safe_list_models(mod) for name, mod in tv_modules.items()}

    lim = os.getenv("TV_TEST_LIMIT_PER_MODULE")
    limit = int(lim) if (lim and lim.isdigit()) else None
    if limit is not None:
        models = {k: v[:limit] for k, v in models.items()}

    return models


@pytest.fixture(scope="session")
def num_classes() -> int:
    """Return a small class count for classifier heads."""
    return 10


@pytest.mark.slow
def test_torchvision_model_creation(
    tv_models: dict[str, list[str]], num_classes: int,
) -> None:
    """Smoke-test that TorchVision models can be instantiated without crashing."""
    failures = _run_creation_with_progress(tv_models, num_classes=num_classes)
    if failures:
        msg = "Certaines créations de modèles TorchVision ont échoué:\n" + "\n".join(
            f"- {m} / {n}: {err}" for m, n, err in failures
        )
        pytest.fail(msg)
