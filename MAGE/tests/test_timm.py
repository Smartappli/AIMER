"""timm model registry smoke tests.

Validate that timm models can be created across modules without crashing,
while keeping progress output readable in both TTY and non-TTY environments.
"""

import importlib.util
import os
import sys
import time

import pytest

TIMM_AVAILABLE = importlib.util.find_spec("timm") is not None
TQDM_AVAILABLE = importlib.util.find_spec("tqdm") is not None

pytestmark = pytest.mark.skipif(
    not (TIMM_AVAILABLE and TQDM_AVAILABLE),
    reason="Optional dependencies `timm` and `tqdm` are required for this test.",
)

if TIMM_AVAILABLE and TQDM_AVAILABLE:
    from timm import create_model, list_models, list_modules
    from tqdm.auto import tqdm


@pytest.fixture(scope="session")
def timm_models() -> dict[str, list[str]]:
    """List timm models grouped by module.

    Optionally limit scope in CI via:
      - ``TIMM_TEST_MODULES="resnet,vit,convnext"``
      - ``TIMM_TEST_LIMIT_PER_MODULE="20"``

    Returns:
        Dictionary mapping module names to model-name lists.

    """
    modules = list(list_modules())

    mods_env = os.getenv("TIMM_TEST_MODULES")
    if mods_env:
        wanted = {m.strip() for m in mods_env.split(",") if m.strip()}
        modules = [m for m in modules if m in wanted]

    limit_env = os.getenv("TIMM_TEST_LIMIT_PER_MODULE")
    limit_per_mod = int(limit_env) if (limit_env and limit_env.isdigit()) else None

    out: dict[str, list[str]] = {}
    for module in modules:
        models = list(list_models(module=module))
        if limit_per_mod is not None:
            models = models[:limit_per_mod]
        out[module] = models
    return out


@pytest.fixture(scope="session")
def num_classes() -> int:
    """Return a small class count for classifier heads.

    Returns:
        Number of classes used to instantiate timm models.

    """
    return 10


@pytest.mark.slow
def test_timm_model_creation(
    timm_models: dict[str, list[str]],
    num_classes: int,
) -> None:
    """Smoke-test that timm models can be instantiated without crashing."""
    total_models = sum(len(model_list) for model_list in timm_models.values())
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    bar_fmt = (
        "{l_bar}{bar} | {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    failures: list[tuple[str, str, str]] = []

    def _try_create(model_name: str) -> tuple[bool, float]:
        """Create one model with pretrained fallback.

        Returns:
            A tuple ``(pretrained_used, elapsed_seconds)``.

        """
        start = time.time()
        try:
            create_model(model_name, pretrained=True, num_classes=num_classes)
            return True, time.time() - start
        except (RuntimeError, OSError, ValueError, TypeError):
            # Fallback sans pretrained (utile si poids indisponibles / offline / etc.)
            start = time.time()
            create_model(model_name, pretrained=False, num_classes=num_classes)
            return False, time.time() - start

    if is_tty:
        _run_creation_tty(
            timm_models=timm_models,
            total_models=total_models,
            bar_fmt=bar_fmt,
            try_create=_try_create,
            failures=failures,
        )
    else:
        _run_creation_non_tty(
            timm_models=timm_models,
            total_models=total_models,
            bar_fmt=bar_fmt,
            try_create=_try_create,
            failures=failures,
        )

    if failures:
        msg = "Certaines créations de modèles ont échoué:\n" + "\n".join(
            f"- {m} / {n}: {err}" for m, n, err in failures
        )
        pytest.fail(msg)


def _run_creation_tty(
    *,
    timm_models: dict[str, list[str]],
    total_models: int,
    bar_fmt: str,
    try_create: object,
    failures: list[tuple[str, str, str]],
) -> None:
    """Run model creation loop with nested TTY progress bars."""
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
        ascii=False,
    ) as p_global:
        for module_name, model_list in timm_models.items():
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
                    model_list=model_list,
                    progress_bars=(p_mod, p_global),
                    try_create=try_create,
                    failures=failures,
                )


def _run_module_creation(
    *,
    module_name: str,
    model_list: list[str],
    progress_bars: tuple[object, object],
    try_create: object,
    failures: list[tuple[str, str, str]],
) -> None:
    """Create all models for one timm module and update progress bars."""
    p_mod, p_global = progress_bars
    for model_name in model_list:
        p_mod.set_postfix_str(model_name)
        p_global.set_postfix_str(f"{module_name} • {model_name}")

        try:
            pretrained_used, elapsed = try_create(model_name)
            status = "pretrained" if pretrained_used else "no-pretrained"
            p_mod.set_postfix_str(f"{model_name} • {status} • {elapsed:.2f}s")
            p_global.set_postfix_str(
                f"{module_name} • {model_name} • {status} • {elapsed:.2f}s",
            )
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            failures.append((module_name, model_name, repr(exc)))

        p_mod.update(1)
        p_global.update(1)


def _run_creation_non_tty(
    *,
    timm_models: dict[str, list[str]],
    total_models: int,
    bar_fmt: str,
    try_create: object,
    failures: list[tuple[str, str, str]],
) -> None:
    """Run model creation loop with a single non-TTY progress bar."""
    with tqdm(
        total=total_models,
        desc="timm create_model",
        unit="model",
        dynamic_ncols=False,
        bar_format=bar_fmt,
        mininterval=0.2,
        smoothing=0.1,
        leave=True,
        disable=False,
        file=sys.stdout,
        ascii=False,
    ) as pbar:
        for module_name, model_list in timm_models.items():
            for model_name in model_list:
                pbar.set_postfix_str(f"{module_name} • {model_name}")
                try:
                    try_create(model_name)
                except (RuntimeError, OSError, ValueError, TypeError) as exc:
                    failures.append((module_name, model_name, repr(exc)))
                pbar.update(1)
