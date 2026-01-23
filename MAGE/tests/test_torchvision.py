import os
import sys
import time
from types import ModuleType
from urllib.error import URLError

import pytest
import torchvision.models as tvm
from tqdm.auto import tqdm


def _safe_list_models(module: ModuleType) -> list[str]:
    """TorchVision >= 0.14: tvm.list_models(module=...)
    Fallback: introspection (moins fiable, mais évite de casser si ancienne version)
    """
    if hasattr(tvm, "list_models"):
        return list(tvm.list_models(module=module))

    # --- Fallback ancien torchvision (pas de registry) ---
    import inspect

    names: list[str] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        # Heuristique: builders torchvision sont souvent en lowercase
        if name.lower() != name:
            continue
        try:
            inspect.signature(fn)
        except Exception:
            continue
        names.append(name)
    return sorted(set(names))


def _safe_get_model_weights_default(model_name: str):
    """Retourne weights_enum.DEFAULT si disponible, sinon None."""
    if not hasattr(tvm, "get_model_weights"):
        return None
    try:
        weights_enum = tvm.get_model_weights(model_name)
        return getattr(weights_enum, "DEFAULT", None)
    except Exception:
        return None


def _safe_get_model(model_name: str, *, weights, num_classes: int):
    """1) Si weights != None => essaie get_model(name, weights=weights) (sans num_classes)
    2) Sinon => essaie get_model(name, weights=None, num_classes=...) puis fallback sans num_classes
    """
    if hasattr(tvm, "get_model"):
        get_model = tvm.get_model

        if weights is not None:
            # Important: éviter num_classes avec weights (souvent incompatible)
            return get_model(model_name, weights=weights)

        # weights=None => on tente num_classes si supporté
        try:
            return get_model(model_name, weights=None, num_classes=num_classes)
        except TypeError:
            return get_model(model_name, weights=None)

    # --- Fallback très ancien torchvision: appel direct au builder ---
    builder = getattr(tvm, model_name, None)
    if builder is None:
        raise RuntimeError(
            f"Model builder not found for {model_name!r} (old torchvision fallback)",
        )

    # Ancienne API: pretrained=True/False
    try:
        return builder(pretrained=True)
    except Exception:
        return builder(pretrained=False)


@pytest.fixture(scope="session")
def tv_modules() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {"classification": tvm}

    # Sous-modules “classiques” (selon version torchvision)
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

    # Optionnel: filtrer modules en CI
    # export TV_TEST_MODULES="classification,detection"
    wanted = os.getenv("TV_TEST_MODULES")
    if wanted:
        keep = {m.strip() for m in wanted.split(",") if m.strip()}
        modules = {k: v for k, v in modules.items() if k in keep}

    return modules


@pytest.fixture(scope="session")
def tv_models(tv_modules: dict[str, ModuleType]) -> dict[str, list[str]]:
    models = {name: _safe_list_models(mod) for name, mod in tv_modules.items()}

    # Optionnel: limiter par module en CI
    # export TV_TEST_LIMIT_PER_MODULE="20"
    lim = os.getenv("TV_TEST_LIMIT_PER_MODULE")
    limit = int(lim) if (lim and lim.isdigit()) else None
    if limit is not None:
        models = {k: v[:limit] for k, v in models.items()}

    return models


@pytest.fixture(scope="session")
def num_classes() -> int:
    return 10


@pytest.mark.slow
def test_torchvision_model_creation(
    tv_models: dict[str, list[str]],
    num_classes: int,
) -> None:
    total_models = sum(len(model_list) for model_list in tv_models.values())
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    bar_fmt = (
        "{l_bar}{bar} | {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    failures: list[tuple[str, str, str]] = []

    def create_one(module_name: str, model_name: str):
        # 1) tenter weights DEFAULT (si dispo)
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
            except (URLError, OSError, RuntimeError, ValueError):
                # download impossible / mismatch / autre -> fallback random
                pass

        # 2) fallback sans weights (random)
        start_time = time.time()
        model = _safe_get_model(
            model_name,
            weights=None,
            num_classes=num_classes,
        )
        return model, "no-weights", time.time() - start_time

    if is_tty:
        # ✅ 2 barres (global + module) proprement
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
                        p_global.set_postfix_str(
                            f"{module_name} • {model_name}",
                        )

                        try:
                            model, status, elapsed = create_one(
                                module_name,
                                model_name,
                            )
                            del model  # éviter d'accumuler en mémoire
                            p_mod.set_postfix_str(
                                f"{model_name} • {status} • {elapsed:.2f}s",
                            )
                            p_global.set_postfix_str(
                                f"{module_name} • {model_name} • {status} • {elapsed:.2f}s",
                            )
                        except Exception as e:
                            failures.append((module_name, model_name, repr(e)))

                        p_mod.update(1)
                        p_global.update(1)
    else:
        # ✅ Pas de TTY => 1 seule barre (visible même dans des logs)
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
            ascii=False,
        ) as pbar:
            for module_name, model_list in tv_models.items():
                for model_name in model_list:
                    pbar.set_postfix_str(f"{module_name} • {model_name}")
                    try:
                        model, _, _ = create_one(module_name, model_name)
                        del model
                    except Exception as e:
                        failures.append((module_name, model_name, repr(e)))
                    pbar.update(1)

    assert (
        not failures
    ), "Certaines créations de modèles TorchVision ont échoué:\n" + "\n".join(
        f"- {m} / {n}: {err}" for m, n, err in failures
    )
