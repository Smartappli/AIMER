import os
import sys
import time

import pytest
from timm import create_model, list_models, list_modules
from tqdm.auto import tqdm


@pytest.fixture(scope="session")
def timm_models() -> dict[str, list[str]]:
    """Liste tous les modèles timm, groupés par module.
    Optionnel: limiter via variables d'env pour éviter un test interminable en CI.
      - TIMM_TEST_MODULES="resnet,vit,convnext"
      - TIMM_TEST_LIMIT_PER_MODULE="20"
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
    return 10


@pytest.mark.slow
def test_timm_model_creation(
    timm_models: dict[str, list[str]], num_classes: int,
) -> None:
    total_models = sum(len(model_list) for model_list in timm_models.values())
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    bar_fmt = (
        "{l_bar}{bar} | {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    failures: list[tuple[str, str, str]] = []

    def _try_create(model_name: str) -> tuple[bool, float]:
        """Retourne (pretrained_used, elapsed_s)."""
        start = time.time()
        try:
            create_model(model_name, pretrained=True, num_classes=num_classes)
            return True, time.time() - start
        except Exception:
            # Fallback sans pretrained (utile si poids indisponibles / offline / etc.)
            start = time.time()
            create_model(model_name, pretrained=False, num_classes=num_classes)
            return False, time.time() - start

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
                    for model_name in model_list:
                        p_mod.set_postfix_str(model_name)
                        p_global.set_postfix_str(f"{module_name} • {model_name}")

                        try:
                            pretrained_used, elapsed = _try_create(model_name)
                            status = (
                                "pretrained" if pretrained_used else "no-pretrained"
                            )
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
                        _try_create(model_name)
                    except Exception as e:
                        failures.append((module_name, model_name, repr(e)))
                    pbar.update(1)

    assert not failures, "Certaines créations de modèles ont échoué:\n" + "\n".join(
        f"- {m} / {n}: {err}" for m, n, err in failures
    )
