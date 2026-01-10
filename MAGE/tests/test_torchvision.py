import sys
import time
from types import ModuleType
from urllib.error import URLError

from django.test import TestCase
from tqdm.auto import tqdm

import torchvision.models as tvm


def _safe_list_models(module: ModuleType) -> list[str]:
    """
    TorchVision >= 0.14: tvm.list_models(module=...)
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
    """
    Retourne weights_enum.DEFAULT si disponible, sinon None.
    """
    if not hasattr(tvm, "get_model_weights"):
        return None
    try:
        weights_enum = tvm.get_model_weights(model_name)
        return getattr(weights_enum, "DEFAULT", None)
    except Exception:
        return None


def _safe_get_model(model_name: str, *, weights, num_classes: int):
    """
    1) Si weights != None => essaie get_model(name, weights=weights) (sans num_classes)
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
        raise RuntimeError(f"Model builder not found for {model_name!r} (old torchvision fallback)")

    # Ancienne API: pretrained=True/False
    try:
        return builder(pretrained=True)
    except Exception:
        # num_classes pas standard partout en ancien API, on reste minimaliste
        return builder(pretrained=False)


class TorchvisionModelsTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        modules: dict[str, ModuleType] = {"classification": tvm}

        # Sous-modules “classiques” (selon version torchvision)
        for sub in ("detection", "segmentation", "video", "optical_flow", "quantization"):
            mod = getattr(tvm, sub, None)
            if isinstance(mod, ModuleType):
                modules[sub] = mod

        cls.modules = modules
        cls.models = {name: _safe_list_models(mod) for name, mod in modules.items()}
        cls.num_classes = 10

    def test_model_creation(self) -> None:
        total_models = sum(len(model_list) for model_list in self.models.values())
        is_tty = sys.stdout.isatty() or sys.stderr.isatty()

        bar_fmt = (
            "{l_bar}{bar} | {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

        def create_one(module_name: str, model_name: str):
            pretrained_used = False
            start_time = time.time()

            # 1) tenter weights DEFAULT (si dispo)
            weights = _safe_get_model_weights_default(model_name)
            if weights is not None:
                try:
                    model = _safe_get_model(model_name, weights=weights, num_classes=self.num_classes)
                    pretrained_used = True
                    return model, "weights", time.time() - start_time
                except (URLError, OSError, RuntimeError, ValueError):
                    # download impossible / mismatch / autre -> fallback random
                    pass

            # 2) fallback sans weights (random)
            start_time = time.time()
            model = _safe_get_model(model_name, weights=None, num_classes=self.num_classes)
            return model, "no-weights", time.time() - start_time

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
                ascii=False,
            ) as p_global:
                for module_name, model_list in self.models.items():
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

                            with self.subTest(module=module_name, model=model_name):
                                model, status, elapsed = create_one(module_name, model_name)
                                # éviter de garder plein de modèles en mémoire
                                del model

                                p_mod.set_postfix_str(f"{model_name} • {status} • {elapsed:.2f}s")
                                p_global.set_postfix_str(
                                    f"{module_name} • {model_name} • {status} • {elapsed:.2f}s"
                                )

                            p_mod.update(1)
                            p_global.update(1)
        else:
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
                for module_name, model_list in self.models.items():
                    for model_name in model_list:
                        pbar.set_postfix_str(f"{module_name} • {model_name}")
                        with self.subTest(module=module_name, model=model_name):
                            model, _, _ = create_one(module_name, model_name)
                            del model
                        pbar.update(1)
