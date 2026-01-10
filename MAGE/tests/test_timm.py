import sys
import time

from django.test import TestCase
from timm import create_model, list_models, list_modules
from tqdm.auto import tqdm  # auto = mieux (terminal / notebook)


class TimmModelsTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.models = {
            module: list(list_models(module=module))
            for module in list_modules()
        }
        cls.num_classes = 10

    def test_model_creation(self) -> None:
        total_models = sum(len(model_list) for model_list in self.models.values())
        is_tty = sys.stdout.isatty() or sys.stderr.isatty()

        bar_fmt = (
            "{l_bar}{bar} | {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

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
                                pretrained_used = True
                                start_time = time.time()
                                try:
                                    create_model(model_name, pretrained=True, num_classes=self.num_classes)
                                except RuntimeError:
                                    pretrained_used = False
                                    start_time = time.time()
                                    create_model(model_name, pretrained=False, num_classes=self.num_classes)

                                elapsed = time.time() - start_time
                                status = "pretrained" if pretrained_used else "no-pretrained"
                                p_mod.set_postfix_str(f"{model_name} • {status} • {elapsed:.2f}s")
                                p_global.set_postfix_str(f"{module_name} • {model_name} • {status} • {elapsed:.2f}s")

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
                for module_name, model_list in self.models.items():
                    for model_name in model_list:
                        pbar.set_postfix_str(f"{module_name} • {model_name}")
                        with self.subTest(module=module_name, model=model_name):
                            try:
                                create_model(model_name, pretrained=True, num_classes=self.num_classes)
                            except RuntimeError:
                                create_model(model_name, pretrained=False, num_classes=self.num_classes)
                        pbar.update(1)
