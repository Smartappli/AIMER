# Copyright (C) 2026 AIMER contributors.

"""Checks for AI library registry consistency."""

from MAGE.api.common import LIBRARIES_BY_CATEGORY


def test_ai_registry_includes_augmentation_and_modeling_libs() -> None:
    """
    AI registry should track albumentations, pycaret, timm and SMP.

    Raises:
        AssertionError: If one expected AI library mapping is missing or wrong.

    """
    ai = LIBRARIES_BY_CATEGORY["AI"]
    if ai["albumentations"] != "albumentations":
        msg = "albumentations should map to albumentations module"
        raise AssertionError(msg)
    if ai["pycaret"] != "pycaret":
        msg = "pycaret should map to pycaret module"
        raise AssertionError(msg)
    if ai["timm"] != "timm":
        msg = "timm should map to timm module"
        raise AssertionError(msg)
    if ai["segmentation-models-pytorch"] != "segmentation_models_pytorch":
        msg = "SMP package should map to segmentation_models_pytorch module"
        raise AssertionError(msg)
