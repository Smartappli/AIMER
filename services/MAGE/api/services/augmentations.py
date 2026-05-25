# Copyright (C) 2026 AIMER contributors.

"""Augmentations service: expose Albumentations availability and presets."""

from __future__ import annotations

from importlib import import_module

from fastapi import APIRouter, FastAPI, HTTPException

router = APIRouter(tags=["augmentations"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """
    Service-local health-check endpoint.

    Returns:
        dict[str, str]: Simple status payload for this microservice.

    """
    return {"augmentations_service": "UP"}


@router.get("/augmentations")
async def augmentation_presets() -> dict[str, object]:
    """
    Return baseline augmentation presets suitable for image datasets.

    Returns:
        dict[str, object]: Albumentations version plus classification and
            segmentation-oriented preset recipes.

    Raises:
        HTTPException: If Albumentations is unavailable.

    """
    try:
        alb = import_module("albumentations")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="albumentations is unavailable",
        ) from exc

    alb_version = getattr(alb, "__version__", None)
    return {
        "library": "albumentations",
        "version": alb_version,
        "presets": {
            "classification_basic": [
                "HorizontalFlip(p=0.5)",
                (
                    "ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, "
                    "rotate_limit=15, p=0.5)"
                ),
                "RandomBrightnessContrast(p=0.3)",
                "Normalize()",
            ],
            "segmentation_basic": [
                "HorizontalFlip(p=0.5)",
                "VerticalFlip(p=0.2)",
                "RandomResizedCrop(height=256, width=256, scale=(0.7, 1.0), p=0.5)",
                "Normalize()",
            ],
        },
    }


@router.get("/augmentations/{preset_name}/validate")
async def validate_preset(preset_name: str) -> dict[str, object]:
    """
    Validate that a preset uses transforms available in Albumentations.

    Args:
        preset_name: Name of the preset to validate.

    Returns:
        dict[str, object]: Validation report including missing transform names.

    Raises:
        HTTPException: If Albumentations is unavailable or preset is unknown.

    """
    payload = await augmentation_presets()
    presets = payload["presets"]
    if not isinstance(presets, dict) or preset_name not in presets:
        raise HTTPException(status_code=404, detail="unknown augmentation preset")

    try:
        alb = import_module("albumentations")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="albumentations is unavailable",
        ) from exc

    steps = presets[preset_name]
    if not isinstance(steps, list):
        raise HTTPException(status_code=500, detail="invalid preset structure")

    transform_names = [step.split("(")[0] for step in steps]
    missing = [name for name in transform_names if getattr(alb, name, None) is None]
    return {
        "preset": preset_name,
        "transforms": transform_names,
        "missing_transforms": missing,
        "is_valid": len(missing) == 0,
    }


service_app = FastAPI(title="Augmentations Service")
service_app.include_router(router)
