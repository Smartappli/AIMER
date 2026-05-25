# Copyright (C) 2026 AIMER contributors.

"""Models service: timm model inventory and pretrained capabilities."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from timm import is_model_pretrained, list_models

router = APIRouter(tags=["models"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """
    Service-local health-check endpoint.

    Returns:
        dict[str, str]: A simple status payload indicating that the models
            service is running.

    """
    return {"models_service": "UP"}


@router.get("/model")
async def model_list() -> list[str]:
    """
    List all model names known by ``timm``.

    Returns:
        list[str]: The list of all model names exposed by ``timm``.

    """
    return list(list_models())


@router.get("/model/{model_id}/is_pretrained")
async def is_pretrained(model_id: str) -> bool:
    """
    Check whether a specific ``timm`` model has pretrained weights.

    Args:
        model_id: The name of the ``timm`` model to inspect.

    Returns:
        bool: ``True`` if pretrained weights are available for the given
            model, otherwise ``False``.

    """
    return is_model_pretrained(model_id)


@router.get("/model/is_pretrained")
async def are_pretrained() -> dict[str, dict[str, bool]]:
    """
    Check pretrained availability for all ``timm`` models.

    Returns:
        dict[str, dict[str, bool]]: A payload containing the pretrained status
            of all known ``timm`` models.

    """
    return {"model_is_pretrained": build_pretrained_map(list_models())}


def build_pretrained_map(models: list[str]) -> dict[str, bool]:
    """
    Build pretrained flags for a provided model list.

    Args:
        models: The list of model names to evaluate.

    Returns:
        dict[str, bool]: A mapping from model name to pretrained availability.

    """
    return {model: is_model_pretrained(model) for model in models}


service_app = FastAPI(title="Models Service")
service_app.include_router(router)
