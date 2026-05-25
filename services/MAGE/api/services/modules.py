# Copyright (C) 2026 AIMER contributors.

"""Modules service: timm module catalogs and details."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from timm import list_models, list_modules

router = APIRouter(tags=["modules"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """
    Service-local health-check endpoint.

    Returns:
        dict[str, str]: A simple status payload indicating that the modules
            service is running.

    """
    return {"modules_service": "UP"}


@router.get("/module")
async def module_list() -> list[str]:
    """
    List all ``timm`` modules (families / namespaces).

    Returns:
        list[str]: The list of available ``timm`` module names.

    """
    return list_modules()


@router.get("/module/{module_id}/details")
async def module_details(module_id: str) -> dict[str, list[str]]:
    """
    List all ``timm`` models for a specific module.

    Args:
        module_id: The name of the ``timm`` module to inspect.

    Returns:
        dict[str, list[str]]: A mapping containing the module name as key and
            the list of associated model names as value.

    """
    return {module_id: list(list_models(module=module_id))}


@router.get("/module/details")
async def module_all_details() -> dict[str, list[str]]:
    """
    List all models for all ``timm`` modules.

    Returns:
        dict[str, list[str]]: A mapping of each ``timm`` module name to its
            corresponding list of model names.

    """
    return {module: list(list_models(module=module)) for module in list_modules()}


service_app = FastAPI(title="Modules Service")
service_app.include_router(router)
