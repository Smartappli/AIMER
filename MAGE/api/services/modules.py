# Copyright (C) 2026 AIMER contributors.

"""Modules service: timm module catalogs and details."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from timm import list_models, list_modules

router = APIRouter(tags=["modules"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Service-local health-check endpoint."""
    return {"modules_service": "UP"}


@router.get("/module")
async def module_list() -> list[str]:
    """List all ``timm`` modules (families / namespaces)."""
    return list_modules()


@router.get("/module/{module_id}/details")
async def module_details(module_id: str) -> dict[str, list[str]]:
    """List all ``timm`` models for a specific module."""
    return {module_id: list(list_models(module=module_id))}


@router.get("/module/details")
async def module_all_details() -> dict[str, list[str]]:
    """List all models for all ``timm`` modules."""
    return {module: list(list_models(module=module)) for module in list_modules()}


service_app = FastAPI(title="Modules Service")
service_app.include_router(router)
