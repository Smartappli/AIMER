# Copyright (C) 2026 AIMER contributors.

"""Libraries service: exposes optional AI/ML dependency versions."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from MAGE.api.common import LIBRARIES_BY_CATEGORY, safe_version

router = APIRouter(tags=["libraries"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """
    Service-local health-check endpoint.

    Returns:
        dict[str, str]: A simple status payload indicating that the libraries
            service is running.

    """
    return {"libraries_service": "UP"}


@router.get("/libraries")
async def libraries() -> dict[str, dict[str, str | None]]:
    """
    Return versions of key AI/ML libraries if installed.

    Returns:
        dict[str, dict[str, str | None]]: A mapping by library category, where
            each category contains package names associated with their detected
            version, or ``None`` when the package is not installed.

    """
    return {
        category: {
            package_name: safe_version(package_name, module_name)
            for package_name, module_name in packages.items()
        }
        for category, packages in LIBRARIES_BY_CATEGORY.items()
    }


service_app = FastAPI(title="Libraries Service")
service_app.include_router(router)
