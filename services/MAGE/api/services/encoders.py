# Copyright (C) 2026 AIMER contributors.

"""Encoders service: inspect SMP encoders and TIMM-backed support."""

from __future__ import annotations

from importlib import import_module

from fastapi import APIRouter, FastAPI, HTTPException

router = APIRouter(tags=["encoders"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Service-local health-check endpoint."""
    return {"encoders_service": "UP"}


@router.get("/encoders")
async def list_encoders() -> dict[str, list[str] | int]:
    """List SMP encoders and a subset that are TIMM-backed."""
    try:
        smp = import_module("segmentation_models_pytorch")
        smp_encoders = smp.encoders
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="segmentation_models_pytorch is unavailable"
        ) from exc

    names = sorted(smp_encoders.get_encoder_names())
    timm_backed = [name for name in names if name.startswith("tu-")]
    return {
        "encoders": names,
        "timm_backed_encoders": timm_backed,
        "total": len(names),
        "timm_backed_total": len(timm_backed),
    }


service_app = FastAPI(title="Encoders Service")
service_app.include_router(router)
