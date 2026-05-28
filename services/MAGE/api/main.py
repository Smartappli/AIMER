# Copyright (C) 2026 AIMER contributors.

"""API gateway exposing both REST endpoints and MCP routes."""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from fastmcp import FastMCP

from api.services.augmentations import augmentation_presets, validate_preset
from api.services.augmentations import service_app as augmentations_service
from api.services.encoders import list_encoders
from api.services.encoders import service_app as encoders_service
from api.services.libraries import libraries
from api.services.libraries import service_app as libraries_service
from api.services.models import (
    are_pretrained,
    is_pretrained,
    model_list,
)
from api.services.models import (
    service_app as models_service,
)
from api.services.modules import (
    module_all_details,
    module_details,
    module_list,
)
from api.services.modules import (
    service_app as modules_service,
)

api = FastAPI(title="MAGE API Gateway")

PUBLIC_PATHS = {"/", "/healthz"}


def _configured_api_key() -> str:
    """Return the configured service API key, if any."""
    return os.getenv("MAGE_API_KEY", "").strip()


def _request_api_key(request: Request) -> str:
    """Extract API key from Authorization bearer or X-API-Key headers."""
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return request.headers.get("x-api-key", "").strip()


@api.get("/")
async def read_root() -> dict[str, str]:
    """
    Gateway-level health-check endpoint.

    Returns:
        dict[str, str]: A simple status payload indicating that the API gateway
            is running.

    """
    return {"API": "UP"}


@api.get("/healthz")
async def healthz() -> dict[str, str]:
    """
    Gateway-level liveness endpoint for deployment smoke tests.

    Returns:
        dict[str, str]: A stable payload indicating the service is running.

    """
    return {"service": "MAGE", "status": "ok"}


# Backward-compatible routes served directly by the gateway.
api.add_api_route("/libraries", libraries, methods=["GET"], tags=["libraries"])
api.add_api_route(
    "/augmentations",
    augmentation_presets,
    methods=["GET"],
    tags=["augmentations"],
)
api.add_api_route(
    "/augmentations/{preset_name}/validate",
    validate_preset,
    methods=["GET"],
    tags=["augmentations"],
)
api.add_api_route("/encoders", list_encoders, methods=["GET"], tags=["encoders"])
api.add_api_route("/model", model_list, methods=["GET"], tags=["models"])
api.add_api_route(
    "/model/{model_id}/is_pretrained",
    is_pretrained,
    methods=["GET"],
    tags=["models"],
)
api.add_api_route(
    "/model/is_pretrained",
    are_pretrained,
    methods=["GET"],
    tags=["models"],
)
api.add_api_route("/module", module_list, methods=["GET"], tags=["modules"])
api.add_api_route(
    "/module/{module_id}/details",
    module_details,
    methods=["GET"],
    tags=["modules"],
)
api.add_api_route(
    "/module/details",
    module_all_details,
    methods=["GET"],
    tags=["modules"],
)

# Microservice mounts (service-style boundaries under dedicated prefixes).
api.mount("/services/libraries", libraries_service)
api.mount("/services/augmentations", augmentations_service)
api.mount("/services/encoders", encoders_service)
api.mount("/services/models", models_service)
api.mount("/services/modules", modules_service)

mcp = FastMCP.from_fastapi(app=api, name="Timm API MCP")
mcp_app = mcp.http_app(path="/mcp")

app = FastAPI(
    title="REST + MCP",
    routes=[*mcp_app.routes, *api.routes],
    lifespan=mcp_app.lifespan,
)


@app.middleware("http")
async def require_service_api_key(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Require a service key for REST and MCP routes when configured."""
    expected_key = _configured_api_key()
    if not expected_key or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    provided_key = _request_api_key(request)
    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)
