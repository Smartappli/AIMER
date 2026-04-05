# Copyright (C) 2026 AIMER contributors.

"""API gateway exposing both REST endpoints and MCP routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastmcp import FastMCP

from MAGE.api.services.libraries import libraries, service_app as libraries_service
from MAGE.api.services.models import (
    are_pretrained,
    is_pretrained,
    model_list,
    service_app as models_service,
)
from MAGE.api.services.modules import (
    module_all_details,
    module_details,
    module_list,
    service_app as modules_service,
)

api = FastAPI(title="MAGE API Gateway")


@api.get("/")
async def read_root() -> dict[str, str]:
    """Gateway-level health-check endpoint."""
    return {"API": "UP"}


# Backward-compatible routes served directly by the gateway.
api.add_api_route("/libraries", libraries, methods=["GET"], tags=["libraries"])
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
api.mount("/services/models", models_service)
api.mount("/services/modules", modules_service)

mcp = FastMCP.from_fastapi(app=api, name="Timm API MCP")
mcp_app = mcp.http_app(path="/mcp")

app = FastAPI(
    title="REST + MCP",
    routes=[*mcp_app.routes, *api.routes],
    lifespan=mcp_app.lifespan,
)
