"""FastAPI app exposing timm metadata and MCP routes."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastmcp import FastMCP
from timm import is_model_pretrained, list_models, list_modules

api = FastAPI()


def _safe_version(pkg_name: str, module_name: str | None = None) -> str | None:
    """Safely resolve a library version.

    Args:
        pkg_name: The distribution/package name as known by packaging metadata.
        module_name: Optional importable module name to check `__version__`.

    Returns:
        The detected version string, or None if the package/module is absent
        or any error occurs.

    """
    try:
        if module_name:
            mod = __import__(module_name)
            detected = getattr(mod, "__version__", None)
            if isinstance(detected, str) and detected:
                return detected
        return version(pkg_name)
    except PackageNotFoundError:
        return None
    except (ImportError, AttributeError):
        return None


@api.get("/")
async def read_root() -> dict[str, str]:
    """Health-check endpoint."""
    return {"API": "UP"}


@api.get("/libraries")
async def libraries() -> dict[str, dict[str, str | None]]:
    """Return versions of key AI/ML libraries if installed."""
    return {
        "AI": {
            "keras": _safe_version("keras", "keras"),
            "segmentation-models-pytorch": _safe_version(
                "segmentation-models-pytorch",
                "segmentation_models_pytorch",
            ),
            "tensorflow": _safe_version("tensorflow", "tensorflow"),
            "timm": _safe_version("timm", "timm"),
            "torch": _safe_version("torch", "torch"),
        },
    }


@api.get("/model")
async def model_list() -> list[str]:
    """List all model names known by `timm`."""
    return list(list_models())


@api.get("/model/{model_id}/is_pretrained")
async def is_pretrained(model_id: str) -> bool:
    """Check whether a specific `timm` model has pretrained weights."""
    return is_model_pretrained(model_id)


@api.get("/model/is_pretrained")
async def are_pretrained() -> dict[str, dict[str, bool]]:
    """Check pretrained availability for all `timm` models."""
    models = list(list_models())
    return {
        "model_is_pretrained": {model: is_model_pretrained(model) for model in models},
    }


@api.get("/module")
async def module_list() -> list[str]:
    """List all `timm` modules (families / namespaces)."""
    return list_modules()


@api.get("/module/{module_id}/details")
async def module_details(module_id: str) -> dict[str, list[str]]:
    """List all `timm` models for a specific module."""
    return {module_id: list(list_models(module=module_id))}


@api.get("/module/details")
async def module_all_details() -> dict[str, list[str]]:
    """List all models for all `timm` modules."""
    return {module: list(list_models(module=module)) for module in list_modules()}


mcp = FastMCP.from_fastapi(app=api, name="Timm API MCP")
mcp_app = mcp.http_app(path="/mcp")

app = FastAPI(
    title="REST + MCP",
    routes=[*mcp_app.routes, *api.routes],
    lifespan=mcp_app.lifespan,
)
