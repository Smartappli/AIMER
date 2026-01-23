from fastapi import FastAPI
from fastmcp import FastMCP

api = FastAPI()


@api.get("/")
async def read_root():
    """Health-check endpoint.

    Returns:
        A small JSON payload indicating that the REST API is up.

    """
    return {"API": "UP"}


@api.get("/libraries")
async def libraries():
    """Return versions of key AI/ML libraries if installed.

    Uses `importlib.metadata.version()` when possible, but can also try to import
    a module and read its `__version__` attribute when a package name and module
    name differ.

    Returns:
        A nested JSON structure containing library versions (or None if missing).

    """
    from importlib.metadata import PackageNotFoundError, version

    def safe_version(pkg_name: str, module_name: str | None = None):
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
                v = getattr(mod, "__version__", None)
                if v:
                    return v
            return version(pkg_name)
        except PackageNotFoundError:
            return None
        except Exception:
            return None

    return {
        "AI": {
            "keras": safe_version("keras", "keras"),
            "segmentation-models-pytorch": safe_version(
                "segmentation-models-pytorch", "segmentation_models_pytorch",
            ),
            "tensorflow": safe_version("tensorflow", "tensorflow"),
            "timm": safe_version("timm", "timm"),
            "torch": safe_version("torch", "torch"),
        },
    }


@api.get("/model")
async def model_list():
    """List all model names known by `timm`.

    Returns:
        A list of model IDs (strings).

    """
    from timm import list_models

    return list(list_models())


@api.get("/model/{model_id}/is_pretrained")
async def is_pretrained(model_id: str):
    """Check whether a specific `timm` model ID has pretrained weights available.

    Args:
        model_id: The `timm` model identifier.

    Returns:
        True if pretrained weights are available for the model, else False.

    """
    from timm import is_model_pretrained

    return is_model_pretrained(model_id)


@api.get("/model/is_pretrained")
async def are_pretrained():
    """Check pretrained availability for all `timm` models.

    Returns:
        A dict mapping each model ID to a boolean indicating pretrained support.

    """
    from timm import is_model_pretrained, list_models

    models = list(list_models())
    return {
        "model_is_pretrained": {model: is_model_pretrained(model) for model in models},
    }


@api.get("/module")
async def module_list():
    """List all `timm` modules (model families / namespaces).

    Returns:
        A list of module names.

    """
    from timm import list_modules

    return list_modules()


@api.get("/module/{module_id}/details")
async def module_details(module_id: str):
    """List all `timm` models for a specific module.

    Args:
        module_id: A module name returned by `/module`.

    Returns:
        A dict `{module_id: [model_ids...]}` for that module.

    """
    from timm import list_models

    return {module_id: list(list_models(module=module_id))}


@api.get("/module/details")
async def module_all_details():
    """List all models for all `timm` modules.

    Returns:
        A dict mapping each module name to the list of model IDs in that module.

    """
    from timm import list_models, list_modules

    return {module: list(list_models(module=module)) for module in list_modules()}


mcp = FastMCP.from_fastapi(
    app=api, name="Timm API MCP",
)  # auto-tooling :contentReference[oaicite:1]{index=1}
mcp_app = mcp.http_app(
    path="/mcp",
)  # endpoint MCP :contentReference[oaicite:2]{index=2}

app = FastAPI(
    title="REST + MCP",
    routes=[*mcp_app.routes, *api.routes],
    lifespan=mcp_app.lifespan,  # important: session manager MCP :contentReference[oaicite:3]{index=3}
)
