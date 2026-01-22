from fastapi import FastAPI
from fastmcp import FastMCP

api = FastAPI()


@api.get("/")
async def read_root():
    return {"API": "UP"}


@api.get("/libraries")
async def libraries():
    from importlib.metadata import version, PackageNotFoundError

    def safe_version(pkg_name: str, module_name: str | None = None):
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
                "segmentation-models-pytorch",
                "segmentation_models_pytorch",
            ),
            "tensorflow": safe_version("tensorflow", "tensorflow"),
            "timm": safe_version("timm", "timm"),
            "torch": safe_version("torch", "torch"),
        }
    }


@api.get("/model")
async def model_list():
    from timm import list_models

    return list(list_models())


@api.get("/model/{model_id}/is_pretrained")
async def is_pretrained(model_id: str):
    from timm import is_model_pretrained

    return is_model_pretrained(model_id)


@api.get("/model/is_pretrained")
async def are_pretrained():
    from timm import list_models, is_model_pretrained

    models = list(list_models())
    return {
        "model_is_pretrained": {
            model: is_model_pretrained(model) for model in models
        }
    }


@api.get("/module")
async def module_list():
    from timm import list_modules

    return list_modules()


@api.get("/module/{module_id}/details")
async def module_details(module_id: str):
    from timm import list_models

    return {module_id: list(list_models(module=module_id))}


@api.get("/module/details")
async def module_all_details():
    from timm import list_models, list_modules

    return {
        module: list(list_models(module=module)) for module in list_modules()
    }


mcp = FastMCP.from_fastapi(
    app=api, name="Timm API MCP"
)  # auto-tooling :contentReference[oaicite:1]{index=1}
mcp_app = mcp.http_app(
    path="/mcp"
)  # endpoint MCP :contentReference[oaicite:2]{index=2}

app = FastAPI(
    title="REST + MCP",
    routes=[*mcp_app.routes, *api.routes],
    lifespan=mcp_app.lifespan,  # important: session manager MCP :contentReference[oaicite:3]{index=3}
)
