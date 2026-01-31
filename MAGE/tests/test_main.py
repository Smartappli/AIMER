import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient


# ---------- Fakes injected BEFORE importing main.py ----------

class FakeTimm(ModuleType):
    def __init__(self):
        super().__init__("timm")
        self.__version__ = "9.9.9"
        self._models = ["resnet18", "vit_base_patch16_224", "convnext_tiny"]
        self._modules = ["resnet", "vit", "convnext"]
        self._pretrained = {
            "resnet18": True,
            "vit_base_patch16_224": False,
            "convnext_tiny": True,
        }

    def list_models(self, module=None):
        if module is None:
            return list(self._models)
        if module == "resnet":
            return ["resnet18"]
        if module == "vit":
            return ["vit_base_patch16_224"]
        if module == "convnext":
            return ["convnext_tiny"]
        return []

    def list_modules(self):
        return list(self._modules)

    def is_model_pretrained(self, model_id: str) -> bool:
        return bool(self._pretrained.get(model_id, False))


class FakeMCPApp:
    def __init__(self):
        from fastapi import FastAPI

        self._app = FastAPI()

        @self._app.get("/mcp/ping")
        async def ping():
            return {"mcp": "ok"}

        self.routes = self._app.routes
        self.lifespan = self._app.router.lifespan_context


class FakeFastMCP:
    @classmethod
    def from_fastapi(cls, app, name: str):
        return cls()

    def http_app(self, path="/mcp"):
        return FakeMCPApp()


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def app_module(monkeypatch):
    """
    Import MAGE/api/main.py with timm and fastmcp mocked.
    """
    # Fake timm
    monkeypatch.setitem(sys.modules, "timm", FakeTimm())

    # Fake fastmcp
    fastmcp_mod = ModuleType("fastmcp")
    fastmcp_mod.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_mod)

    # Path: MAGE/tests/test_main.py -> MAGE/api/main.py
    test_dir = Path(__file__).resolve().parent
    mage_dir = test_dir.parent
    main_py = mage_dir / "api" / "main.py"

    if not main_py.exists():
        raise FileNotFoundError(f"main.py not found at {main_py}")

    return import_module_from_path("mage_api_main_under_test", main_py)


@pytest.fixture()
def client(app_module):
    return TestClient(app_module.app)


# ---------- Tests ----------

def test_root_healthcheck(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"API": "UP"}


def test_model_list(client):
    r = client.get("/model")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert "resnet18" in data


def test_is_pretrained_single(client):
    r = client.get("/model/resnet18/is_pretrained")
    assert r.status_code == 200
    assert r.json() is True

    r2 = client.get("/model/vit_base_patch16_224/is_pretrained")
    assert r2.status_code == 200
    assert r2.json() is False

    r3 = client.get("/model/unknown/is_pretrained")
    assert r3.status_code == 200
    assert r3.json() is False


def test_are_pretrained_all_models(client):
    r = client.get("/model/is_pretrained")
    assert r.status_code == 200
    payload = r.json()
    assert "model_is_pretrained" in payload
    m = payload["model_is_pretrained"]
    assert m["resnet18"] is True
    assert m["vit_base_patch16_224"] is False


def test_module_list(client):
    r = client.get("/module")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert "resnet" in data


def test_module_details(client):
    r = client.get("/module/resnet/details")
    assert r.status_code == 200
    data = r.json()
    assert "resnet" in data
    assert data["resnet"] == ["resnet18"]


def test_module_all_details(client):
    r = client.get("/module/details")
    assert r.status_code == 200
    data = r.json()
    assert data["resnet"] == ["resnet18"]
    assert data["vit"] == ["vit_base_patch16_224"]
    assert data["convnext"] == ["convnext_tiny"]


def test_mcp_route_is_present(client):
    r = client.get("/mcp/ping")
    assert r.status_code == 200
    assert r.json() == {"mcp": "ok"}


def test_libraries_versions_prefers_module_dunder_version(client):
    r = client.get("/libraries")
    assert r.status_code == 200
    ai = r.json()["AI"]
    assert ai["timm"] == "9.9.9"


def test_libraries_missing_packages_return_none(client, monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"keras", "tensorflow", "segmentation_models_pytorch", "torch"}:
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    import importlib.metadata as md

    def fake_version(_):
        raise md.PackageNotFoundError

    monkeypatch.setattr(md, "version", fake_version)

    r = client.get("/libraries")
    assert r.status_code == 200
    ai = r.json()["AI"]

    assert ai["timm"] == "9.9.9"
    assert ai["keras"] is None
    assert ai["tensorflow"] is None
    assert ai["torch"] is None
    assert ai["segmentation-models-pytorch"] is None


def test_libraries_handles_unexpected_exception(client, monkeypatch):
    real_import = __import__

    def boom_import(name, *args, **kwargs):
        if name == "keras":
            raise RuntimeError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", boom_import)

    r = client.get("/libraries")
    assert r.status_code == 200
    ai = r.json()["AI"]
    assert ai["keras"] is None
