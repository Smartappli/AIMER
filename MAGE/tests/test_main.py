"""Tests for the FastAPI endpoints defined in ``MAGE/api/main.py``."""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeTimm(ModuleType):
    """In-memory stand-in for ``timm`` used by API tests."""

    def __init__(self) -> None:
        """Initialize deterministic fake timm metadata."""
        super().__init__("timm")
        self.__version__ = "9.9.9"
        self._models = ["resnet18", "vit_base_patch16_224", "convnext_tiny"]
        self._modules = ["resnet", "vit", "convnext"]
        self._pretrained = {
            "resnet18": True,
            "vit_base_patch16_224": False,
            "convnext_tiny": True,
        }

    def list_models(self, module: str | None = None) -> list[str]:
        """Return all models or those in one fake module.

        Returns:
            list[str]: Matching fake model identifiers.

        """
        if module is None:
            return list(self._models)
        mapping = {
            "resnet": ["resnet18"],
            "vit": ["vit_base_patch16_224"],
            "convnext": ["convnext_tiny"],
        }
        return list(mapping.get(module, []))

    def list_modules(self) -> list[str]:
        """Return fake module names.

        Returns:
            list[str]: Available fake module names.

        """
        return list(self._modules)

    def is_model_pretrained(self, model_id: str) -> bool:
        """Tell whether a fake model is marked as pretrained.

        Returns:
            bool: ``True`` for pretrained fake models.

        """
        return bool(self._pretrained.get(model_id, False))


class FakeMCPApp:
    """Minimal fake app exposing the MCP ping route."""

    def __init__(self) -> None:
        """Build a minimal FastAPI app with one route."""
        self._app = FastAPI()

        @self._app.get("/mcp/ping")
        def ping() -> dict[str, str]:
            return {"mcp": "ok"}

        self.routes = self._app.routes
        self.lifespan = self._app.router.lifespan_context


class FakeFastMCP:
    """Fake ``FastMCP`` facade used while importing the API module."""

    @classmethod
    def from_fastapi(cls, _app: FastAPI, _name: str) -> FakeFastMCP:
        """Return a fake wrapper instance.

        Returns:
            FakeFastMCP: Fake MCP wrapper.

        """
        return cls()

    @staticmethod
    def http_app(_path: str = "/mcp") -> FakeMCPApp:
        """Return a fake MCP app.

        Returns:
            FakeMCPApp: Fake MCP HTTP app.

        """
        return FakeMCPApp()


def import_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Import a module from an explicit file path.

    Returns:
        ModuleType: Imported module instance.

    Raises:
        RuntimeError: If a module spec cannot be created.

    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        msg = f"Cannot create spec for {file_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``MAGE/api/main.py`` with timm and fastmcp mocked.

    Returns:
        ModuleType: Imported API module under test.

    Raises:
        FileNotFoundError: If ``MAGE/api/main.py`` cannot be located.

    """
    monkeypatch.setitem(sys.modules, "timm", FakeTimm())

    fastmcp_mod = ModuleType("fastmcp")
    fastmcp_mod.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_mod)

    main_py = Path(__file__).resolve().parent.parent / "api" / "main.py"
    if not main_py.exists():
        msg = f"main.py not found at {main_py}"
        raise FileNotFoundError(msg)

    return import_module_from_path("mage_api_main_under_test", main_py)


@pytest.fixture
def client(app_module: ModuleType) -> TestClient:
    """Build a test client for the imported app module.

    Returns:
        TestClient: Client bound to the app under test.

    """
    return TestClient(app_module.app)


def check(condition: object, message: str) -> None:
    """Raise an error if a condition is false.

    Raises:
        AssertionError: If ``condition`` is falsy.

    """
    if not condition:
        raise AssertionError(message)


HTTP_OK = 200


def test_root_healthcheck(client: TestClient) -> None:
    """`/` should report API availability."""
    response = client.get("/")
    check(response.status_code == HTTP_OK, "Expected 200 on root endpoint")
    check(response.json() == {"API": "UP"}, "Unexpected root payload")


def test_model_list(client: TestClient) -> None:
    """`/model` should return the fake timm model list."""
    response = client.get("/model")
    check(response.status_code == HTTP_OK, "Expected 200 on /model")
    data = response.json()
    check(isinstance(data, list), "Expected list from /model")
    check("resnet18" in data, "resnet18 missing from /model response")


def test_is_pretrained_single(client: TestClient) -> None:
    """Per-model pretrained endpoint should reflect fake metadata."""
    response = client.get("/model/resnet18/is_pretrained")
    check(response.status_code == HTTP_OK, "Expected 200 for resnet18")
    check(response.json() is True, "resnet18 should be pretrained")

    response_vit = client.get("/model/vit_base_patch16_224/is_pretrained")
    check(response_vit.status_code == HTTP_OK, "Expected 200 for vit")
    check(response_vit.json() is False, "vit should not be pretrained")

    response_unknown = client.get("/model/unknown/is_pretrained")
    check(response_unknown.status_code == HTTP_OK, "Expected 200 for unknown")
    check(response_unknown.json() is False, "unknown model should be False")


def test_are_pretrained_all_models(client: TestClient) -> None:
    """Aggregate pretrained endpoint should include known fake models."""
    response = client.get("/model/is_pretrained")
    check(response.status_code == HTTP_OK, "Expected 200 on aggregate endpoint")
    payload = response.json()
    check("model_is_pretrained" in payload, "Missing model_is_pretrained key")
    model_flags = payload["model_is_pretrained"]
    check(model_flags["resnet18"] is True, "resnet18 flag should be True")
    check(
        model_flags["vit_base_patch16_224"] is False,
        "vit_base_patch16_224 flag should be False",
    )


def test_module_list(client: TestClient) -> None:
    """`/module` should return fake module names."""
    response = client.get("/module")
    check(response.status_code == HTTP_OK, "Expected 200 on /module")
    data = response.json()
    check(isinstance(data, list), "Expected list from /module")
    check("resnet" in data, "resnet module missing")


def test_module_details(client: TestClient) -> None:
    """`/module/<name>/details` should return module-scoped model lists."""
    response = client.get("/module/resnet/details")
    check(response.status_code == HTTP_OK, "Expected 200 on module details")
    data = response.json()
    check("resnet" in data, "resnet key missing in module details")
    check(data["resnet"] == ["resnet18"], "Unexpected resnet details payload")


def test_module_all_details(client: TestClient) -> None:
    """`/module/details` should include all fake module mappings."""
    response = client.get("/module/details")
    check(response.status_code == HTTP_OK, "Expected 200 on all module details")
    data = response.json()
    check(data["resnet"] == ["resnet18"], "resnet mapping mismatch")
    check(data["vit"] == ["vit_base_patch16_224"], "vit mapping mismatch")
    check(data["convnext"] == ["convnext_tiny"], "convnext mapping mismatch")


def test_mcp_route_is_present(client: TestClient) -> None:
    """Fake MCP route should be mounted under `/mcp/ping`."""
    response = client.get("/mcp/ping")
    check(response.status_code == HTTP_OK, "Expected 200 on /mcp/ping")
    check(response.json() == {"mcp": "ok"}, "Unexpected MCP ping payload")


def test_libraries_versions_prefers_module_dunder_version(client: TestClient) -> None:
    """`/libraries` should expose the fake timm ``__version__``."""
    response = client.get("/libraries")
    check(response.status_code == HTTP_OK, "Expected 200 on /libraries")
    ai = response.json()["AI"]
    check(ai["timm"] == "9.9.9", "timm version should come from fake module")


def test_libraries_missing_packages_return_none(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional AI deps should be reported as ``None``."""
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"keras", "tensorflow", "segmentation_models_pytorch", "torch"}:
            msg = "not installed"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    def fake_version(_: str) -> str:
        raise md.PackageNotFoundError

    monkeypatch.setattr(md, "version", fake_version)

    response = client.get("/libraries")
    check(response.status_code == HTTP_OK, "Expected 200 on /libraries fallback")
    ai = response.json()["AI"]

    check(ai["timm"] == "9.9.9", "timm version should still resolve")
    check(ai["keras"] is None, "keras should be None when missing")
    check(ai["tensorflow"] is None, "tensorflow should be None when missing")
    check(ai["torch"] is None, "torch should be None when missing")
    check(
        ai["segmentation-models-pytorch"] is None,
        "segmentation-models-pytorch should be None when missing",
    )


def test_libraries_handles_unexpected_exception(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected import failures should also degrade to ``None`` values."""
    real_import = __import__

    def boom_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "keras":
            msg = "boom"
            raise RuntimeError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", boom_import)

    response = client.get("/libraries")
    check(response.status_code == HTTP_OK, "Expected 200 on /libraries with error")
    ai = response.json()["AI"]
    check(ai["keras"] is None, "keras should be None on unexpected import error")
