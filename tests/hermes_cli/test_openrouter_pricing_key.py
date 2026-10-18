"""get_pricing_for_provider("openrouter") must not send OPENROUTER_API_KEY to openrouter.ai when
the configured OpenRouter base URL is somewhere else (a Jentera runtime's own proxy)."""

import io
import json

import pytest

from hermes_cli import models


@pytest.fixture
def captured(monkeypatch):
    requests = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(req, timeout=None):
        requests.append(req)
        return _Resp(json.dumps({"data": [{"id": "m/x", "pricing": {"prompt": "1", "completion": "2"}}]}).encode())

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fake_open)
    monkeypatch.setattr(models, "_pricing_cache", {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-runtime-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    return requests


def _auth(req):
    return req.get_header("Authorization")


def test_key_is_sent_to_openrouter_when_base_is_openrouter(captured, monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    assert models.get_pricing_for_provider("openrouter", force_refresh=True)["m/x"]["prompt"] == "1"
    assert captured[0].full_url.startswith("https://openrouter.ai/")
    assert _auth(captured[0]) == "Bearer sk-or-v1-runtime-key"


def test_key_is_sent_when_nothing_is_configured(captured):
    models.get_pricing_for_provider("openrouter", force_refresh=True)
    assert _auth(captured[0]) == "Bearer sk-or-v1-runtime-key"


def test_key_is_not_sent_to_openrouter_when_env_base_is_elsewhere(captured, monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://api.jentera.ai/v1/model")
    models.get_pricing_for_provider("openrouter", force_refresh=True)
    assert captured[0].full_url.startswith("https://openrouter.ai/")
    assert _auth(captured[0]) is None


def test_key_is_not_sent_when_config_base_is_elsewhere(captured, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openrouter", "base_url": "https://api.jentera.ai/v1/model"}},
    )
    models.get_pricing_for_provider("openrouter", force_refresh=True)
    assert _auth(captured[0]) is None


def test_lookalike_host_is_not_openrouter(captured, monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai.evil.example/v1")
    models.get_pricing_for_provider("openrouter", force_refresh=True)
    assert _auth(captured[0]) is None
