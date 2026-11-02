from __future__ import annotations

import json
import threading
import time
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from plugins.jentera import signin_step_in


class Runner:
    status = 200
    body = {"ok": True, "stepped": True}
    calls = []


@pytest.fixture()
def runner(monkeypatch):
    Runner.status, Runner.body, Runner.calls = 200, {"ok": True, "stepped": True}, []
    monkeypatch.setenv("JENTERA_RUNNER_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("API_SERVER_KEY", "test-key")

    def open_request(request, timeout):
        Runner.calls.append((request.full_url, request.get_header("Authorization"), json.loads(request.data)))
        response = BytesIO(Runner.body if isinstance(Runner.body, bytes) else json.dumps(Runner.body).encode())
        response.status = Runner.status
        return response

    monkeypatch.setattr(signin_step_in._OPENER, "open", open_request)
    return Runner


def test_posts_scan_id_with_runner_auth(runner):
    assert signin_step_in.request_step_in("scan-1") is True
    assert runner.calls == [("http://127.0.0.1:8080/v1/signin/step-in", "Bearer test-key", {"scanId": "scan-1"})]


@pytest.mark.parametrize("status,body", [(200, {"stepped": False}), (200, {"stepped": 1}),
    (200, {"stepped": "true"}), (200, []), (200, b"not json"), (200, {}),
    (202, {"stepped": True}), (404, {"stepped": True}), (500, {"stepped": True}),
    (200, b" " * 64001)])
def test_only_200_with_literal_true_is_accepted(runner, status, body):
    runner.status, runner.body = status, body
    assert signin_step_in.request_step_in("scan-1") is False


def test_missing_key_or_scan_id_does_not_request(runner, monkeypatch):
    assert signin_step_in.request_step_in("") is False
    monkeypatch.delenv("API_SERVER_KEY")
    assert signin_step_in.request_step_in("scan-1") is False
    assert runner.calls == []


def test_slow_request_is_bounded(runner, monkeypatch):
    done = threading.Event()
    def slow(*args, **kwargs):
        done.wait(5)
        raise OSError("unreachable")
    monkeypatch.setattr(signin_step_in._OPENER, "open", slow)
    started = time.monotonic()
    try:
        assert signin_step_in.request_step_in("scan-1", timeout=0.05) is False
        assert time.monotonic() - started < 2.0
    finally:
        done.set()


@pytest.mark.parametrize("error", [OSError("unreachable"), TimeoutError(), URLError("offline"),
    HTTPError("http://127.0.0.1:8080", 404, "off", {}, None)])
def test_transport_failure_is_refused(runner, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(signin_step_in._OPENER, "open", fail)
    assert signin_step_in.request_step_in("scan-1") is False
