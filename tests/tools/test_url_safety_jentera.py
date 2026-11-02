"""Jentera: the connect-time SSRF guard must not block what a sprite needs.

* the model endpoint (https://api.jentera.ai/v1/model) and ordinary public sites,
* the runner on loopback, which the jentera plugin reaches with urllib (not the
  guarded httpx client), so the guard cannot affect it.
"""

import http.server
import json
import threading
from pathlib import Path

import pytest

from tools import url_safety


def _public(monkeypatch, host_ip="104.18.2.2"):
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    monkeypatch.setattr(
        url_safety, "_getaddrinfo",
        lambda hostname, port=None: [(2, 1, 6, "", (
            hostname if hostname[0].isdigit() else host_ip, port or 443))],
    )


@pytest.mark.parametrize("url", [
    "https://api.jentera.ai/v1/model",
    "https://example.com/some/page",
    "https://www.skills.sh/sitemap.xml",
])
def test_public_urls_pass_preflight_and_connect(monkeypatch, url):
    _public(monkeypatch)
    assert url_safety.is_safe_url(url) is True
    host = url.split("/")[2]
    assert url_safety._resolved_http_connect_ips(host, 443, "https") == ["104.18.2.2"]


def test_metadata_and_internal_still_blocked(monkeypatch):
    _public(monkeypatch)
    assert url_safety.is_safe_url("http://169.254.169.254/latest/meta-data/") is False
    assert url_safety.is_safe_url("http://metadata.google.internal/") is False
    assert url_safety.is_safe_url("http://127.0.0.1:8080/v1/handoff") is False
    monkeypatch.setattr(
        url_safety, "_getaddrinfo",
        lambda hostname, port=None: [(2, 1, 6, "", ("169.254.169.254", port or 80))],
    )
    with pytest.raises(url_safety.SSRFConnectionBlocked):
        url_safety._resolved_http_connect_ips("rebind.example.com", 80, "http")


def test_jentera_plugin_does_not_use_the_guarded_client():
    root = Path(__file__).resolve().parents[2] / "plugins" / "jentera"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "url_safety" not in text, path
        assert "create_ssrf_safe" not in text, path


def test_plugin_reaches_runner_on_loopback(monkeypatch):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            seen.append(self.path)
            body = json.dumps({"available": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("JENTERA_RUNNER_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("API_SERVER_KEY", "k")
        from plugins.jentera import handoff
        assert handoff.check_handoff_available() is True
        assert seen
    finally:
        server.shutdown()

@pytest.mark.parametrize("answer, expected", [("104.18.2.2", True), ("127.0.0.1", False), ("169.254.169.254", False), (None, False)])
def test_dns_failure_retries_once_then_checks_every_address(monkeypatch, answer, expected):
    import socket
    _public(monkeypatch)
    monkeypatch.setattr(url_safety, "_proxy_dns_delegation_enabled", lambda: False)
    calls, waits = [], []
    def lookup(*args):
        calls.append(args)
        if len(calls) == 1 or answer is None:
            raise socket.gaierror("temporary DNS failure")
        return [(2, 1, 6, "", (answer, 443))]
    monkeypatch.setattr(url_safety, "_getaddrinfo", lookup)
    monkeypatch.setattr("time.sleep", waits.append)
    assert url_safety.is_safe_url("https://mytax.hasil.gov.my") is expected
    assert len(calls) == 2
    assert waits == [0.3]

def test_dns_reason_reaches_browser_and_cdp(monkeypatch):
    import socket
    from tools import browser_tool, browser_cdp_tool
    _public(monkeypatch)
    monkeypatch.setattr(url_safety, "_proxy_dns_delegation_enabled", lambda: False)
    monkeypatch.setattr(url_safety, "_getaddrinfo", lambda *a: (_ for _ in ()).throw(socket.gaierror("offline")))
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task, url: task)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda _: True)
    url = "https://mytax.hasil.gov.my"
    expected = "Could not look up the site's address; try again"
    assert json.loads(browser_tool.browser_navigate(url))["error"] == expected
    result = browser_cdp_tool._browser_cdp_private_guard(method="Page.navigate", params={"url": url}, task_id="t")
    assert json.loads(result)["error"] == expected
    assert json.loads(browser_cdp_tool._private_page_guard_error(url, "DOM.getDocument"))["error"] == expected
    # The reason cannot leak to another URL or a later private-address refusal.
    assert url_safety.url_safety_error("https://other.example", "blocked") == "blocked"
    monkeypatch.setattr(url_safety, "_getaddrinfo", lambda *a: [(2, 1, 6, "", ("127.0.0.1", 443))])
    assert not url_safety.is_safe_url(url)
    assert url_safety.url_safety_error(url, "blocked") == "blocked"

def test_dns_diagnostic_is_scoped_to_last_refused_url(monkeypatch):
    import socket
    _public(monkeypatch)
    monkeypatch.setattr(url_safety, "_proxy_dns_delegation_enabled", lambda: False)
    monkeypatch.setattr(url_safety, "_getaddrinfo", lambda *a: (_ for _ in ()).throw(socket.gaierror("offline")))
    monkeypatch.setattr("time.sleep", lambda _: None)
    url = "https://mytax.hasil.gov.my"
    assert not url_safety.is_safe_url(url)
    assert url_safety.url_safety_error(url, "blocked") == "Could not look up the site's address; try again"
    assert url_safety.url_safety_error("https://other.example", "blocked") == "blocked"
    assert url_safety.is_always_blocked_url("http://169.254.169.254")
    assert url_safety.url_safety_error(url, "blocked") == "blocked"
    monkeypatch.setattr(url_safety, "_getaddrinfo", lambda *a: [(2, 1, 6, "", ("127.0.0.1", 443))])
    assert not url_safety.is_safe_url(url)
    assert url_safety.url_safety_error(url, "blocked") == "blocked"
