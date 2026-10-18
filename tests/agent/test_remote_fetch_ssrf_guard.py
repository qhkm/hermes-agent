"""Remote-party-supplied URL fetches refuse internal targets before a socket opens.

Provider response URLs and model-supplied image references go through
``agent.provider_media.open_remote_stream`` (SSRF pre-flight per hop plus a
connect-time guard). A loopback listener records hits: the invariant is that it
sees none for a hostile target. Ported from upstream 3933fdf63b / PR #114887.
"""

import http.server
import threading

import pytest

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63f8cfc00000000201010018dd8db00000000049454e44ae426082"
)
METADATA = "http://169.254.169.254/latest/meta-data/"


class _Handler(http.server.BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, *_a):
        pass

    def do_GET(self):
        type(self).hits.append((self.path, self.headers.get("Authorization")))
        if self.path == "/to-metadata":
            self.send_response(302)
            self.send_header("Location", METADATA)
            self.end_headers()
        elif self.path == "/no-location":
            self.send_response(302)
            self.end_headers()
        elif self.path == "/start":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG)))
            self.end_headers()
            self.wfile.write(PNG)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    _Handler.hits = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def allow_private(monkeypatch):
    from tools import url_safety
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "1")
    url_safety._reset_allow_private_cache()
    yield
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()


@pytest.fixture(autouse=True)
def _reset_private_cache():
    from tools import url_safety
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def test_image_save_refuses_private_target_without_connecting(server):
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="SSRF"):
        save_url_image(f"{server}/image.png")
    assert _Handler.hits == []


def test_video_save_refuses_private_target_without_connecting(server):
    from agent.video_gen_provider import save_url_video

    with pytest.raises(ValueError, match="SSRF"):
        save_url_video(f"{server}/clip.mp4")
    assert _Handler.hits == []


def test_metadata_address_refused_even_with_private_urls_allowed(allow_private):
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="SSRF"):
        save_url_image(METADATA)


def test_redirect_to_metadata_is_refused(server, allow_private):
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="SSRF"):
        save_url_image(f"{server}/to-metadata")
    assert [p for p, _ in _Handler.hits] == ["/to-metadata"]


def test_redirect_without_location_fails_closed(server, allow_private):
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="Location"):
        save_url_image(f"{server}/no-location")


def test_openai_reference_image_refuses_private_target(server):
    from plugins.image_gen.openai import _load_image_bytes

    with pytest.raises(ValueError, match="SSRF"):
        _load_image_bytes(f"{server}/image.png")
    assert _Handler.hits == []


def test_headers_go_to_first_hop_only(server, allow_private):
    from agent.provider_media import open_remote_stream

    with open_remote_stream(f"{server}/start", timeout=5, headers={"Authorization": "Bearer k"}) as resp:
        assert resp.read() == PNG
    assert _Handler.hits == [("/start", "Bearer k"), ("/final", None)]
