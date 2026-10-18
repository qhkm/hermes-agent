"""SSRF-guarded download of media URLs that a remote party supplied.

Provider response URLs (image/video generation) and model-supplied image
references are remote-party-controlled: a hostile or compromised endpoint could
point them at internal or cloud-metadata addresses. Every hop is validated with
``tools.url_safety`` before a socket opens, and again at TCP connect by the
guarded transport (closing the DNS-rebinding window). Caller headers go to the
first hop only. Ported from upstream NousResearch/hermes-agent 3933fdf63b.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional
from urllib.parse import urljoin

REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 5


@contextmanager
def open_remote_stream(
    url: str,
    *,
    timeout: float,
    label: str = "Download",
    headers: Optional[Dict[str, str]] = None,
) -> Iterator["httpx.Response"]:  # noqa: F821 - httpx imported lazily
    """Stream *url* through the SSRF guard and yield the final successful response.

    Redirects are followed manually (at most ``MAX_REDIRECTS``), each target
    re-validated; a 3xx without ``Location`` fails closed. *headers* (e.g. a
    provider's auth) are sent to the first hop only. Raises ``ValueError`` for a
    blocked URL and ``httpx.HTTPStatusError`` for an HTTP error status.
    """
    from tools.url_safety import create_ssrf_safe_client, is_safe_url

    current_url, hop_headers = url, headers
    for _ in range(MAX_REDIRECTS + 1):
        if not is_safe_url(current_url):
            raise ValueError(f"{label} URL failed the SSRF safety check: {current_url}")
        with create_ssrf_safe_client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", current_url, headers=hop_headers) as response:
                if response.status_code in REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError(f"{label} redirected without a Location: {current_url}")
                    current_url, hop_headers = urljoin(current_url, location), None
                    continue
                if not response.is_success:
                    response.read()
                    response.raise_for_status()
                yield response
                return
    raise ValueError(f"{label} exceeded {MAX_REDIRECTS} redirects: {url}")
