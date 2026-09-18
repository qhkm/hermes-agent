"""A JavaScript page that has not finished painting reads as empty.

Firecrawl returns success with almost no markdown when a single-page app
is still rendering: developers.bukku.my came back as 2 words on the
default call and 574 with an eight-second wait, measured 2026-09-17. The
agent cannot tell that apart from a genuinely short page, so it answers
from nothing. One retry with wait_for recovers it, and only for the pages
that need it — a global wait would tax every read to fix a handful.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from plugins.web.firecrawl import provider as fc


class _Client:
    """Records every scrape call so the test can assert on the retry."""

    def __init__(self, pages: List[str]) -> None:
        self._pages = pages
        self.calls: List[Dict[str, Any]] = []

    def scrape(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        markdown = self._pages[min(len(self.calls) - 1, len(self._pages) - 1)]
        return {"markdown": markdown, "metadata": {"title": "Docs", "sourceURL": kwargs["url"]}}


@pytest.fixture(autouse=True)
def _allow_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fc, "check_website_access", lambda url: None)
    monkeypatch.setattr(fc, "is_safe_url", lambda url: True)
    # Keep the direct-SDK path: the keyless ring never reaches ``_get_firecrawl_client``.
    monkeypatch.setattr(fc, "_use_keyless_ring", lambda: False)


def _extract(url: str) -> List[Dict[str, Any]]:
    provider = fc.FirecrawlWebSearchProvider()
    return asyncio.run(provider.extract([url], format="markdown"))


def test_retries_with_a_wait_when_the_page_comes_back_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(["Docs", " ".join(["word"] * 574)])
    monkeypatch.setattr(fc, "_get_firecrawl_client", lambda: client)

    results = _extract("https://developers.bukku.my/")

    assert len(client.calls) == 2, "a thin result must be retried once"
    assert client.calls[0].get("wait_for") is None, "the first attempt stays fast"
    assert client.calls[1].get("wait_for") == fc.THIN_RENDER_WAIT_MS
    assert len(results[0]["content"].split()) == 574, "the fuller result is kept"


def test_does_not_retry_a_page_that_already_has_content(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client([" ".join(["word"] * 300)])
    monkeypatch.setattr(fc, "_get_firecrawl_client", lambda: client)

    _extract("https://example.com/article")

    assert len(client.calls) == 1, "a page with content must cost only one scrape"


def test_keeps_the_first_result_when_the_retry_is_no_better(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely short page pays one extra scrape and loses nothing."""
    client = _Client(["a list of links", ""])
    monkeypatch.setattr(fc, "_get_firecrawl_client", lambda: client)

    results = _extract("https://example.com/tags")

    assert len(client.calls) == 2
    assert results[0]["content"] == "a list of links"
