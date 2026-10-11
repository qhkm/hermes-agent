"""Jentera integration plugin — bundled, auto-loaded.

Registers one tool, ``business_records``, which reads the business's own
records through Jentera's control plane. The credential for the connected
system stays there; this machine holds nothing that opens the owner's
books.
"""

from __future__ import annotations

from plugins.jentera.handoff import ASK_SPECIALIST_SCHEMA, check_handoff_available, handle_ask_specialist
from plugins.jentera.site import (
    SITE_APPLY_SCHEMA,
    SITE_DRAFT_SCHEMA,
    SITE_GET_SCHEMA,
    SITE_IMAGE_SCHEMA,
    handle_site_apply,
    handle_site_draft,
    handle_site_get,
    handle_site_image,
)
from plugins.jentera.tools import (
    BUSINESS_RECORDS_SCHEMA,
    CONNECT_SERVICE_SCHEMA,
    check_available,
    handle_business_records,
    handle_connect_service,
)


def register(ctx) -> None:
    """Register the Jentera tools. Called once by the plugin loader."""
    ctx.register_tool(
        name="business_records",
        toolset="jentera",
        schema=BUSINESS_RECORDS_SCHEMA,
        handler=handle_business_records,
        check_fn=check_available,
        emoji="📒",
    )
    ctx.register_tool(
        name="connect_service",
        toolset="jentera",
        schema=CONNECT_SERVICE_SCHEMA,
        handler=handle_connect_service,
        check_fn=check_available,
        emoji="🔌",
    )
    ctx.register_tool(
        name="ask_specialist",
        toolset="jentera",
        schema=ASK_SPECIALIST_SCHEMA,
        handler=handle_ask_specialist,
        check_fn=check_handoff_available,
        emoji="🤝",
    )
    for name, schema, handler, emoji in (
        ("site_draft", SITE_DRAFT_SCHEMA, handle_site_draft, "🌐"),
        ("site_get", SITE_GET_SCHEMA, handle_site_get, "🌐"),
        ("site_apply", SITE_APPLY_SCHEMA, handle_site_apply, "🌐"),
        ("site_image", SITE_IMAGE_SCHEMA, handle_site_image, "🖼️"),
    ):
        ctx.register_tool(
            name=name,
            toolset="jentera",
            schema=schema,
            handler=handler,
            check_fn=check_available,
            emoji=emoji,
        )
