"""Jentera integration plugin — bundled, auto-loaded.

Registers one tool, ``business_records``, which reads the business's own
records through Jentera's control plane. The credential for the connected
system stays there; this machine holds nothing that opens the owner's
books.
"""

from __future__ import annotations

from plugins.jentera.tools import (
    BUSINESS_RECORDS_SCHEMA,
    check_available,
    handle_business_records,
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
