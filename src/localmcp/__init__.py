"""Reusable foundations for local MCP servers.

Optional capabilities are resolved lazily so importing :mod:`localmcp` alone
does not require an LLM, MCP, telemetry, or workflow stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from localmcp.server import STDIOServer as STDIOServer
    from localmcp.server import main as main

__all__ = ["STDIOServer", "main"]

__version__ = "0.1.3"


def __getattr__(name: str) -> Any:
    if name == "STDIOServer":
        from localmcp.server import STDIOServer

        return STDIOServer
    if name == "main":
        from localmcp.server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
