from __future__ import annotations

import argparse
from typing import Any, Sequence

from . import cli


def _serve_with_quality_layout(args: argparse.Namespace) -> dict[str, Any]:
    from .web_quality_layout import run_server

    run_server(args.host, args.port, open_browser=not args.no_open)
    return {"status": "stopped"}


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate CLI commands while routing `serve` through the current Web UI."""
    original_serve = cli._serve
    cli._serve = _serve_with_quality_layout
    try:
        return cli.main(argv)
    finally:
        cli._serve = original_serve


if __name__ == "__main__":
    raise SystemExit(main())
