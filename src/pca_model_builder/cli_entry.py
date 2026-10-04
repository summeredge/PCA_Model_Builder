from __future__ import annotations

from typing import Sequence

from . import cli


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate all commands to the shared CLI implementation."""
    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
