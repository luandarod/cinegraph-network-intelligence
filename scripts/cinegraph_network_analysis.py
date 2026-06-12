"""CLI wrapper for the CineGraph analytical pipeline."""

from __future__ import annotations

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from cinegraph_network_intelligence.cli import main


if __name__ == "__main__":
    main()
