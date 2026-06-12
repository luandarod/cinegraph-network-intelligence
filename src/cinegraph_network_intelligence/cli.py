from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed and analytical outputs for CineGraph Network Intelligence.")
    parser.add_argument("raw_dir", nargs="?", default="data/raw", help="Directory containing raw CineGraph CSV files.")
    parser.add_argument("output_dir", nargs="?", default="data", help="Directory where analytical outputs will be written.")
    args = parser.parse_args()
    run_pipeline(Path(args.raw_dir), Path(args.output_dir))
