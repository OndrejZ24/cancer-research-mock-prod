"""Build the analysis datasets from the raw MSK-CHORD export (same as `chordsurv build`).

Usage:  python scripts/build_datasets.py          (writes data/processed/*.parquet)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chordsurv.pipeline import build  # noqa: E402

if __name__ == "__main__":
    build()
