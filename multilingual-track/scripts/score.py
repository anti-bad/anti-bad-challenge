#!/usr/bin/env python3
"""Score multilingual-classification predictions."""

from __future__ import annotations

import sys
from pathlib import Path

CLASSIFICATION_SCRIPTS = Path(__file__).resolve().parents[2] / "classification-track" / "scripts"
sys.path.insert(0, str(CLASSIFICATION_SCRIPTS))

from score import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
