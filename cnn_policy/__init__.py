"""TurboPi no-text CNN policy package."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_DATA_ROOT = str(REPO_ROOT / "data" / "cnn_square_loop")
LEGACY_DATA_ROOT = "/workspace/data/cnn_square_loop"

DEFAULT_FRAME_HISTORY = 3
DEFAULT_IMAGE_WIDTH = 160
DEFAULT_IMAGE_HEIGHT = 120

