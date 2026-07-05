from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coomfy_utils import CoomfyIntToFloat


def test_coomfy_int_to_float():
    node = CoomfyIntToFloat()
    assert node.convert(24) == (24.0,)
    assert node.convert(1) == (1.0,)
