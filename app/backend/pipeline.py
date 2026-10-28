"""The single import point for the frozen demo pipeline, so every backend module reuses exactly the same functions."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "demo") not in sys.path:
    sys.path.insert(0, str(REPO / "demo"))

import infer_partial_ecg as demo  # noqa: E402

LEAD_NAMES = list(demo.LEAD_NAMES)
LEAD_SETS = sorted((sorted(s, key=demo.LEAD_INDEX.get) for s in demo.LEAD_SETS), key=lambda s: (len(s), [demo.LEAD_INDEX[x] for x in s]))
