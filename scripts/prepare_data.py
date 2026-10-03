"""Place the challenge CSV at data/raw/rasmussen_stability.csv and sanity-check it.

Usage:
    python scripts/prepare_data.py /path/to/downloaded.csv
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from hlaforge import data

TARGET = Path("data/raw/rasmussen_stability.csv")


def main() -> int:
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        if not src.exists():
            print(f"source not found: {src}")
            return 1
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, TARGET)
        print(f"copied {src} -> {TARGET}")

    df = data.load_raw(TARGET)
    print(data.profile(df).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
