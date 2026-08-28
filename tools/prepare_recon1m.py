#!/usr/bin/env python
"""Build the two local caches required for ReCon1M SGDet evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orsgg.data.recon1m import load_recon1m_split, save_recon1m_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="ReCon1M root containing train/, val/, and test/ directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()

    for split in ("train", "test"):
        dataset = load_recon1m_split(args.data_root, split)
        output = args.output_dir / f"recon1m_{split}.pkl"
        save_recon1m_cache(dataset, output)
        print(f"{split}: {len(dataset)} images -> {output}")


if __name__ == "__main__":
    main()
