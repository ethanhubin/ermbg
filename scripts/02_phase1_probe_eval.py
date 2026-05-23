"""Phase 1 entry point.

Equivalent to `ermbg validate` but runnable directly without the package being
installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg.phase1 import run_phase1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("samples/inputs"))
    p.add_argument("--out", type=Path, default=Path("samples/outputs/phase1"))
    p.add_argument("--generators", default="synthetic")
    p.add_argument("--colors", default="white,black,cyan")
    p.add_argument("--backend", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_phase1(
        input_dir=args.input,
        out_dir=args.out,
        generator_names=[g.strip() for g in args.generators.split(",") if g.strip()],
        color_names=[c.strip() for c in args.colors.split(",") if c.strip()],
        backend=args.backend,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
