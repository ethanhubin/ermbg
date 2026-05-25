"""Run a no-mask vs subject-mask ERMBG regression comparison.

This is aimed at sample-12-like failures: pale subject-owned regions on a
white background where matting recall fails, but a prompt-aware ownership mask
can authorize a conservative alpha repair.

Usage:
    .venv/bin/python scripts/04_subject_mask_regression.py \
        --input samples/inputs/12.png \
        --subject-mask samples/outputs/clipseg_12/clipseg_3.png \
        --out samples/outputs/sample12_subject_regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg import matte_image


def _pick(report: dict[str, Any]) -> dict[str, Any]:
    qa = report.get("qa", {})
    halo = qa.get("edge_halo_score_per_bg", {})
    repair = report.get("keyer", {}).get("subject_repair", {})
    return {
        "strategy": report.get("strategy", {}).get("name"),
        "recomp": qa.get("recomposition_error_on_observed_bg"),
        "black_halo": halo.get("black"),
        "halo_mean": qa.get("edge_halo_score_mean"),
        "accepted_components": repair.get("accepted_components", 0),
        "accepted_pixels": repair.get("accepted_pixels", 0),
        "rejected_components": repair.get("rejected_components", 0),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Input image")
    p.add_argument("--subject-mask", type=Path, required=True, help="Ownership mask image")
    p.add_argument("--out", type=Path, required=True, help="Output root")
    p.add_argument("--backend", default="auto", help="auto | birefnet | grabcut")
    p.add_argument("--matting-model", default="ZhengPeng7/BiRefNet-matting")
    p.add_argument(
        "--assert-improved",
        action="store_true",
        help="Exit non-zero unless subject-mask improves recomp and black halo.",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    nomask_dir = args.out / "nomask"
    subject_dir = args.out / f"subject_{args.subject_mask.stem}"

    nomask = matte_image(
        args.input,
        output_dir=nomask_dir,
        qa=True,
        backend=args.backend,
        matting_model=args.matting_model,
    )
    subject = matte_image(
        args.input,
        subject_mask=args.subject_mask,
        output_dir=subject_dir,
        qa=True,
        backend=args.backend,
        matting_model=args.matting_model,
    )

    summary = {
        "input": str(args.input),
        "subject_mask": str(args.subject_mask),
        "nomask": _pick(nomask.report),
        "subject_masked": _pick(subject.report),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f'{"variant":<16} {"strategy":<14} {"recomp":<10} {"black_halo":<11} {"halo_mean":<10} {"accepted_px"}')
    for name, row in (("nomask", summary["nomask"]), ("subject_mask", summary["subject_masked"])):
        print(
            f'{name:<16} {str(row["strategy"]):<14} {_fmt(row["recomp"]):<10} '
            f'{_fmt(row["black_halo"]):<11} {_fmt(row["halo_mean"]):<10} {row["accepted_pixels"]}'
        )

    if args.assert_improved:
        n = summary["nomask"]
        s = summary["subject_masked"]
        ok = (
            s["accepted_pixels"] > 0
            and s["recomp"] is not None
            and n["recomp"] is not None
            and s["black_halo"] is not None
            and n["black_halo"] is not None
            and s["recomp"] < n["recomp"]
            and s["black_halo"] <= n["black_halo"]
        )
        if not ok:
            raise SystemExit("subject-mask regression failed acceptance check")


if __name__ == "__main__":
    main()
