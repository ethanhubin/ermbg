"""Run the local ownership planner over game-eval matte outputs.

This is the exploration entry point for the local ownership branch. It runs
matting, extracts local evidence regions, scores deterministic ownership
hypotheses, and writes a self-contained report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg import io
from ermbg.matting import matte as run_matte
from ermbg.ownership import rank_regions_ownership, resolve_execution_masks
from ermbg.segmenter import build_segmenter
from ermbg.vlm_semantic import MattingSemanticPrior

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json"


def _load_debug_region_module():
    path = PROJECT_ROOT / "scripts" / "07_vlm_planner_debug.py"
    spec = importlib.util.spec_from_file_location("ermbg_vlm_planner_debug", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEBUG_REGIONS = _load_debug_region_module()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_cases(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a cases list")
    return [case for case in cases if isinstance(case, dict)]


def _case_input_screen(case: dict[str, Any]) -> str:
    input_value = str(case.get("input", ""))
    for screen in ("green", "blue", "white"):
        if isinstance(case.get(screen), str) and str(case[screen]) == input_value:
            return screen
    screen = case.get("screen")
    if isinstance(screen, str) and screen:
        return screen
    path_screen = Path(input_value).stem
    return path_screen if path_screen else "input"


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save_matte_outputs(
    *,
    image_srgb: np.ndarray,
    input_path: Path,
    segmenter: Any,
    out_dir: Path,
    semantic_prior: Any | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    result = run_matte(image_srgb, segmenter=segmenter, semantic_prior=semantic_prior)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.save_rgb(out_dir / "input.png", image_srgb)
    io.save_rgba(out_dir / "rgba.png", result.rgba)
    io.save_mask(out_dir / "alpha.png", result.alpha)
    io.save_mask(out_dir / "shadow.png", result.debug["shadow_alpha"])
    io.save_mask(out_dir / "shadow_physical.png", result.debug["shadow_alpha_physical"])
    io.save_rgb(out_dir / "foreground.png", result.foreground_srgb)
    report = {
        "input": _rel(input_path),
        "out_dir": _rel(out_dir),
        "rgba": _rel(out_dir / "rgba.png"),
        "alpha": _rel(out_dir / "alpha.png"),
        "shadow": _rel(out_dir / "shadow.png"),
        "shadow_physical": _rel(out_dir / "shadow_physical.png"),
        "foreground": _rel(out_dir / "foreground.png"),
        "background_color": list(result.background_color),
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis is not None else None,
        "keyer": result.debug.get("keyer", {}),
        "shadow_info": result.debug.get("shadow", {}),
        "strategy": result.debug.get("strategy", {}),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report, result.rgba


def _selected_role_masks(
    ownership: list[dict[str, Any]],
    regions: list[Any],
    shape: tuple[int, int],
    *,
    confidence_min: float = 0.45,
) -> dict[str, np.ndarray]:
    masks = {
        "hole": np.zeros(shape, dtype=bool),
        "opaque_subject": np.zeros(shape, dtype=bool),
        "subject_soft_layer": np.zeros(shape, dtype=bool),
        "shadow_like_layer": np.zeros(shape, dtype=bool),
        "conservative_unknown": np.zeros(shape, dtype=bool),
    }
    region_by_id = {str(region.id): region for region in regions}
    for item in ownership:
        selected = item.get("selected") if isinstance(item, dict) else None
        region_meta = item.get("region") if isinstance(item, dict) else None
        if not isinstance(selected, dict) or not isinstance(region_meta, dict):
            continue
        role = selected.get("role")
        if role not in masks:
            continue
        try:
            confidence = float(selected.get("confidence", selected.get("score", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < confidence_min:
            continue
        region_id = str(region_meta.get("id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            continue
        masks[role] |= np.asarray(region.mask, dtype=bool)
    return masks


def _expected_role(case: dict[str, Any]) -> str | None:
    sample_id = str(case.get("sample_id", ""))
    if sample_id in {"B004", "B005", "B019", "B020"}:
        return "shadow_like_layer"
    if sample_id in {"B011", "B026", "B041", "B046", "I010", "I019", "C004", "C009"}:
        return "subject_soft_layer"
    return None


def _role_hit(row: dict[str, Any], expected_role: str | None) -> bool | None:
    if expected_role is None:
        return None
    selected_roles = {
        item.get("selected", {}).get("role")
        for item in row.get("ownership", [])
        if isinstance(item.get("selected"), dict)
    }
    return expected_role in selected_roles


def _write_role_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    tile_w, tile_h, label_h, cols = 260, 260, 52, 3
    sheet = Image.new("RGB", (cols * tile_w, ((len(ok_rows) + cols - 1) // cols) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(ok_rows):
        rgba = np.asarray(Image.open(PROJECT_ROOT / row["rgba"]).convert("RGBA"), dtype=np.uint8)
        tile = _checker_composite(rgba)
        pil = Image.fromarray(tile, mode="RGB")
        pil.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + label_h)
        roles = ",".join(row.get("top_roles", []))[:34]
        draw.text((x + 6, y + 6), str(row.get("sample_code", "")), fill=(0, 0, 0))
        draw.text((x + 6, y + 24), roles, fill=(0, 0, 0))
        sheet.paste(pil, (x + (tile_w - pil.width) // 2, y + label_h + (tile_h - pil.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _checker_composite(rgba: np.ndarray, cell: int = 28) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // cell + yy // cell) % 2) * 70 + 185).astype(np.uint8)
    bg = np.dstack([checker, checker, checker])
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def run(args: argparse.Namespace) -> None:
    cases = _load_cases(args.manifest)
    if args.sample_id:
        sample_ids = {item.strip() for item in args.sample_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("sample_id", "")) in sample_ids]
    cases = [case for case in cases if isinstance(case.get("input"), str)]

    out_root = args.out_dir
    matte_root = out_root / "matte"
    local_root = out_root / "local_ownership"
    matte_root.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)

    segmenter = build_segmenter(backend="auto")
    rows: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        sample_id = str(case.get("sample_id") or case_id)
        expected_role = _expected_role(case)
        screen = _case_input_screen(case)
        sample_code = f"{sample_id}-{screen[:1].upper()}"
        print(f"[{index}/{total}] {sample_code} {case_id}/{screen}", flush=True)
        input_path = PROJECT_ROOT / str(case["input"])
        image_srgb = _load_rgb(input_path)
        matte_dir = matte_root / case_id / screen
        local_dir = local_root / case_id / screen
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            matte_report, rgba = _save_matte_outputs(
                image_srgb=image_srgb,
                input_path=input_path,
                segmenter=segmenter,
                out_dir=matte_dir,
            )
            bg = tuple(int(c) for c in matte_report["background_color"])
            regions, evidence_info = DEBUG_REGIONS.extract_debug_regions(
                image_srgb,
                rgba,
                bg,
                coalesce=True,
                merge_distance_px=3,
            )
            ownership = rank_regions_ownership(image_srgb, rgba, bg, regions)
            raw_role_masks = _selected_role_masks(ownership, regions, image_srgb.shape[:2])
            role_masks = resolve_execution_masks(raw_role_masks, image_srgb.shape[:2])
            mask_dir = local_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            for role, mask in role_masks.items():
                io.save_mask(mask_dir / f"{role}.png", mask.astype(np.float32))

            protected_report: dict[str, Any] | None = None
            soft_mask = role_masks["subject_soft_layer"]
            shadow_mask = role_masks["shadow_like_layer"]
            if soft_mask.any():
                protected_prior = MattingSemanticPrior(
                    subject_material_mask=soft_mask.astype(np.float32),
                    subject_mask=soft_mask.astype(np.float32),
                    shadow_ownership_mask=shadow_mask.astype(np.float32) if shadow_mask.any() else np.zeros(image_srgb.shape[:2], dtype=np.float32),
                    source="local_ownership",
                )
                protected_report, _ = _save_matte_outputs(
                    image_srgb=image_srgb,
                    input_path=input_path,
                    segmenter=segmenter,
                    out_dir=out_root / "protected_matte" / case_id / screen,
                    semantic_prior=protected_prior,
                )
            top_roles = [
                str(item["selected"]["role"])
                for item in ownership
                if isinstance(item.get("selected"), dict)
            ]
            row = {
                "status": "ok",
                "sample_id": sample_id,
                "sample_code": sample_code,
                "case_id": case_id,
                "sample_screen": screen,
                "expected_role": expected_role,
                "rgba": matte_report["rgba"],
                "protected_rgba": protected_report["rgba"] if protected_report else None,
                "background_color": list(bg),
                "region_count": len(regions),
                "top_roles": top_roles,
                "role_counts": {role: top_roles.count(role) for role in sorted(set(top_roles))},
                "role_mask_pixels": {role: int(mask.sum()) for role, mask in role_masks.items()},
                "role_mask_paths": {
                    role: _rel(mask_dir / f"{role}.png")
                    for role in role_masks
                },
                "ownership": ownership,
                "evidence_info": evidence_info,
            }
            row["expected_role_hit"] = _role_hit(row, expected_role)
            (local_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(row)
            print(
                f"  regions={len(regions)} roles={row['role_counts']} "
                f"expected_hit={row['expected_role_hit']}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic script
            row = {
                "status": "error",
                "sample_id": sample_id,
                "sample_code": sample_code,
                "case_id": case_id,
                "sample_screen": screen,
                "expected_role": expected_role,
                "error": str(exc),
            }
            rows.append(row)
            print(f"  ERROR {exc}", flush=True)

    hit_rows = [row for row in rows if row.get("expected_role_hit") is not None]
    hit_count = sum(1 for row in hit_rows if row.get("expected_role_hit") is True)
    report = {
        "run_id": out_root.name,
        "case_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "expected_role_hit_count": hit_count,
        "expected_role_required_count": len(hit_rows),
        "rows": rows,
    }
    (local_root / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_role_sheet(rows, local_root / "role_sheet.png")
    print(
        f"Done: ok={report['ok_count']}/{len(rows)} "
        f"expected_role_hit={hit_count}/{len(hit_rows)}",
        flush=True,
    )
    print(f"Report: {_rel(local_root / 'eval_report.json')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "out" / "local_ownership_20260527")
    parser.add_argument("--sample-id", default="", help="Comma-separated sample ids, e.g. B001,I011,C004")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
