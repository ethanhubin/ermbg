"""Generate README and docs gallery pages from a full game-eval batch."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = PROJECT_ROOT / "out" / "direct_worker_game_input_20260610_v002"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "eval-gallery.md"
DEFAULT_README = PROJECT_ROOT / "README.md"
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "eval-gallery"
README_START = "<!-- ERMBG_EVAL_GALLERY:START -->"
README_END = "<!-- ERMBG_EVAL_GALLERY:END -->"


@dataclass(frozen=True)
class GalleryCase:
    sample_id: str
    case: str
    category: str
    screen: str
    status: str
    algorithm: str
    execution_profile: str
    execution_backend: str
    source_input: str
    thumb_name: str
    case_manifest: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _rel_from(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _resize_contact_sheet(src: Path, dest: Path, *, width: int, quality: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    if image.width > width:
        height = round(image.height * (width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)


def _case_sample_id(case_name: str, manifest: dict[str, Any]) -> str:
    metadata = manifest.get("extra", {}).get("case_metadata", {})
    sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    prefix = case_name.split("_", 1)[0]
    return prefix if prefix else case_name


def _collect_cases(batch: Path, asset_dir: Path, *, thumb_width: int, quality: int) -> tuple[dict[str, Any], list[GalleryCase]]:
    summary_path = batch / "summary.json"
    summary = _load_json(summary_path)
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise ValueError(f"{summary_path} does not contain a runs list")

    cases: list[GalleryCase] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case") or "")
        if not case_name:
            continue
        manifest_ref = row.get("artifact_manifest")
        if not isinstance(manifest_ref, str) or not manifest_ref:
            raise ValueError(f"{case_name} is missing artifact_manifest")

        manifest_path = PROJECT_ROOT / manifest_ref
        manifest = _load_json(manifest_path)
        outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
        contact_sheet = outputs.get("contact_sheet")
        if not isinstance(contact_sheet, str):
            raise ValueError(f"{manifest_path} is missing outputs.contact_sheet")

        case_dir = manifest_path.parent
        contact_sheet_path = case_dir / contact_sheet
        if not contact_sheet_path.exists():
            raise FileNotFoundError(contact_sheet_path)

        metadata = manifest.get("extra", {}).get("case_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        route = manifest.get("route") if isinstance(manifest.get("route"), dict) else {}
        runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}

        sample_id = _case_sample_id(case_name, manifest)
        thumb_name = f"{_safe_slug(sample_id)}.jpg"
        _resize_contact_sheet(contact_sheet_path, asset_dir / thumb_name, width=thumb_width, quality=quality)

        cases.append(
            GalleryCase(
                sample_id=sample_id,
                case=case_name,
                category=str(metadata.get("category") or "unknown"),
                screen=str(metadata.get("screen") or row.get("sample_screen") or ""),
                status=str(row.get("status") or ""),
                algorithm=str(route.get("algorithm") or runtime.get("algorithm") or ""),
                execution_profile=str(route.get("execution_profile") or ""),
                execution_backend=str(runtime.get("execution_backend") or ""),
                source_input=str(manifest.get("request", {}).get("source_input") or row.get("input") or ""),
                thumb_name=thumb_name,
                case_manifest=_project_rel(manifest_path),
            )
        )

    return summary, cases


def _html_cell(case: GalleryCase, img_src: str, *, image_width: int, width_percent: int, href: str) -> str:
    title = html.escape(f"{case.sample_id} · {case.category} · {case.execution_profile}")
    alt = html.escape(f"{case.sample_id} result contact sheet")
    href = html.escape(href)
    img_src = html.escape(img_src)
    status = html.escape(case.status)
    profile = html.escape(case.execution_profile)
    algorithm = html.escape(case.algorithm)
    sample_id = html.escape(case.sample_id)
    return (
        f'<td width="{width_percent}%" valign="top">'
        f'<a href="{href}" title="{title}"><img src="{img_src}" width="{image_width}" alt="{alt}"></a><br>'
        f'<sub><b>{sample_id}</b> · {status}<br>{profile}<br>{algorithm}</sub>'
        f'</td>'
    )


def _html_table(cases: list[GalleryCase], *, image_src, href, image_width: int, columns: int) -> str:
    width_percent = max(1, 100 // columns)
    lines = ["<table>"]
    for index in range(0, len(cases), columns):
        lines.append("<tr>")
        for case in cases[index : index + columns]:
            lines.append(
                _html_cell(
                    case,
                    image_src(case),
                    image_width=image_width,
                    width_percent=width_percent,
                    href=href(case),
                )
            )
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _summary_line(summary: dict[str, Any], cases: list[GalleryCase]) -> str:
    batch = html.escape(str(summary.get("batch") or ""))
    backend = html.escape(str(summary.get("backend") or ""))
    run_count = int(summary.get("run_count") or len(cases))
    ok_count = int(summary.get("ok_count") or sum(1 for item in cases if item.status == "ok"))
    categories = Counter(case.category for case in cases)
    category_text = " / ".join(f"{name} {count}" for name, count in sorted(categories.items()))
    return f"来源 batch: `{batch}`。后端: `{backend}`。结果: `{ok_count}/{run_count}` ok。覆盖: {category_text}。"


def _write_metadata(asset_dir: Path, summary: dict[str, Any], cases: list[GalleryCase]) -> None:
    payload = {
        "source_batch": summary.get("batch"),
        "backend": summary.get("backend"),
        "run_count": summary.get("run_count"),
        "ok_count": summary.get("ok_count"),
        "cases": [case.__dict__ for case in cases],
    }
    (asset_dir / "gallery.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_doc(doc_path: Path, asset_dir: Path, summary: dict[str, Any], cases: list[GalleryCase]) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[GalleryCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)

    asset_rel = _rel_from(asset_dir, doc_path.parent)
    lines = [
        "# ERMBG 全量样本结果 Gallery",
        "",
        _summary_line(summary, cases),
        "",
        "每张缩略图是对应 case 的 contact sheet，用于快速检查 input、RGBA、alpha、trimap 等输出。原始 batch 仍保留在 `out/`，文档中只提交压缩后的展示图。",
        "",
        "## 汇总",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| Batch | `{summary.get('batch')}` |",
        f"| Backend | `{summary.get('backend')}` |",
        f"| Cases | `{summary.get('ok_count')}/{summary.get('run_count')}` ok |",
        f"| Artifact manifest | `{summary.get('artifact_manifest')}` |",
        "",
    ]

    for category in ("button", "icon", "character", "unknown"):
        category_cases = grouped.get(category)
        if not category_cases:
            continue
        lines.extend(
            [
                f"## {category.title()} ({len(category_cases)})",
                "",
                _html_table(
                    category_cases,
                    image_src=lambda item, rel=asset_rel: f"{rel}/{item.thumb_name}",
                    href=lambda item: f"#{item.sample_id.lower()}",
                    image_width=260,
                    columns=3,
                ),
                "",
            ]
        )
        for case in category_cases:
            lines.extend(
                [
                    f'<a id="{case.sample_id.lower()}"></a>',
                    "",
                    f"### {case.sample_id}",
                    "",
                    f"- Case: `{case.case}`",
                    f"- Status: `{case.status}`",
                    f"- Profile: `{case.execution_profile}`",
                    f"- Algorithm: `{case.algorithm}`",
                    f"- Execution backend: `{case.execution_backend}`",
                    f"- Input: `{case.source_input}`",
                    f"- Manifest: `{case.case_manifest}`",
                    "",
                ]
            )

    doc_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _readme_block(readme_path: Path, asset_dir: Path, doc_path: Path, summary: dict[str, Any], cases: list[GalleryCase]) -> str:
    asset_rel = _rel_from(asset_dir, readme_path.parent)
    doc_rel = _rel_from(doc_path, readme_path.parent)
    table = _html_table(
        cases,
        image_src=lambda item: f"{asset_rel}/{item.thumb_name}",
        href=lambda item: f"{doc_rel}#{item.sample_id.lower()}",
        image_width=220,
        columns=4,
    )
    return "\n".join(
        [
            README_START,
            "## 全量样本结果",
            "",
            _summary_line(summary, cases),
            "",
            f"完整展示页见 [{doc_rel}]({doc_rel})。",
            "",
            "<details>",
            f"<summary>展开 {len(cases)} 个样本缩略图</summary>",
            "",
            table,
            "",
            "</details>",
            README_END,
        ]
    )


def _update_readme(readme_path: Path, block: str) -> None:
    original = readme_path.read_text(encoding="utf-8")
    if README_START in original and README_END in original:
        pattern = re.compile(f"{re.escape(README_START)}.*?{re.escape(README_END)}", re.DOTALL)
        updated = pattern.sub(block, original)
    else:
        marker = "\n## 真实执行主线"
        if marker not in original:
            raise ValueError("README insertion marker not found")
        updated = original.replace(marker, f"\n{block}\n\n---\n{marker}", 1)
    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--thumb-width", type=int, default=520)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()

    batch = args.batch.resolve()
    asset_dir = (args.asset_root / _safe_slug(batch.name)).resolve()
    summary, cases = _collect_cases(batch, asset_dir, thumb_width=args.thumb_width, quality=args.jpeg_quality)
    _write_metadata(asset_dir, summary, cases)
    _write_doc(args.doc.resolve(), asset_dir, summary, cases)
    if not args.no_readme:
        block = _readme_block(args.readme.resolve(), asset_dir, args.doc.resolve(), summary, cases)
        _update_readme(args.readme.resolve(), block)

    print(
        json.dumps(
            {
                "batch": _project_rel(batch),
                "doc": _project_rel(args.doc.resolve()),
                "asset_dir": _project_rel(asset_dir),
                "case_count": len(cases),
                "ok_count": summary.get("ok_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
