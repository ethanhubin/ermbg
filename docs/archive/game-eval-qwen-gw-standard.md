# Game Eval Qwen G/W Standard

Current Game Eval standard:

```text
vlm_eval_game_qwen_gw_vNNN_YYYYMMDD
```

Use a monotonically increasing small version (`v001`, `v002`, ...) when multiple
runs happen on the same day.

The standard runner is:

```bash
.venv/bin/python scripts/09_game_eval_qwen_batch.py \
  --out-dir out/vlm_eval_game_qwen_gw_v001_20260526
```

Contract:

- Samples: all `samples/vlm_eval_game/manifest.json` cases, G01-G09.
- Variants: both `green` and `white`; one web row per sample variant.
- Provider: remote ComfyUI Qwen through `Qwen3_VQA`.
- Region budget: Qwen sees a capped evidence set by default (`--max-regions 24`,
  `--max-region-crops 6`) so planner latency stays bounded.
- Output root:

```text
out/vlm_eval_game_qwen_gw_vNNN_YYYYMMDD/
  matte/<case_id>/<green|white>/
  vlm_qwen/<case_id>/<green|white>/
  vlm_qwen/eval_report.json
  vlm_qwen/eval_report.md
  vlm_qwen/selected_candidates_checker_sheet.png
```

The Web Game Eval page only discovers standard batches matching:

```text
out/vlm_eval_game_qwen_gw_v*
```

Open:

```text
/eval/game
```

or select a specific run:

```text
/eval/game?run=vlm_eval_game_qwen_gw_v001_20260526
```
