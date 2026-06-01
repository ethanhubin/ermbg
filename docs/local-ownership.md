# Local Ownership

Local ownership is the deterministic evidence layer used by known-background
matting. It decides which operation owns a region before execution masks are
created.

```text
known-background image
  -> local matte
  -> evidence regions
  -> ownership scoring
  -> execution-mask arbitration
```

## Roles

| Role | Meaning | Execution intent |
|---|---|---|
| `hole` | background / transparent opening | keep alpha low |
| `opaque_subject` | hard subject support was missed | allow guarded alpha repair |
| `subject_soft_layer` | glass, glow, smoke, antialiasing, translucent material | preserve soft alpha |
| `shadow_like_layer` | scalar darkening of known background near subject support | preserve measured shadow |
| `conservative_unknown` | ambiguous local evidence | keep base alpha |

## Signals

Ownership uses measurable local signals:

- alpha distribution;
- OKLab distance from known background;
- local saturation and chroma shift;
- scalar darkening fit, `C_linear ~= scale * B_linear`;
- topology: subject support, border touch, exterior fraction;
- existing risk/debug region evidence.

Rules must stay feature-based. Do not encode sample IDs, filenames, or
one-off coordinates.

## Arbitration

Region scoring is permissive; execution masks are stricter:

- tiny soft-layer speckles are dropped;
- coherent soft material suppresses small shadow-like fragments;
- shadow-only regions keep the base matte path;
- protected soft-layer rerender is used only when a coherent
  `subject_soft_layer` exists.

This separates translucent subject material from measured shadows on the same
known background.

## Code

- `ermbg/ownership.py`: signal measurement, role ranking, execution-mask
  arbitration.
- `ermbg/matting.py`: `subject_material_mask` execution constraint.
- `ermbg/web.py`: browses local ownership and Game Eval batches.

## Tests

```powershell
.\.venv\Scripts\pytest.exe tests\test_ownership.py tests\test_shadow.py tests\test_risk.py
```

Artifact-producing ownership probes should write to a batch directory under
`out/` with a summary JSON.
