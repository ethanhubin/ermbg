# CorridorKey Full Test Samples v1

This directory is the canonical full test sample set for CorridorKey / ERMBG after phase 1 sample construction.

Phase 1 status: complete. The approved set contains 83 samples:

- Button: 54 cases, including outlined/unoutlined/translucent shadow matrices, white-outline buttons, and real glass buttons.
- Icon/effect: 20 cases, including hard boundaries, soft boundaries, translucent icons, particles, and smooth glow.
- Character: 9 cases at 1024x1024, each intentionally combining hair/fur detail, hard opaque edges, translucent material, and/or glow.

Background convention:

- Green screen: RGB(0, 200, 0)
- Blue screen: RGB(0, 0, 200)

Use `manifest.json` as the machine-readable entry point. Each case has exactly the screen variant that was approved for it, exposed through either the `green` or `blue` key. Test runners should count only variants present in a case instead of multiplying every case by every possible background.

Next phase: run full recognition/matting evaluation over this confirmed set, inspect failures by family, then tune CorridorKey route selection and per-route parameters.
