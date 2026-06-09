"""Constant full-frame CorridorKey hint contract.

CorridorKey is currently steered only by a scalar hint strength that is applied
to the whole frame. Older feature-derived hint variants were intentionally
removed from the production code path because they made results harder to
reason about and did not generalize across samples.
"""

from __future__ import annotations

from typing import Final

CORRIDORKEY_DEFAULT_HINT_VALUE: Final[float] = 0.32
CORRIDORKEY_HINT_STRENGTHS: Final[tuple[float, ...]] = (0.0, 0.16, 0.32, 0.5, 0.7)


def corridorkey_full_frame_prior_value(
    *,
    execution_profile: str,
    screen_mode: str,
) -> tuple[float, str]:
    """Return the default full-frame CorridorKey soft-prior hint value."""

    _ = (execution_profile, screen_mode)
    return CORRIDORKEY_DEFAULT_HINT_VALUE, "soft_prior"


def corridorkey_hint_strengths() -> tuple[float, ...]:
    """Return the exposed constant hint strengths in UI candidate order."""

    return CORRIDORKEY_HINT_STRENGTHS


__all__ = [
    "CORRIDORKEY_DEFAULT_HINT_VALUE",
    "CORRIDORKEY_HINT_STRENGTHS",
    "corridorkey_full_frame_prior_value",
    "corridorkey_hint_strengths",
]
