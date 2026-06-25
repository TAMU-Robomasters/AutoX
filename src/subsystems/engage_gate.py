"""The auto-aim engagement gate (plan 09, Phase 5) — a pure, testable decision.

Auto-aim is **autonomous and fail-OPEN**: it fires on its own unless something
*fresh and definitive* says otherwise. Two things can hold fire:

* a **fresh** ``EngageDirective`` from the AutoNav brain with ``engage=False`` (a
  stale or missing directive is ignored — the sentry keeps engaging even if the
  brain dies or the link stalls);
* (opt-in) a **match interlock**: when enabled, a definitively *inactive* match
  (pre-/post-match) holds fire. An unknown match state (``None`` — the current
  firmware has no wire message yet) fails OPEN.

Kept free of engine/ROS state so the policy is unit-tested directly.
"""

from typing import Optional

from src.types.sentry import EngageDirective


def engagement_allowed(
    directive: Optional[EngageDirective],
    now: float,
    staleness_s: float,
    match_state: Optional[bool],
    match_interlock: bool,
) -> bool:
    """Return whether auto-aim is cleared to fire (fail-OPEN).

    Args:
        directive: the latest ``EngageDirective`` seen, or ``None``.
        now: current wall-clock time (``time.time()``), compared to the stamp.
        staleness_s: a directive older than this is ignored (fail open).
        match_state: ``True`` active / ``False`` inactive / ``None`` unknown.
        match_interlock: whether the match interlock is enforced at all.
    """
    if (
        directive is not None
        and (now - directive.stamp) < staleness_s
        and not directive.engage
    ):
        return False
    if match_interlock and match_state is False:
        return False
    return True
