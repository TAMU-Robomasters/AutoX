"""Unit tests for the auto-aim engagement gate (plan 09, Phase 5).

The gate is fail-OPEN: auto-aim fires unless something fresh and definitive holds
it. These pin that policy down (missing/stale directives, fresh holds, the opt-in
match interlock).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.engage_gate import engagement_allowed  # noqa: E402
from src.types.sentry import EngageDirective  # noqa: E402

NOW = 1000.0
STALE = 0.5


def test_no_directive_fails_open():
    """No directive at all -> engage (autonomous)."""
    assert engagement_allowed(None, NOW, STALE, None, False) is True


def test_fresh_hold_blocks():
    """A fresh engage=False holds fire."""
    d = EngageDirective(engage=False, stamp=NOW - 0.1)
    assert engagement_allowed(d, NOW, STALE, None, False) is False


def test_stale_hold_is_ignored_fail_open():
    """A hold directive older than the staleness window is ignored (fail open)."""
    d = EngageDirective(engage=False, stamp=NOW - 1.0)  # > STALE old
    assert engagement_allowed(d, NOW, STALE, None, False) is True


def test_fresh_engage_true_allows():
    """A fresh engage=True clears fire."""
    d = EngageDirective(engage=True, stamp=NOW - 0.1)
    assert engagement_allowed(d, NOW, STALE, None, False) is True


def test_match_interlock_holds_only_when_inactive():
    """With the interlock on, only a definitively inactive match holds fire."""
    assert engagement_allowed(None, NOW, STALE, False, True) is False  # inactive
    assert engagement_allowed(None, NOW, STALE, True, True) is True  # active
    assert engagement_allowed(None, NOW, STALE, None, True) is True  # unknown -> open


def test_match_interlock_off_ignores_match_state():
    """With the interlock off, match state is irrelevant (fail open)."""
    assert engagement_allowed(None, NOW, STALE, False, False) is True
