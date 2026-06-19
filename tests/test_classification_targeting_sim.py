"""Sim test: classification + targeting under async, multi-camera, variable-fps input.

Position-level scene (no pixels/PnP): several ground-truth robots are seen by
several fake cameras sampling at different, runtime-variable rates. We merge the
cameras' latest detections and drive the real ``RobotClassificationModule`` +
``TargetingModule``, asserting icons bucket to the right robot and the closest
robot is selected -- including a moving case where the closest one changes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [
    sys.argv[0]
]  # before any config-touching src import (quik_config parses argv)

from src.subsystems.classification import RobotClassificationModule  # noqa: E402
from src.subsystems.targeting import TargetingModule  # noqa: E402
from src.types.autoaim import FullStateAutoAimContext  # noqa: E402
from tests.sim.scene import (  # noqa: E402
    ICON_HERO,
    ICON_SENTRY,
    ICON_SENTRY_ALT,
    ICON_STANDARD,
    FakeCameraStream,
    SimScene,
    merge_latest,
    step_streams,
)


def run_decision(panels):
    """Run classification + targeting on a panel list; return the populated context."""
    ctx = FullStateAutoAimContext()
    ctx.panels = list(panels)
    RobotClassificationModule(ctx).run()
    TargetingModule(ctx).run()
    return ctx


def test_multi_robot_classification_and_closest_target():
    """Three robots seen by three cameras: each buckets correctly, closest is targeted."""
    scene = SimScene(
        [
            SimScene.at("sentry", ICON_SENTRY, (0, 300, 0)),
            SimScene.at("hero", ICON_HERO, (0, 150, 0)),  # closest
            SimScene.at("standard", ICON_STANDARD, (0, 500, 0)),
        ]
    )
    streams = [
        FakeCameraStream(scene, fps=90, name="c0", seed=1),
        FakeCameraStream(scene, fps=15, name="c1", seed=2),
        FakeCameraStream(scene, fps=30, name="c2", seed=3),
    ]
    for s in streams:
        s.sample(0.0)

    ctx = run_decision(merge_latest(streams))

    # One panel per robot per camera => three panels per robot, each bucketed by icon.
    assert len(ctx.sentry.panels) == 3
    assert len(ctx.hero.panels) == 3
    assert len(ctx.standard.panels) == 3
    assert all(p.icon == ICON_SENTRY for p in ctx.sentry.panels)
    assert all(p.icon == ICON_HERO for p in ctx.hero.panels)
    assert all(p.icon == ICON_STANDARD for p in ctx.standard.panels)
    assert ctx.target_robot is not None
    assert ctx.target_robot.name == "hero"


def test_alt_sentry_icon_buckets_to_sentry():
    """Icon 3 maps to the sentry robot too (ICON_TO_ROBOT_NAME)."""
    scene = SimScene([SimScene.at("sentry", ICON_SENTRY_ALT, (0, 200, 0))])
    stream = FakeCameraStream(scene, fps=30, seed=7)
    stream.sample(0.0)

    ctx = run_decision(merge_latest([stream]))

    assert len(ctx.sentry.panels) == 1
    assert not ctx.hero.panels and not ctx.standard.panels
    assert ctx.target_robot is not None and ctx.target_robot.name == "sentry"


def test_no_panels_yields_no_target():
    """Empty detections => every bucket empty and no target selected."""
    ctx = run_decision([])
    assert ctx.target_robot is None
    assert not ctx.sentry.panels and not ctx.hero.panels and not ctx.standard.panels


def test_async_variable_fps_keeps_correct_target():
    """Cameras at different (incl. time-varying) fps still always target the closest."""
    scene = SimScene(
        [
            SimScene.at("hero", ICON_HERO, (0, 120, 0)),  # always closest
            SimScene.at("sentry", ICON_SENTRY, (0, 350, 0)),
            SimScene.at("standard", ICON_STANDARD, (0, 600, 0)),
        ]
    )
    streams = [
        FakeCameraStream(scene, fps=90, name="fast", seed=1, noise_cm=0.3),
        FakeCameraStream(scene, fps=15, name="slow", seed=2, phase=0.013, noise_cm=0.3),
        # runtime-variable rate: ramps from ~10 to ~40 fps over the run.
        FakeCameraStream(
            scene, fps=lambda t: 10.0 + 30.0 * t, name="varying", seed=3, noise_cm=0.3
        ),
    ]

    targets = []

    def on_tick(_t, strs):
        merged = merge_latest(strs)
        if merged:
            targets.append(run_decision(merged).target_robot.name)

    step_streams(streams, duration=1.0, on_tick=on_tick)

    assert targets, "expected at least one decision"
    # Hero is closest and present in every camera's sample, so once anything is
    # merged the closest pick is always hero -- regardless of which cameras fired.
    assert set(targets) == {"hero"}


def test_moving_target_switches_pick():
    """As robots move, the closest one (hence the target) changes mid-run."""
    duration = 1.0
    scene = SimScene(
        [
            # standard starts close, recedes; hero starts far, approaches. They
            # cross around t=0.5 -> the target must switch standard -> hero.
            SimScene.moving(
                "standard", ICON_STANDARD, (0, 150, 0), (0, 600, 0), duration
            ),
            SimScene.moving("hero", ICON_HERO, (0, 600, 0), (0, 150, 0), duration),
        ]
    )
    streams = [
        FakeCameraStream(scene, fps=90, name="a", seed=1, noise_cm=0.1),
        FakeCameraStream(scene, fps=15, name="b", seed=2, phase=0.02, noise_cm=0.1),
    ]

    trace = []  # (t, target_name)

    def on_tick(t, strs):
        merged = merge_latest(strs)
        if merged:
            trace.append((t, run_decision(merged).target_robot.name))

    step_streams(streams, duration=duration, on_tick=on_tick)

    names = [n for _t, n in trace]
    assert names, "expected decisions over the run"
    # Starts on standard (closest early), ends on hero (closest late).
    early = [n for t, n in trace if t < 0.25]
    late = [n for t, n in trace if t > 0.75]
    assert early and all(n == "standard" for n in early)
    assert late and all(n == "hero" for n in late)
    # And the switch actually happened exactly once in the sequence.
    switches = sum(1 for a, b in zip(names, names[1:]) if a != b)
    assert switches == 1


@pytest.mark.parametrize("noise_cm", [0.0, 0.5, 2.0])
def test_robust_to_measurement_noise(noise_cm):
    """Bucketing + closest-pick survive realistic position noise."""
    scene = SimScene(
        [
            SimScene.at("hero", ICON_HERO, (0, 140, 0)),
            SimScene.at("standard", ICON_STANDARD, (0, 500, 0)),
        ]
    )
    stream = FakeCameraStream(scene, fps=30, seed=11, noise_cm=noise_cm)
    stream.sample(0.0)
    ctx = run_decision(merge_latest([stream]))
    assert ctx.target_robot is not None and ctx.target_robot.name == "hero"
