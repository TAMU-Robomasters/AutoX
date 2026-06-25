"""Tests for the circlet cross-engine plumbing + geometry (no detector/mrcal/camera).

Covers the pieces that don't need the classical detector (and thus mrcal) or a
real camera: the orchestrator's inter-engine queue wiring, the Engine pub/sub
helpers, CircletDetections pickle round-trip (it crosses a multiprocessing.Queue),
and the chassis/turret geometry + icon classification in circlet_support.
The full CircletEngine process is exercised on an mrcal-equipped machine via
docs/camera-driver-testing.md (@CIRCLET @MOCK_CAM @SENTRY).
"""

import pickle
import sys
import time
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [
    sys.argv[0]
]  # before any config-touching src import (quik_config parses argv)

from src.core.engine import Engine  # noqa: E402
from src.core.module import Context  # noqa: E402
from src.core.orchestrator import _wire_engine_queues  # noqa: E402
from src.subsystems.circlet_support import (  # noqa: E402
    chassis_to_turret_matrix,
    circlet_panels_to_robots,
    extrinsic_from_cfg,
    panel_to_chassis,
)
from src.types.autoaim import ArmorPanel  # noqa: E402
from src.types.circlet import CircletDetections, CircletPanel  # noqa: E402


class _Engine(Engine):
    """Minimal concrete engine (no modules) to exercise the pub/sub helpers."""

    def initialize(self):
        pass

    def execute(self):
        pass


def _make_engine():
    return _Engine(modules=[], context_type=Context)


# --------------------------------------------------------------------------
# Orchestrator queue wiring
# --------------------------------------------------------------------------


def test_wire_queues_shares_one_queue_per_link():
    """A publisher + subscriber of the same name get one shared queue."""

    class Pub:
        publishes_queue = "circlet_detections"
        subscribes_queue = None

    class Sub:
        publishes_queue = None
        subscribes_queue = "circlet_detections"

    registry = _wire_engine_queues([Pub, Sub])
    assert set(registry) == {"circlet_detections"}


def test_wire_queues_duplicate_publisher_is_hard_error():
    """Two engines publishing the same link name raise at wiring time."""

    class P1:
        publishes_queue = "x"
        subscribes_queue = None

    class P2:
        publishes_queue = "x"
        subscribes_queue = None

    with pytest.raises(RuntimeError, match="published by both"):
        _wire_engine_queues([P1, P2])


def test_wire_queues_no_links_is_empty():
    """An engine declaring no queues produces an empty registry."""

    class Plain:
        pass

    assert _wire_engine_queues([Plain]) == {}


def test_wire_queues_multilink_collects_all_names():
    """A multi-link subscriber + per-name publishers get one queue per link."""

    class AutoAim:
        subscribes_queues = ["circlet_detections", "engage_directive"]

    class Circlet:
        publishes_queue = "circlet_detections"

    class AutoNav:
        publishes_queues = ["engage_directive"]

    registry = _wire_engine_queues([AutoAim, Circlet, AutoNav])
    assert set(registry) == {"circlet_detections", "engage_directive"}


def test_wire_queues_duplicate_publisher_across_single_and_multi():
    """One publisher via single-link + another via multi-link still hard-errors."""

    class P1:
        publishes_queue = "engage_directive"

    class P2:
        publishes_queues = ["engage_directive"]

    with pytest.raises(RuntimeError, match="published by both"):
        _wire_engine_queues([P1, P2])


# --------------------------------------------------------------------------
# Engine pub/sub helpers (last-value semantics)
# --------------------------------------------------------------------------


def test_publish_then_subscribe_gets_newest():
    """publish() keeps last-value semantics: the consumer reads the newest message."""
    engine = _make_engine()
    q = Queue()
    engine._publish_q = q
    engine._subscribe_q = q
    engine.publish("a")
    time.sleep(0.05)  # let the queue feeder thread flush
    engine.publish("b")
    time.sleep(0.05)
    assert engine.latest_subscribed() == "b"
    assert engine.latest_subscribed() is None  # drained


def test_subscribe_drains_to_newest():
    """latest_subscribed() drains a backlog and returns only the newest message."""
    engine = _make_engine()
    q = Queue()
    engine._subscribe_q = q
    for msg in ("a", "b", "c"):
        q.put(msg)
    time.sleep(0.1)
    assert engine.latest_subscribed() == "c"
    assert engine.latest_subscribed() is None


def test_publish_and_subscribe_are_noops_without_queues():
    """Without wired queues, publish/latest_subscribed are safe no-ops."""
    engine = _make_engine()
    engine.publish("ignored")  # no _publish_q -> no error
    assert engine.latest_subscribed() is None


def test_named_links_route_independently():
    """publish/latest_subscribed(name=...) address distinct named links."""
    engine = _make_engine()
    qa, qb = Queue(), Queue()
    engine._publish_qs = {"a": qa, "b": qb}
    engine._subscribe_qs = {"a": qa, "b": qb}

    engine.publish("to-a", "a")
    engine.publish("to-b", "b")
    time.sleep(0.05)  # let the queue feeder threads flush
    # Each name reads only its own link's newest message — no cross-talk.
    assert engine.latest_subscribed("a") == "to-a"
    assert engine.latest_subscribed("b") == "to-b"
    assert engine.latest_subscribed("a") is None  # drained
    assert engine.latest_subscribed("missing") is None  # unknown name -> no-op


def test_single_multilink_infers_without_name():
    """With exactly one multi-link and no name, publish/subscribe use it."""
    engine = _make_engine()
    q = Queue()
    engine._publish_qs = {"only": q}
    engine._subscribe_qs = {"only": q}
    engine.publish("x")  # name omitted -> the sole multi link
    time.sleep(0.05)
    assert engine.latest_subscribed() == "x"


# --------------------------------------------------------------------------
# CircletDetections crosses a Queue -> must pickle cleanly
# --------------------------------------------------------------------------


def test_circlet_detections_pickle_roundtrip():
    """CircletDetections survives pickling (it crosses a multiprocessing.Queue)."""
    msg = CircletDetections(
        panels=[
            CircletPanel(
                icon=1,
                position=np.array([1.0, 2.0, 3.0]),
                yaw=0.5,
                camera_id=2,
                timestamp=9.0,
            ),
        ],
        timestamp=9.0,
    )
    back = pickle.loads(pickle.dumps(msg))
    assert back.timestamp == 9.0
    assert len(back.panels) == 1
    assert back.panels[0].icon == 1
    assert back.panels[0].camera_id == 2
    np.testing.assert_allclose(back.panels[0].position, [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------
# Geometry: camera->chassis extrinsic
# --------------------------------------------------------------------------


def test_identity_extrinsic_preserves_position():
    """An identity camera->chassis extrinsic leaves the position unchanged (cm round-trips)."""
    panel = ArmorPanel(
        icon=0,
        position=np.array([10.0, 20.0, 30.0]),
        orientation=None,
        bbx=None,
        contour=None,
    )
    cp = panel_to_chassis(panel, extrinsic_from_cfg(0.0), camera_id=0, timestamp=1.0)
    np.testing.assert_allclose(cp.position, [10.0, 20.0, 30.0], atol=1e-9)
    assert cp.camera_id == 0 and cp.icon == 0


def test_yaw_extrinsic_rotates_position():
    """A +90 deg mounting yaw maps chassis +x to +y."""
    panel = ArmorPanel(
        icon=0,
        position=np.array([10.0, 0.0, 0.0]),
        orientation=None,
        bbx=None,
        contour=None,
    )
    # +90 deg about chassis z maps +x -> +y.
    cp = panel_to_chassis(panel, extrinsic_from_cfg(90.0), camera_id=3, timestamp=1.0)
    np.testing.assert_allclose(cp.position, [0.0, 10.0, 0.0], atol=1e-6)
    assert cp.camera_id == 3


def test_translation_offsets_position():
    """A camera mounting translation offsets the chassis-frame position (cm)."""
    panel = ArmorPanel(
        icon=0,
        position=np.array([0.0, 0.0, 0.0]),
        orientation=None,
        bbx=None,
        contour=None,
    )
    cp = panel_to_chassis(
        panel,
        extrinsic_from_cfg(0.0, translation_cm=(5.0, -7.0, 2.0)),
        camera_id=0,
        timestamp=1.0,
    )
    np.testing.assert_allclose(cp.position, [5.0, -7.0, 2.0], atol=1e-6)


def test_chassis_to_turret_identity_at_zero():
    """Zero gimbal yaw/pitch gives an identity chassis->turret rotation."""
    np.testing.assert_allclose(
        chassis_to_turret_matrix(0.0, 0.0), np.eye(3), atol=1e-12
    )


# --------------------------------------------------------------------------
# Classification of circlet detections by icon
# --------------------------------------------------------------------------


def test_circlet_panels_classify_by_icon():
    """Circlet panels bucket into robots by icon (incl. icon 3 -> sentry)."""
    det = CircletDetections(
        panels=[
            CircletPanel(
                icon=1,
                position=np.array([0.0, 150.0, 0.0]),
                yaw=0.0,
                camera_id=0,
                timestamp=1.0,
            ),
            CircletPanel(
                icon=2,
                position=np.array([0.0, 400.0, 0.0]),
                yaw=0.0,
                camera_id=1,
                timestamp=1.0,
            ),
            CircletPanel(
                icon=3,
                position=np.array([0.0, 600.0, 0.0]),
                yaw=0.0,
                camera_id=2,
                timestamp=1.0,
            ),
        ]
    )
    robots = circlet_panels_to_robots(det, turret_rotation=None)
    assert set(robots) == {"hero", "standard", "sentry"}  # icons 1/2/3
    assert len(robots["hero"].panels) == 1
    # hero (150) is the closest robot.
    closest = min(
        robots.values(), key=lambda r: float(np.linalg.norm(r.panels[0].position))
    )
    assert closest.name == "hero"


def test_circlet_classification_applies_turret_rotation():
    """The gimbal-pose rotation is applied when lifting circlet panels to the turret frame."""
    det = CircletDetections(
        panels=[
            CircletPanel(
                icon=1,
                position=np.array([10.0, 0.0, 0.0]),
                yaw=0.0,
                camera_id=0,
                timestamp=1.0,
            ),
        ]
    )
    # 90 deg gimbal yaw: chassis +x -> turret +y (Rz(-yaw) applied to the point).
    rotation = chassis_to_turret_matrix(np.pi / 2, 0.0)
    robots = circlet_panels_to_robots(det, turret_rotation=rotation)
    np.testing.assert_allclose(
        robots["hero"].panels[0].position, [0.0, -10.0, 0.0], atol=1e-6
    )
