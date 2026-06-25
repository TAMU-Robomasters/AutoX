"""``AutoNav`` — the sentry behaviour brain driving nav2 (plan 09, Phase 4).

Folds the pure :class:`~src.subsystems.sentry_brain.SentryBrain` (the doctrine:
progress to the point; slow side-step evade on enemy contact; resume) into an
engine that drives **nav2** through ``nav2_simple_commander``'s ``BasicNavigator``
and exchanges two cross-engine messages with auto-aim:

* subscribes ``aim_target`` (:class:`~src.types.sentry.AimTarget`) — whether
  auto-aim sees an enemy; this is the brain's only live input;
* publishes ``engage_directive`` (:class:`~src.types.sentry.EngageDirective`) —
  fail-OPEN (always engage in this doctrine).

The brain decides a **chassis mode** each tick; this engine maps it to nav2 goals:
``progress`` -> ``goToPose`` the next waypoint (looped); ``evade`` -> cancel and
ping-pong small lateral ``goToPose`` goals around the spot where contact began. The
robot is holonomic and its yaw is owned by the MCU, so goal orientation is ignored
(identity) and a lateral map-frame offset is a pure side-step.

``rclpy`` + ``nav2_simple_commander`` are imported lazily in :meth:`initialize` (the
child process) so the module stays importable without ROS. Only launched when
``ros.stack == nav2``. nav2 must be up: ``initialize`` blocks on
``waitUntilNav2Active`` before the brain starts ticking.
"""

import time

from src.core.engine import Engine
from src.subsystems.sentry_brain import (
    CHASSIS_EVADE,
    CHASSIS_PROGRESS,
    BrainInputs,
    SentryBrain,
)
from src.toolbox.globals import config
from src.types.null import NullContext
from src.types.sentry import EngageDirective


class AutoNav(Engine[NullContext]):
    """Drive nav2 from the sentry brain; exchange target/engage with auto-aim."""

    subscribes_queue = "aim_target"  # enemy visibility from auto-aim
    publishes_queue = "engage_directive"  # to auto-aim (fail-open)

    def __init__(self, driver_registry=None):
        """Construct with no modules (the brain + nav2 client live in the child)."""
        super().__init__(
            modules=[], context_type=NullContext, driver_registry=driver_registry
        )

    # ------------------------------------------------------------------
    # Lifecycle (child process)
    # ------------------------------------------------------------------

    def initialize(self):
        """Bring up rclpy + BasicNavigator, wait for nav2, build the brain."""
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

        self._rclpy = rclpy
        self._PoseStamped = PoseStamped
        self._TaskResult = TaskResult

        cfg = config.autonav
        self._map_frame = cfg.map_frame
        self._waypoints = [(float(p[0]), float(p[1])) for p in cfg.waypoints]
        self._evade_offset = float(cfg.evade_offset)
        self._tick_dt = 1.0 / float(cfg.tick_rate)

        self._wp_idx = 0
        self._mode = None  # last chassis mode acted on (for transition detection)
        self._goal_active = False
        self._last_xy = (float(cfg.initial_pose[0]), float(cfg.initial_pose[1]))
        self._evade_center = None
        self._evade_side = 1

        self._brain = SentryBrain()

        if not rclpy.ok():
            rclpy.init()
        self._nav = BasicNavigator()
        self._nav.setInitialPose(self._pose(*self._last_xy))
        self.log.info("AutoNav waiting for nav2 to become active…")
        self._nav.waitUntilNav2Active()  # blocks until amcl + bt_navigator ACTIVE
        self.log.info(
            "nav2 active — sentry brain online (%d waypoints, evade ±%.2fm)",
            len(self._waypoints),
            self._evade_offset,
        )

    def execute(self):
        """One brain tick: read auto-aim, decide, drive nav2, emit engage."""
        target = self.latest_subscribed()  # AimTarget | None
        enemy = bool(target.visible) if target is not None else False
        out = self._brain.tick(BrainInputs(enemy_visible=enemy))

        self._update_pose_cache()
        if out.chassis == CHASSIS_PROGRESS:
            self._drive_progress()
        else:
            self._drive_evade()
        self._mode = out.chassis

        self.publish(EngageDirective(engage=out.engage))
        time.sleep(self._tick_dt)  # pace the brain (~tick_rate Hz)

    # ------------------------------------------------------------------
    # nav2 goal management
    # ------------------------------------------------------------------

    def _pose(self, x, y):
        """A map-frame PoseStamped at (x, y); orientation identity (yaw=MCU)."""
        p = self._PoseStamped()
        p.header.frame_id = self._map_frame
        p.header.stamp = self._nav.get_clock().now().to_msg()
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.orientation.w = 1.0
        return p

    def _update_pose_cache(self):
        """Cache the robot's current map position from nav feedback (evade anchor)."""
        if not self._goal_active:
            return
        fb = self._nav.getFeedback()
        pose = getattr(fb, "current_pose", None) if fb is not None else None
        if pose is not None:
            self._last_xy = (pose.pose.position.x, pose.pose.position.y)

    def _drive_progress(self):
        """Head to the current waypoint; advance to the next when it's reached."""
        if self._mode != CHASSIS_PROGRESS:  # just left evade (or starting up)
            self._cancel()
            self._send_waypoint_goal()
            return
        if self._goal_active and self._nav.isTaskComplete():
            reached = self._nav.getResult() == self._TaskResult.SUCCEEDED
            self._goal_active = False
            if reached:
                self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
            self._send_waypoint_goal()
        elif not self._goal_active:
            self._send_waypoint_goal()

    def _drive_evade(self):
        """Slow side-step: ping-pong small lateral goals around the contact spot."""
        if self._mode != CHASSIS_EVADE:  # just made contact: stop, anchor here
            self._cancel()
            self._evade_center = self._last_xy
            self._evade_side = 1
            self._send_evade_goal()
            return
        if self._goal_active and self._nav.isTaskComplete():
            self._goal_active = False
            self._evade_side *= -1  # to the other side
            self._send_evade_goal()
        elif not self._goal_active:
            self._send_evade_goal()

    def _send_waypoint_goal(self):
        self._nav.goToPose(self._pose(*self._waypoints[self._wp_idx]))
        self._goal_active = True

    def _send_evade_goal(self):
        cx, cy = self._evade_center or self._last_xy
        self._nav.goToPose(self._pose(cx, cy + self._evade_side * self._evade_offset))
        self._goal_active = True

    def _cancel(self):
        if self._goal_active:
            self._nav.cancelTask()
            self._goal_active = False
