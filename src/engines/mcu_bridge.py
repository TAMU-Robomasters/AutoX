"""``McuBridgeEngine`` — the inverted ``uart_odom_node`` (plan 09).

In plan 02 a nav2 node (``uart_odom_node``) **owned** ``/dev/ttyTHS1`` and AutoX
reached the latency-critical transform/firing-solution traffic over a ROS service
— a DDS round-trip on every aim frame. Plan 09 inverts that: AutoX's ``McuDriver``
owns the UART (so ``'t'``/``'d'`` stay in-process on the queue-RPC fast path), and
this engine is the *latency-tolerant* nav bridge AutoX hosts. It:

* polls the MCU for chassis odometry (``'q'``) via :class:`McuClient` (low-priority
  lane) and republishes it as ROS ``/odom`` (+ ``odom->base_link`` TF) and
  ``/embedded_odom``, plus a static ``embedded_odom->odom`` TF;
* subscribes ``/cmd_vel``, rotates the base_link ``Twist`` into the embedded world
  frame by the latest gyro heading, and sends it to the MCU (``'v'``);
* keeps a zero-velocity safety watchdog so the chassis stops if ``/cmd_vel`` goes
  stale (nav2 dies / goal completes).

The frames + axis mapping are ported byte-for-byte from
``nav2-sim-testing/.../uart_odom_node.py`` so the two are interchangeable on the
wire. The ``on_query_transform`` / ``on_firing_solution`` ROS handlers from that
node are intentionally **dropped** — in the inverted design auto-aim reaches the
MCU in-process through its own :class:`McuClient`, never over ROS.

``rclpy`` and the ROS message/tf types are imported lazily in :meth:`initialize`
(the child process) so this module stays importable on a machine without ROS, and
so nothing ROS-related is touched in the parent before fork (the multiprocessing
rule). Only launched when ``ros.stack == nav2`` and ``autonav.publish_odom`` (AutoX
is the odom source — real hardware / the MCU harness, not the Gazebo full-sim where
Gazebo owns ``/odom``).
"""

import math
import time

from src.core.engine import Engine
from src.drivers.mcu import McuDriver
from src.toolbox.globals import config
from src.types.null import NullContext

# -90° about +z as a quaternion: embedded_odom (x=right, y=forward) -> odom
# (x=forward, y=left). qz=sin(-pi/4), qw=cos(-pi/4).
_EMB_TO_ODOM_QZ = -0.7071067811865476
_EMB_TO_ODOM_QW = 0.7071067811865476


class McuBridgeEngine(Engine[NullContext]):
    """Bridge the AutoX-owned MCU UART to nav2's ``/odom`` + ``/cmd_vel``."""

    drivers = {"mcu": McuDriver}

    def __init__(self, driver_registry=None):
        """Construct with no modules (pure rclpy<->MCU bridge)."""
        super().__init__(
            modules=[], context_type=NullContext, driver_registry=driver_registry
        )

    # ------------------------------------------------------------------
    # Lifecycle (child process)
    # ------------------------------------------------------------------

    def initialize(self):
        """Bring up rclpy, the node, its pub/sub/TF, and the poll/watchdog timers."""
        import rclpy  # lazy: keeps this module importable without ROS
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self._rclpy = rclpy
        self._Odometry = Odometry
        self._TransformStamped = TransformStamped

        cfg = config.autonav
        self._odom_frame = cfg.odom_frame
        self._base_frame = cfg.base_frame
        self._embedded_odom_frame = cfg.embedded_odom_frame
        self._publish_tf = bool(cfg.publish_tf)
        self._cmd_vel_timeout = float(cfg.cmd_vel_timeout)

        self._mcu = self.driver("mcu")  # McuClient (built before initialize())

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("autox_mcu_bridge")

        self._odom_pub = self._node.create_publisher(Odometry, "/odom", 10)
        self._embedded_odom_pub = self._node.create_publisher(
            Odometry, "/embedded_odom", 10
        )
        self._tf_broadcaster = TransformBroadcaster(self._node)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self._node)

        # Latest gyro heading from the odom poll, cached so the (independently
        # firing) cmd_vel callback can rotate base_link commands into the world
        # frame. Both run on this node's single-threaded executor -> no lock.
        self._latest_theta = 0.0
        self._last_cmd_vel_time = 0.0  # monotonic s; 0 = none received yet

        self._node.create_subscription(Twist, cfg.cmd_vel_topic, self._on_cmd_vel, 10)

        # Static embedded_odom->odom (latched, so late subscribers still get it).
        self._publish_static_embedded_to_odom_tf()

        # Odom poll at odom_rate; zero-velocity safety watchdog at 10 Hz.
        self._node.create_timer(1.0 / float(cfg.odom_rate), self._poll_odom)
        self._node.create_timer(0.1, self._cmd_vel_watchdog)

        self.log.info(
            "McuBridge up: /odom @ %.0f Hz, /cmd_vel bridge (timeout %.2fs)",
            float(cfg.odom_rate),
            self._cmd_vel_timeout,
        )

    def execute(self):
        """Pump the rclpy executor once (services timers + the cmd_vel sub)."""
        try:
            # Blocks up to 0.1s when idle (no busy-loop); returns as soon as the
            # 100 Hz odom timer / a cmd_vel message is ready.
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
        except Exception:  # keep the bridge alive across a transient ROS error
            self.log.exception("McuBridge spin_once failed")

    # ------------------------------------------------------------------
    # Timers / callbacks
    # ------------------------------------------------------------------

    def _poll_odom(self):
        """Poll the MCU for odometry and republish it as /odom + /embedded_odom."""
        odom = self._mcu.get_odometry()
        if odom is None:
            return
        x, y, theta = odom
        self._latest_theta = theta  # for the cmd_vel world-frame rotation
        self._publish_embedded_odom(x, y, theta)
        self._publish_transformed_odom(x, y, theta)
        if self.log.isEnabledFor(10):  # DEBUG: hot path, gate the format
            self.log.debug(
                "odom emb x=%.3f y=%.3f theta=%.1fdeg", x, y, math.degrees(theta)
            )

    def _on_cmd_vel(self, msg):
        """Rotate a base_link Twist into the embedded world frame and send it.

        DWB publishes /cmd_vel in base_link (x=forward, y=left); the chassis spins
        independently under MCU control, so we express the command in the fixed
        world frame and let the MCU do world->chassis with its own live gyro angle.
        ``angular.z`` is discarded — yaw is owned by the MCU.
        """
        th = self._latest_theta
        cos_th, sin_th = math.cos(th), math.sin(th)
        v_odom_x = cos_th * msg.linear.x - sin_th * msg.linear.y  # forward
        v_odom_y = sin_th * msg.linear.x + cos_th * msg.linear.y  # left
        emb_vx = -v_odom_y  # embedded +x = right
        emb_vy = v_odom_x  # embedded +y = forward
        self._last_cmd_vel_time = time.monotonic()
        self._mcu.send_velocity(emb_vx, emb_vy)

    def _cmd_vel_watchdog(self):
        """Command zero velocity if /cmd_vel has gone stale (safety stop)."""
        if self._last_cmd_vel_time == 0.0:
            return  # never commanded yet — don't fight other senders
        if time.monotonic() - self._last_cmd_vel_time > self._cmd_vel_timeout:
            self._mcu.send_velocity(0.0, 0.0)

    # ------------------------------------------------------------------
    # Publish helpers (frames/axes ported from uart_odom_node)
    # ------------------------------------------------------------------

    def _now(self):
        return self._node.get_clock().now().to_msg()

    def _publish_embedded_odom(self, x, y, theta):
        """Publish raw odometry in the embedded_odom frame."""
        msg = self._Odometry()
        msg.header.stamp = self._now()
        msg.header.frame_id = self._embedded_odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        self._embedded_odom_pub.publish(msg)

    def _publish_transformed_odom(self, x, y, theta):
        """Publish /odom in the ROS odom frame (+ odom->base_link TF).

        embedded (x=right, y=forward) -> odom (x=forward, y=left):
        ``odom_x = emb_y``, ``odom_y = -emb_x``; theta carries over (theta=0 faces
        embedded +y = ROS +x, no offset).
        """
        odom_x, odom_y = y, -x
        qz, qw = math.sin(theta / 2.0), math.cos(theta / 2.0)
        now = self._now()

        msg = self._Odometry()
        msg.header.stamp = now
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x = float(odom_x)
        msg.pose.pose.position.y = float(odom_y)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self._odom_pub.publish(msg)

        if self._publish_tf:
            t = self._TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._odom_frame
            t.child_frame_id = self._base_frame
            t.transform.translation.x = float(odom_x)
            t.transform.translation.y = float(odom_y)
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(t)

    def _publish_static_embedded_to_odom_tf(self):
        """Latch the static embedded_odom->odom transform (-90° about z)."""
        t = self._TransformStamped()
        t.header.stamp = self._now()
        t.header.frame_id = self._embedded_odom_frame
        t.child_frame_id = self._odom_frame
        t.transform.rotation.z = _EMB_TO_ODOM_QZ
        t.transform.rotation.w = _EMB_TO_ODOM_QW
        self._static_tf_broadcaster.sendTransform(t)
        self.log.info("published static TF %s -> %s", "embedded_odom", self._odom_frame)
