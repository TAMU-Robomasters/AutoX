"""ROS/DDS round-trip latency — the cost the inverted UART keeps OFF the aim path.

Times a ``std_srvs/Trigger`` request/response over DDS (a server node + a client
node; service calls go through the RMW, no intra-process shortcut). This is
representative of the deprecated ``MCU=ROS`` transform query that plan 02 put on
*every* auto-aim frame, and of the per-hop cost nav pays to receive ``/odom`` from
the bridge. Compare against bench_mcu_latency.py's in-process numbers.

    python docker/bench_ros_latency.py     # in the autox container (ROS sourced)
"""

import statistics
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger


def report(samples, label):
    """Print percentile latency stats (milliseconds) for a sample list."""
    s = sorted(samples)
    pct = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    print(
        f"  {label:34s} n={len(s):5d}  p50={pct(0.5):6.3f}  p90={pct(0.9):6.3f}  "
        f"p99={pct(0.99):6.3f}  max={s[-1]:7.3f}  mean={statistics.mean(s):6.3f}   (ms)"
    )


def main():
    """Bench a DDS service round-trip between two nodes in this process."""
    rclpy.init()
    server = rclpy.create_node("bench_srv")
    server.create_service(
        Trigger,
        "/bench/trigger",
        lambda req, resp: setattr(resp, "success", True) or resp,
    )
    exe = SingleThreadedExecutor()
    exe.add_node(server)
    spinner = threading.Thread(target=exe.spin, daemon=True)
    spinner.start()

    client_node = rclpy.create_node("bench_cli")
    cli = client_node.create_client(Trigger, "/bench/trigger")
    cli.wait_for_service()

    def call_once():
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(client_node, fut)
        return fut.result()

    for _ in range(100):  # warm up
        call_once()

    n = 2000
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        call_once()
        samples.append((time.perf_counter() - t) * 1e3)
    print("ROS/DDS service round-trip (rmw_fastrtps_cpp):")
    report(samples, "Trigger round-trip")

    exe.shutdown()
    server.destroy_node()
    client_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
