#!/usr/bin/env python3
"""
最小检查: 确认 Unity 侧的挖土量是否通过 ROS 发出来了.

用法:
  # 只订阅, 打印挖土量 / 铲斗质量 (每 1s 一行)
  python3 rl/train/check_soil.py

  # 额外测一次 reset: 5 秒后发 /unity/reset_terrain, 看挖土量是否归零
  python3 rl/train/check_soil.py --reset-test

运行前提:
  - Unity 已进入 Play 模式
  - ros_tcp_endpoint 桥已连上 (否则收不到任何话题)
"""
import time
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool


class SoilChecker(Node):
    def __init__(self, reset_test: bool):
        super().__init__('soil_checker')
        self._vol = None
        self._mass = None
        self._last_print = 0.0
        self.create_subscription(Float64, '/unity/soil_volume', self._cb_vol, 10)
        self.create_subscription(Float64, '/unity/bucket_mass', self._cb_mass, 10)
        self._reset_pub = self.create_publisher(Bool, '/unity/reset_terrain', 10)
        self._reset_test = reset_test
        self._start = time.monotonic()
        self.get_logger().info('订阅中: /unity/soil_volume, /unity/bucket_mass')
        self.get_logger().info('挖土时会看到 soil_volume 增长; 一直 None 说明没收到')

    def _cb_vol(self, msg):
        self._vol = msg.data

    def _cb_mass(self, msg):
        self._mass = msg.data

    def tick(self):
        now = time.monotonic()
        # 5 秒后测一次 reset
        if self._reset_test and now - self._start > 5.0 and not getattr(self, '_reset_sent', False):
            self._reset_sent = True
            msg = Bool()
            msg.data = True
            self._reset_pub.publish(msg)
            self.get_logger().warn('>>> 已发 /unity/reset_terrain=True, 观察 soil_volume 是否归零')

        if now - self._last_print >= 1.0:
            self._last_print = now
            vol = 'None' if self._vol is None else f'{self._vol:.5f} m³'
            mass = 'None' if self._mass is None else f'{self._mass:.3f} kg'
            print(f'[t={now - self._start:5.1f}s] 挖土量={vol}  铲斗质量={mass}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reset-test', action='store_true')
    args = ap.parse_args()

    rclpy.init()
    node = SoilChecker(args.reset_test)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.tick()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
