import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

from excavator_kinematics.mdh import forward_kinematics, jacobian_position, MDH_PARAMS
from excavator_trajectory.trajectory import (
    find_closest, find_closest_continuous, get_point, get_q_at_s,
)
from excavator_controller.apf import (
    attractive_force, attractive_torque, joint_limit_torque, tipover_torque,
)
from excavator_control.reference_progress import advance_reference_progress
from excavator_control.terminal_state import (
    TerminalConfig, TerminalObservation, TerminalPhase, TerminalStateMachine,
)

# ── 绘图后端 (VNC 桌面弹窗) ──
_has_display = 'DISPLAY' in os.environ and os.environ.get('DISPLAY', '')
if _has_display:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt


class APF_Controller(Node):
    def __init__(self):
        super().__init__('apf_controller')

        self.K_att = 300.0
        # 力矩→位置增量增益. 实测 0.005 时收敛太慢 (齿尖蹭进管线斥力区被弹开),
        # 0.03 能收敛但关节动作偏猛 → 带动履带抖动. 折中取 0.02, 挖掘速度放慢一些.
        # 力矩→位置增量增益. 0.012 (降低): 工作装置运动更缓, 减小惯性冲击 → 减少机体左右抖
        self.K_imp = 0.012
        # 零空间参考构型增益: 把 bucket 等冗余自由度拉向示教参考, 防止乱卷
        self.K_joint = 60.0
        # bucket 独立高增益: q4 完全跟随示教卷斗节奏, 不被引力/漂移带偏
        # 250 确保放宽 max_delta 后卷斗力矩仍足够 (跟得上轨迹推进)
        self.K_joint_bucket = 250.0
        self.dt = 0.02
        self.q_min = np.array([MDH_PARAMS['joint_limits'][k][0]
                               for k in ['q1', 'q2', 'q3', 'q4']])
        self.q_max = np.array([MDH_PARAMS['joint_limits'][k][1]
                               for k in ['q1', 'q2', 'q3', 'q4']])
        self.joint_names = ['base_to_body_joint', 'body_to_boom_joint',
                            'boom_to_arm_joint', 'arm_to_bucket_joint']
        # 发布端 offset (JointStatePublisher): 控制器 FK 用它才能匹配物理齿尖
        # (零位验证: FK 误差 0.0003m). 轨迹是离线安全化的根轨迹, 控制器跟踪它.
        self.ros_offset = np.array([0.0, 0.765, -0.743, -0.05])

        self.sub = self.create_subscription(JointState, '/unity/joint_states', self._callback, 10)
        self.pub = self.create_publisher(JointState, '/unity/joint_command', 10)

        # 管线在 Σ0 系坐标. Unity 场景 underground_pipe 世界坐标
        # (-7.5, -1.2, 0), base_footprint 世界 X=-0.8699 → Σ0 = (6.6301, 0, -1.2).
        self.pipe_pos = np.array([6.6301, 0.0, -1.2])

        # ── 循环记录 ──
        # 轨迹进度初始化为 None: 首次收到关节状态时用全局最近点定位,
        # 机器可能生成在轨迹任意位置(实测常在 s≈0.87 末端), 不应从 0 开始。
        self.s_prev = None
        self.s_ref = None
        self.s_ref_step = 0.01
        # 一次作业终点：先进入保持卷斗，再用真实终点位置和 q4 角度共同判定完成。
        self.done = False
        self.terminal = TerminalStateMachine(TerminalConfig())
        self._run_started_at = time.monotonic()
        q_terminal = get_q_at_s(1.0) - self.ros_offset
        self.q_terminal = np.clip(q_terminal, self.q_min, self.q_max)
        # 铲斗卷斗专用平滑进度: 慢慢追 s_star, 避免 q4 参考随 s 跳变
        # (s 在挖掘段会一步跳 0.4+, 若 q4 直接跟 s, 卷斗永远跟不上)
        self.s_q4 = None
        # 关节指令低通滤波状态 (抑制急加速急停 → 反作用力平滑 → 减少机体左右抖)
        self.q_smooth = None
        # ── Z-based 铲斗卷斗: 入土即卷, 用硬目标 ──
        self._z_curl_active = False
        # ZMP 稳定裕度 (方案B: 监测稳定性)
        self.alpha_margin = 1.0
        self.cycle_count = 0
        self.cycle_data: list[dict] = []  # {(s, d_pipe, p_x, p_z)}

        self._saved = False
        self.get_logger().info(
            'APF Controller ready '
            f'mode=one_shot_done_no_reset pid={os.getpid()} '
            f'source={Path(__file__).resolve()} '
            f'ros_offset={np.array2string(self.ros_offset, precision=3)} '
            f'q4_limits=[{self.q_min[3]:.4f}, {self.q_max[3]:.4f}] '
            f'q4_terminal={self.q_terminal[3]:+.4f} '
            f'terminal_endpoint_tol={self.terminal.config.endpoint_error_threshold:.2f} '
            f'terminal_q4_tol={self.terminal.config.q4_error_threshold:.2f} '
            f'terminal_timeout={self.terminal.config.hold_timeout_sec:.1f}s '
            'q4_limit_torque=disabled'
        )

    def save_data(self):
        """析构或 Ctrl+C 时自动调用，保存 cycle_data 到 data/."""
        if self._saved or len(self.cycle_data) < 10:
            return
        self._saved = True
        from datetime import datetime
        from pathlib import Path
        out_dir = Path(__file__).resolve().parents[3] / "data"
        out_dir.mkdir(exist_ok=True)
        s_all = np.array([d['s'] for d in self.cycle_data])
        dp_all = np.array([d['d_pipe'] for d in self.cycle_data])
        px_all = np.array([d['p_x'] for d in self.cycle_data])
        pz_all = np.array([d['p_z'] for d in self.cycle_data])
        t_all = np.array([d['t'] for d in self.cycle_data])
        tag = datetime.now().strftime("%m%d_%H%M")
        fpath = out_dir / f"cycle_data_{tag}.npz"
        np.savez(str(fpath),
                 t=t_all, s=s_all, d_pipe=dp_all, p_x=px_all, p_z=pz_all,
                 cycle_count=self.cycle_count, dt=self.dt, pipe_pos=self.pipe_pos)
        self.get_logger().info(
            f'Data saved: {len(self.cycle_data)} frames, '
            f'{self.cycle_count} cycles → {fpath}')
        print(f"[save] {len(self.cycle_data)} frames, {self.cycle_count} cycles → {fpath}")

    def __del__(self):
        try:
            self.save_data()
        except Exception:
            pass

    def _callback(self, msg: JointState):
        # 一次作业模式: 完成后停止发布控制指令，挖掘机停在终点位置，
        # 不再拉向新一轮挖掘。
        if self.done:
            return
        q = self._parse_q(msg)
        if q is None:
            return

        # FK (输出轨迹坐标系, 与 WAYPOINTS 一致)
        p_tip, _ = forward_kinematics(q)

        if self.terminal.phase == TerminalPhase.TERMINAL_HOLD_TIMEOUT:
            frozen_q = self.terminal.update(
                time.monotonic(),
                TerminalObservation(
                    s_star=self.s_prev or 0.0,
                    path_error=0.0,
                    endpoint_error=float(np.linalg.norm(p_tip - get_point(1.0))),
                    q=q,
                    q4_terminal_target=self.q_terminal[3],
                ),
            ).frozen_q
            self.q_smooth = np.clip(frozen_q, self.q_min, self.q_max)
            self._publish_cmd(self.q_smooth)
            return

        # 轨迹最近点: 带连续性 (防止 s 在相邻段跳变 → 挖掘段被跳过 → 铲斗不卷)
        # 首次用全局最近点初始化, 之后用连续搜索 (s_prev 约束)
        # 终点保持阶段冻结 s: 不再搜索最近点, 防止安全化轨迹中更近的回折路点
        # 把 s 拉回挖掘段 → 齿尖在两个点之间来回拉 (震荡).
        if self.terminal.phase == TerminalPhase.TERMINAL_HOLD:
            s_star = self.s_prev
        elif self.s_prev is None:
            s_star, _ = find_closest(p_tip)
            self.s_prev = s_star
            self.get_logger().info(
                f'[init] 初始化 s_prev={self.s_prev:.3f} '
                f'({self.s_prev * 100:.0f}% 处), 从当前位置开始跟踪')
        else:
            s_star, _ = find_closest_continuous(p_tip, self.s_prev)
        path_err = float(np.linalg.norm(p_tip - get_point(s_star)))
        endpoint_err = float(np.linalg.norm(p_tip - get_point(1.0)))
        self.s_ref = advance_reference_progress(
            self.s_ref, s_star, self.s_ref_step
        )
        terminal_decision = self.terminal.update(
            time.monotonic(),
            TerminalObservation(
                s_star=s_star,
                path_error=path_err,
                endpoint_error=endpoint_err,
                q=q,
                q4_terminal_target=self.q_terminal[3],
            ),
        )

        if terminal_decision.transitioned:
            if terminal_decision.reason == 'terminal_entry':
                self.get_logger().info(
                    f'[terminal-entry] s={s_star:.3f} path_err={path_err:.3f}m '
                    f'endpoint_err={endpoint_err:.3f}m q4={q[3]:+.4f} '
                    f'q4_target={self.q_terminal[3]:+.4f}'
                )
            elif terminal_decision.reason == 'terminal_complete':
                self.done = True
                self.get_logger().info(
                    f'[terminal-complete] hold={terminal_decision.hold_elapsed_sec:.2f}s '
                    f'endpoint_err={endpoint_err:.3f}m q4={q[3]:+.4f} '
                    f'q4_err={q[3] - self.q_terminal[3]:+.4f}'
                )
                return
            elif terminal_decision.reason == 'terminal_hold_timeout':
                self.q_smooth = np.clip(
                    terminal_decision.frozen_q, self.q_min, self.q_max
                )
                self._publish_cmd(self.q_smooth)
                self.get_logger().warning(
                    f'[terminal-hold-timeout] hold={terminal_decision.hold_elapsed_sec:.2f}s '
                    f'endpoint_err={endpoint_err:.3f}m q4={q[3]:+.4f} '
                    f'q4_err={q[3] - self.q_terminal[3]:+.4f} '
                    f'frozen_q={np.array2string(self.q_smooth, precision=3)} '
                    'done=False'
                )
                return

        # 有限前视：s_ref+0.05 确保目标始终前移 (即使 s_star 暂时卡住),
        # s_star+0.10 允许 carrot 跳过局部路径弯曲 (如录制时的小幅回撤).
        if terminal_decision.force_terminal_target:
            s_target = 1.0
        else:
            s_target = min(self.s_ref + 0.05, s_star + 0.10, 1.0)
        p_target = get_point(s_target)

        # ── 任务空间引力 (跟踪安全化轨迹, 只驱动 swing/boom/arm) ──
        # bucket 是完全冗余自由度: 若让引力也驱动 q4, 它会自由漂移过卷.
        # 故 q4 不参与任务空间跟踪 (tau_att[3]=0), 由下方卷斗控制驱动.
        F_att = attractive_force(p_tip, p_target, self.K_att)
        J_p = jacobian_position(q)
        tau_total = attractive_torque(F_att, J_p)
        tau_total[3] = 0.0

        # ── 零空间: 跟踪示教参考构型 (boom/arm 冗余方向) ──
        # get_q_at_s 返回录制 q_ros (ROS空间), 减 offset 还原 raw 与 q 一致。
        # 终点保持时 q1–q3 直接跟终点构型，避免 s_star 量化使回位提前停止。
        if terminal_decision.force_terminal_target:
            q_ref = self.q_terminal
        else:
            q_ref = get_q_at_s(self.s_ref) - self.ros_offset
        N = np.eye(4) - J_p.T @ np.linalg.pinv(J_p).T
        tau_null = self.K_joint * (q_ref - q)
        tau_null[3] = 0.0   # q4 不在此投影 (投影会削弱卷斗驱动力)
        tau_total += N @ tau_null

        # ── 铲斗卷斗: Z-based 入土触发 ──
        # 齿尖 Z < Z_CURL_ENTRY 时激活卷斗, q4 目标从当前值单调推向终端值.
        # ── Z-based 铲斗卷斗: 入土即卷, 用硬目标直接拉 ──
        # 齿尖入土 → q4 目标 = -1.4; 出坑 → q4 目标 = q_terminal(-1.76).
        # 不用渐进 target (会产生跟踪滞后), max_delta 已提供速率限制.
        _Q4_IN_SOIL = -1.4      # 土中硬目标, 实际 τ=250×Δq, max_delta=0.08 平滑

        if terminal_decision.force_terminal_target:
            q4_target = self.q_terminal[3]
        else:
            if p_tip[2] < -0.03:
                self._z_curl_active = True
            if self._z_curl_active:
                q4_target = _Q4_IN_SOIL if p_tip[2] < 0.0 else self.q_terminal[3]
            else:
                q4_target = q_ref[3]   # 入土前: 保持全开

        q4_clip = np.clip(q4_target, self.q_min[3], self.q_max[3])
        tau_bucket = self.K_joint_bucket * (q4_clip - q[3])
        tau_total[3] = tau_bucket

        # 保留 q1–q3 的软限位保护；q4 由物理限位和最终命令裁剪保护。
        tau_limit = joint_limit_torque(q, self.q_min, self.q_max)
        tau_limit[3] = 0.0
        tau_total += tau_limit

        # ── ZMP 倾覆保护 (L1 最高优先级) ──
        # α_margin < α_thresh 时斥力把工作装置推回安全区
        tau_tip, self.alpha_margin = tipover_torque(q)
        tau_total += tau_tip

        # 关节位置增量.
        # - 提升/回位段 (s>=0.68): 正常速度
        # - 下挖段 (s<0.68): boom 0.03 arm 0.04
        q_des = q + self.K_imp * tau_total * self.dt
        if s_star >= 0.68:
            max_delta = np.array([0.06, 0.04, 0.05, 0.08])
        else:
            max_delta = np.array([0.06, 0.03, 0.04, 0.08])
        delta_q = np.clip(q_des - q, -max_delta, max_delta)

        # ── q4 直接驱动: 绕过 K_imp/max_delta 流水线 ──
        # 常规 tau→位置流水线对 q4 太慢 (max_delta 0.08 × 滤波 0.5 × LockController 柔度
        # → 实际仅 0.006 rad/step). 卷斗激活时用独立高速通道直接跟目标.
        _q4_step = 0.10      # rad/step (5 rad/s @ 50Hz)
        if self._z_curl_active or terminal_decision.force_terminal_target:
            _q4_err = q4_clip - q[3]
            if abs(_q4_err) > _q4_step:
                _q4_err = np.sign(_q4_err) * _q4_step
            delta_q[3] = _q4_err

        q_des = q + delta_q

        # 关节指令低通滤波 (q1–q3). q4 卷斗激活时跳过滤波, 直接跟目标.
        if self.q_smooth is None:
            self.q_smooth = q_des.copy()
        else:
            self.q_smooth = 0.5 * q_des + 0.5 * self.q_smooth
            if self._z_curl_active or terminal_decision.force_terminal_target:
                self.q_smooth[3] = q_des[3]   # q4: 不滤波
        e_des = np.clip(self.q_smooth, self.q_min, self.q_max)

        self._publish_cmd(e_des)

        # ── 记录循环数据 ──
        d_pipe = float(np.sqrt((p_tip[0] - self.pipe_pos[0])**2 +
                               (p_tip[2] - self.pipe_pos[2])**2))
        # 仅路径跟踪阶段检测历史循环记录，终点保持阶段绝不能触发新循环。
        if (self.terminal.phase == TerminalPhase.FOLLOW_PATH and
                self.s_prev > 0.9 and s_star < 0.1):
            self.cycle_count += 1
            self.get_logger().info(
                f'══════ Cycle {self.cycle_count} complete: '
                f'{len(self.cycle_data)} frames logged ══════')

        self.cycle_data.append({
            't': time.monotonic() - self._run_started_at,
            's': float(s_star), 'd_pipe': d_pipe,
            'p_x': float(p_tip[0]), 'p_z': float(p_tip[2]),
        })
        self.s_prev = s_star

        # 日志. path_err 是齿尖到最近轨迹点的误差；终点保持还需 endpoint_err 和 q4。
        self.get_logger().info(
            f'phase={terminal_decision.phase.value} s={s_star:.2f} '
            f's_ref={self.s_ref:.2f} path_err={path_err:.3f}m '
            f'endpoint_err={endpoint_err:.3f}m '
            f'd_pipe={d_pipe:.2f}m α_m={self.alpha_margin:.3f} '
            f'z={p_tip[2]:+.2f} q4={q[3]:+.3f} '
            f'|tau|={np.linalg.norm(tau_total):.0f}',
            throttle_duration_sec=0.2
        )
        if terminal_decision.force_terminal_target:
            self.get_logger().info(
                f'[terminal-hold] hold={terminal_decision.hold_elapsed_sec:.2f}/'
                f'{self.terminal.config.hold_timeout_sec:.1f}s target_s=1.000 '
                f'endpoint_err={endpoint_err:.3f}m '
                f'q4={q[3]:+.4f} q4_target={self.q_terminal[3]:+.4f} '
                f'q4_err={q[3] - self.q_terminal[3]:+.4f} '
                f'endpoint_ok={terminal_decision.endpoint_reached} '
                f'q4_ok={terminal_decision.q4_reached}',
                throttle_duration_sec=0.5
            )

    def _parse_q(self, msg: JointState) -> np.ndarray | None:
        if len(msg.position) < 4:
            return None
        q = np.zeros(4)
        for i, name in enumerate(self.joint_names):
            try:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
            except ValueError:
                return None
        # 还原为 AGX raw 角 (FK 标定模型基于 raw)
        return q - self.ros_offset

    def _publish_cmd(self, q_des: np.ndarray):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        # 内部用 raw 角, 发布时加回发布端 offset (Unity 订阅端期望 ROS 空间)
        msg.position = (q_des + self.ros_offset).tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = APF_Controller()

    try:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    finally:
        node.save_data()
        rclpy.shutdown()

    # ── Ctrl+C 后自动弹图 ──
    if not _has_display:
        print("[plot] No DISPLAY available, skip figure window.")
        return
    if len(node.cycle_data) < 10:
        print("[plot] Not enough data to plot.")
        return

    dp_all = np.array([d['d_pipe'] for d in node.cycle_data])
    px_all = np.array([d['p_x'] for d in node.cycle_data])
    pz_all = np.array([d['p_z'] for d in node.cycle_data])
    t = np.array([d['t'] for d in node.cycle_data])

    # ── Nature Skills style ──
    PAL = {
        "blue": "#0F4D92", "red": "#B64342", "teal": "#42949E",
        "grey": "#767676", "dark": "#4D4D4D",
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4.5, 4.2), num='APF Tracking Report', sharex=True
    )

    # ── Panel A: d_pipe vs time ──
    ax1.plot(t, dp_all, color=PAL["blue"], linewidth=0.8)
    ax1.axhline(y=1.5, color=PAL["red"], linewidth=0.6, linestyle='--', alpha=0.6)
    ax1.text(t[-1]*1.01, 1.52, r'$d_{\rm safe}=1.5$', fontsize=6, color=PAL["red"], va='bottom')
    ax1.set_ylabel(r'$d_{\rm pipe}$ (m)', fontsize=8, labelpad=3)
    ax1.tick_params(labelsize=7, pad=2)

    # ── Panel B: tip trajectory in XZ-plane ──
    ax2.plot(px_all, pz_all, color=PAL["blue"], linewidth=0.8)
    pipe_s0 = node.pipe_pos
    ax2.plot(pipe_s0[0], pipe_s0[2], 'o', color=PAL["red"], markersize=5)
    ax2.text(pipe_s0[0], pipe_s0[2] + 0.15, 'Pipe', fontsize=6, color=PAL["red"], ha='center')
    ax2.set_xlabel('X — forward (m)', fontsize=8, labelpad=3)
    ax2.set_ylabel('Z — height (m)', fontsize=8, labelpad=3)
    ax2.tick_params(labelsize=7, pad=2)
    ax2.set_aspect('equal')

    fig.suptitle(f'APF Tracking — {node.cycle_count} cycle(s), '
                 f'{len(node.cycle_data)} frames',
                 fontsize=9, y=0.98)
    fig.tight_layout(pad=0.6)
    plt.show()
    print(f"[plot] {node.cycle_count} cycle(s), {len(node.cycle_data)} frames shown.")
