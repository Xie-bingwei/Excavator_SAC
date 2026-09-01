#!/usr/bin/env python3
"""
UnityExcavatorEnv — 在线真实土壤 RL 环境 (gym.Env).

把 Unity/AGX 真实物理仿真通过 ROS 包装成 gym 环境, 供 SB3 SAC 在线训练:
  - reset(): 先回位(把铲斗拉回起点构型) → 发 /unity/reset_terrain 重置地形 → 返回初始 obs
  - step(action): APF 基线 + SAC 残差 → 发布 /unity/joint_command,
                  等下一帧关节状态, 用 /unity/soil_volume 增量当奖励

奖励: reward = reward_scale * Δvolume − w_pipe·max(0, d_safe−d_pipe) − w_hit·[d_pipe < r_pipe+margin]

依赖 ROS 话题 (由 Unity MassVolumeCounter.cs + JointStatePublisher.cs 提供):
  订阅: /unity/joint_states (sensor_msgs/JointState)
        /unity/soil_volume   (std_msgs/Float64, 累计挖土体积 m³)
  发布: /unity/joint_command (sensor_msgs/JointState)
        /unity/reset_terrain (std_msgs/Bool)
"""
import os
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 运动学 / 轨迹 / APF 模块 (与 control.py 一致的导入路径) ──
os.environ.setdefault('TRAJECTORY_GENERATING', '1')
_WS = Path(__file__).resolve().parents[2]  # rl/env → ex_ws
for _pkg in ['excavator_kinematics', 'excavator_trajectory',
             'excavator_controller', 'excavator_control']:
    _d = str(_WS / 'src' / _pkg)
    if _d not in sys.path:
        sys.path.insert(0, _d)

from excavator_kinematics.mdh import (  # noqa: E402
    forward_kinematics, jacobian_position, MDH_PARAMS,
)
from excavator_trajectory.trajectory import (  # noqa: E402
    find_closest, find_closest_continuous, get_point, get_q_at_s,
)
from excavator_controller.apf import (  # noqa: E402
    attractive_force, attractive_torque, joint_limit_torque, tipover_torque,
)
from excavator_control.reference_progress import advance_reference_progress  # noqa: E402
from excavator_control.terminal_state import (  # noqa: E402
    TerminalConfig, TerminalObservation, TerminalPhase, TerminalStateMachine,
)


class CyclePhase(str, Enum):
    """完整作业循环阶段: 挖 → 摆90° → 倒土 → 回位 → (循环)."""
    DIG = 'dig'
    SWING = 'swing'
    DUMP = 'dump'
    RETURN = 'return'


class APFStepper:
    """APF 控制步进器: 从 control.py 抽出的同步控制逻辑 (挖掘段).

    step() 给定当前关节角 q 与 SAC 残差 action, 返回发布指令 q_des (raw 空间)
    以及该 episode 是否完成 (terminal_complete / hold_timeout)。
    """

    def __init__(self):
        # 控制增益 (与 control.py 保持一致)
        self.K_att = 300.0
        self.K_imp = 0.012
        self.K_joint = 60.0
        self.K_joint_bucket = 250.0
        self.dt = 0.02
        self.q_min = np.array([MDH_PARAMS['joint_limits'][k][0]
                               for k in ['q1', 'q2', 'q3', 'q4']])
        self.q_max = np.array([MDH_PARAMS['joint_limits'][k][1]
                               for k in ['q1', 'q2', 'q3', 'q4']])
        self.ros_offset = np.array([0.0, 0.765, -0.743, -0.05])
        self.pipe_pos = np.array([6.6301, 0.0, -1.2])
        self.s_ref_step = 0.01
        self.q_terminal = np.clip(get_q_at_s(1.0) - self.ros_offset,
                                  self.q_min, self.q_max)
        self._Q4_IN_SOIL = -1.4  # 土中卷斗硬目标 (与 control.py 一致)
        # ── 摆动 + 倒土 (单周期作业循环: 挖→摆90°→倒土, 见 ROADMAP.md) ──
        self.swing_target = +np.pi / 2   # 摆 90° (顺/逆皆可, 先取 +90°)
        self.swing_tol = 0.05            # 摆动到位容差 rad (~3°)
        self._swing_step = 0.08          # 摆动限速 rad/step
        self.q4_open = self.q_max[3]     # 铲斗全开角 (倒土)
        self.q4_open_tol = 0.08          # 开斗到位容差 rad
        self._dump_step = 0.10           # 开斗限速 rad/step
        self._swing_q_hold = None        # 摆动阶段冻结的 q2/q3/q4 参考
        self._dump_q_hold = None         # 倒土阶段冻结的 q1/q2/q3 参考
        # ── 回位 (倒土后回开始位姿, 循环挖掘) ──
        self.home_q_raw = np.clip(get_q_at_s(0.0) - self.ros_offset,
                                  self.q_min, self.q_max)
        self.home_tol = 0.05             # 回位到位容差 rad
        self._return_step = 0.05         # 回位各关节限速 rad/step
        self.reset()

    def reset(self):
        """重置作业循环状态 (每 episode 开始时调用)."""
        self.done = False
        self.phase = CyclePhase.DIG
        self._swing_q_hold = None
        self._dump_q_hold = None
        self._reset_dig_state()
        # 最新状态 (供 obs 构建)
        self.p_tip = None
        self.alpha_margin = 1.0
        self.d_pipe = 5.8

    def _reset_dig_state(self):
        """重置挖掘段跟踪状态 (episode 开始 / 回位后重新进入 DIG 时调用)."""
        self.s_prev = None
        self.s_ref = None
        self.s_q4 = None
        self.q_smooth = None
        self._z_curl_active = False
        self.s_star = 0.0
        self.terminal = TerminalStateMachine(TerminalConfig())

    def step(self, q: np.ndarray, base_angvel: np.ndarray,
             action: np.ndarray) -> tuple[np.ndarray, bool]:
        """阶段机派发: DIG → SWING → DUMP → RETURN → (循环), 返回 (q_des_raw, done)."""
        if self.phase == CyclePhase.SWING:
            return self._step_swing(q, action)
        if self.phase == CyclePhase.DUMP:
            return self._step_dump(q, action)
        if self.phase == CyclePhase.RETURN:
            return self._step_return(q, action)
        return self._step_dig(q, base_angvel, action)

    def _lowpass(self, q_des: np.ndarray) -> np.ndarray:
        if self.q_smooth is None:
            self.q_smooth = q_des.copy()
        else:
            self.q_smooth = 0.5 * q_des + 0.5 * self.q_smooth
        return np.clip(self.q_smooth, self.q_min, self.q_max)

    def start_return(self):
        """由环境在检测到倒空后调用, 进入回位阶段."""
        self.phase = CyclePhase.RETURN

    def _step_dig(self, q: np.ndarray, base_angvel: np.ndarray,
                  action: np.ndarray) -> tuple[np.ndarray, bool]:
        """DIG 阶段: APF 轨迹跟踪 + Z-based 卷斗 (原挖掘逻辑)."""
        p_tip, _ = forward_kinematics(q)
        self.p_tip = p_tip

        # ── 终点保持超时 → 冻结当前构型, 结束该 episode ──
        if self.terminal.phase == TerminalPhase.TERMINAL_HOLD_TIMEOUT:
            frozen_q = self.terminal.update(
                time.monotonic(),
                TerminalObservation(
                    s_star=self.s_prev or 0.0, path_error=0.0,
                    endpoint_error=float(np.linalg.norm(p_tip - get_point(1.0))),
                    q=q, q4_terminal_target=self.q_terminal[3],
                ),
            ).frozen_q
            self.q_smooth = np.clip(frozen_q, self.q_min, self.q_max)
            self.done = True
            return self.q_smooth, self.done

        # ── 轨迹最近点 (带连续性) ──
        if self.terminal.phase == TerminalPhase.TERMINAL_HOLD:
            s_star = self.s_prev
        elif self.s_prev is None:
            s_star, _ = find_closest(p_tip)
            self.s_prev = s_star
        else:
            s_star, _ = find_closest_continuous(p_tip, self.s_prev)
        path_err = float(np.linalg.norm(p_tip - get_point(s_star)))
        endpoint_err = float(np.linalg.norm(p_tip - get_point(1.0)))
        self.s_ref = advance_reference_progress(self.s_ref, s_star, self.s_ref_step)
        terminal_decision = self.terminal.update(
            time.monotonic(),
            TerminalObservation(
                s_star=s_star, path_error=path_err, endpoint_error=endpoint_err,
                q=q, q4_terminal_target=self.q_terminal[3],
            ),
        )

        # 挖掘完成: 进入摆动阶段 (摆90°后倒土)
        if terminal_decision.reason == 'terminal_complete':
            self.phase = CyclePhase.SWING
            self.s_star = s_star
            self.s_prev = s_star
            self._swing_q_hold = q.copy()   # 冻结 q2/q3/q4 参考
            return np.clip(q, self.q_min, self.q_max), False

        # ── 任务空间引力 (只驱动 swing/boom/arm) ──
        if terminal_decision.force_terminal_target:
            s_target = 1.0
        else:
            s_target = min(self.s_ref + 0.05, s_star + 0.10, 1.0)
        p_target = get_point(s_target)

        F_att = attractive_force(p_tip, p_target, self.K_att)
        J_p = jacobian_position(q)
        tau_total = attractive_torque(F_att, J_p)
        tau_total[3] = 0.0

        # ── 零空间: 跟踪示教参考构型 ──
        if terminal_decision.force_terminal_target:
            q_ref = self.q_terminal
        else:
            q_ref = get_q_at_s(self.s_ref) - self.ros_offset
        N = np.eye(4) - J_p.T @ np.linalg.pinv(J_p).T
        tau_null = self.K_joint * (q_ref - q)
        tau_null[3] = 0.0
        tau_total += N @ tau_null

        # ── 铲斗卷斗: Z-based 入土触发 ──
        if terminal_decision.force_terminal_target:
            q4_target = self.q_terminal[3]
        else:
            if p_tip[2] < -0.03:
                self._z_curl_active = True
            if self._z_curl_active:
                q4_target = self._Q4_IN_SOIL if p_tip[2] < 0.0 else self.q_terminal[3]
            else:
                q4_target = q_ref[3]
        q4_clip = np.clip(q4_target, self.q_min[3], self.q_max[3])
        tau_bucket = self.K_joint_bucket * (q4_clip - q[3])
        tau_total[3] = tau_bucket

        # ── 软限位 + ZMP 倾覆保护 ──
        tau_limit = joint_limit_torque(q, self.q_min, self.q_max)
        tau_limit[3] = 0.0
        tau_total += tau_limit
        tau_tip, self.alpha_margin = tipover_torque(q)
        tau_total += tau_tip

        # ── 关节位置增量 (带 max_delta 限速) ──
        q_des = q + self.K_imp * tau_total * self.dt
        if s_star >= 0.68:
            max_delta = np.array([0.06, 0.04, 0.05, 0.08])
        else:
            max_delta = np.array([0.06, 0.03, 0.04, 0.08])
        delta_q = np.clip(q_des - q, -max_delta, max_delta)

        # ── q4 直接驱动 (绕过 K_imp/max_delta, 独立高速通道) ──
        _q4_step = 0.10
        if self._z_curl_active or terminal_decision.force_terminal_target:
            _q4_err = q4_clip - q[3]
            if abs(_q4_err) > _q4_step:
                _q4_err = np.sign(_q4_err) * _q4_step
            delta_q[3] = _q4_err

        q_des = q + delta_q
        # ── SAC RL 残差 ──
        q_des += action

        # ── 关节指令低通滤波 (q4 卷斗激活时跳过滤波) ──
        if self.q_smooth is None:
            self.q_smooth = q_des.copy()
        else:
            self.q_smooth = 0.5 * q_des + 0.5 * self.q_smooth
            if self._z_curl_active or terminal_decision.force_terminal_target:
                self.q_smooth[3] = q_des[3]
        q_des_out = np.clip(self.q_smooth, self.q_min, self.q_max)

        self.s_star = s_star
        self.s_prev = s_star
        self.d_pipe = float(np.sqrt((p_tip[0] - self.pipe_pos[0]) ** 2 +
                                    (p_tip[2] - self.pipe_pos[2]) ** 2))
        return q_des_out, self.done

    # ────────────────────────────────────────────────
    #  SWING / DUMP 阶段 (摆90° + 倒土)
    # ────────────────────────────────────────────────
    def _step_swing(self, q: np.ndarray, action: np.ndarray):
        """SWING 阶段: P 控制 q1 → swing_target, 冻结 q2/q3/q4 (举升姿态)."""
        if self._swing_q_hold is None:
            self._swing_q_hold = q.copy()
        q_target = self._swing_q_hold.copy()
        q_target[0] = self.swing_target

        err = q_target - q
        err[0] = np.clip(err[0], -self._swing_step, self._swing_step)
        err[1:] = np.clip(err[1:], -0.02, 0.02)   # 其余关节轻微保持

        tau_limit = joint_limit_torque(q, self.q_min, self.q_max)
        tau_limit[3] = 0.0
        tau_tip, self.alpha_margin = tipover_torque(q)
        q_des = q + err + self.K_imp * (tau_limit + tau_tip) * self.dt
        q_des_out = self._lowpass(q_des)

        # 刷新 obs 状态
        self.p_tip, _ = forward_kinematics(q)
        self.d_pipe = float(np.sqrt((self.p_tip[0] - self.pipe_pos[0]) ** 2 +
                                    (self.p_tip[2] - self.pipe_pos[2]) ** 2))
        self.s_star = 1.0
        self.s_prev = 1.0

        if abs(q[0] - self.swing_target) < self.swing_tol:
            self.phase = CyclePhase.DUMP
            self._dump_q_hold = q.copy()
        return q_des_out, False

    def _step_dump(self, q: np.ndarray, action: np.ndarray):
        """DUMP 阶段: P 控制 q4 → 全开(倒土), 冻结 q1(已摆到位)/q2/q3."""
        if self._dump_q_hold is None:
            self._dump_q_hold = q.copy()
        q_target = self._dump_q_hold.copy()
        q_target[3] = self.q4_open

        err = q_target - q
        err[3] = np.clip(err[3], -self._dump_step, self._dump_step)
        err[:3] = np.clip(err[:3], -0.02, 0.02)

        tau_limit = joint_limit_torque(q, self.q_min, self.q_max)
        tau_tip, self.alpha_margin = tipover_torque(q)
        q_des = q + err + self.K_imp * (tau_limit + tau_tip) * self.dt
        q_des_out = self._lowpass(q_des)

        # 刷新 obs 状态
        self.p_tip, _ = forward_kinematics(q)
        self.d_pipe = float(np.sqrt((self.p_tip[0] - self.pipe_pos[0]) ** 2 +
                                    (self.p_tip[2] - self.pipe_pos[2]) ** 2))
        self.s_star = 1.0
        self.s_prev = 1.0
        return q_des_out, False

    def _step_return(self, q: np.ndarray, action: np.ndarray):
        """RETURN 阶段: 比例控制回开始位姿 (与 reset 的 _home 一致), 到位后重置挖掘状态转 DIG."""
        err = self.home_q_raw - q
        err = np.clip(err, -self._return_step, self._return_step)
        q_des_out = np.clip(q + err, self.q_min, self.q_max)

        # 刷新 obs 状态
        self.p_tip, _ = forward_kinematics(q)
        self.d_pipe = float(np.sqrt((self.p_tip[0] - self.pipe_pos[0]) ** 2 +
                                    (self.p_tip[2] - self.pipe_pos[2]) ** 2))
        self.s_star = 0.0
        self.s_prev = 0.0

        if np.all(np.abs(self.home_q_raw - q) < self.home_tol):
            self._reset_dig_state()
            self.phase = CyclePhase.DIG
            self._swing_q_hold = None
            self._dump_q_hold = None
        return q_des_out, False


class UnityExcavatorEnv(gym.Env):
    """在线真实土壤环境: ROS <-> Unity/AGX 同步步进."""

    metadata = {"render_modes": []}

    JOINT_NAMES = ['base_to_body_joint', 'body_to_boom_joint',
                   'boom_to_arm_joint', 'arm_to_bucket_joint']
    ROS_OFFSET = np.array([0.0, 0.765, -0.743, -0.05])

    def __init__(self, node=None, action_max: float = 0.01,
                 reward_scale: float = 100.0, max_steps: int = 3000,
                 home_tol: float = 0.05, home_max_iters: int = 600,
                 r_pipe: float = 0.2, margin: float = 0.3,
                 d_safe: float = 1.2, w_pipe: float = 0.5, w_hit: float = 100.0):
        super().__init__()
        import rclpy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64, Bool

        self.action_max = action_max
        self.reward_scale = reward_scale
        self.max_steps = max_steps
        self.home_tol = home_tol
        self.home_max_iters = home_max_iters
        # ── 管线安全参数 (对齐离线 excavator_env, 见 ROADMAP.md D2) ──
        self.r_pipe = r_pipe    # 管线半径 m (Unity underground_pipe Cylinder r=0.2)
        self.margin = margin    # 安全裕度 m
        self.d_safe = d_safe    # 软惩罚半径 m (离线 w_pipe 同款)
        self.w_pipe = w_pipe    # 软惩罚权重
        self.w_hit = w_hit      # 撞管硬惩罚权重 (d_pipe < r_pipe+margin 时)
        # ── 倒土完成判定 (用铲斗内土质量: 质量为 0 才算倒净) ──
        self.dump_empty_mass_thresh = 0.5  # kg, 铲斗内土质量低于此值视为已倒净
        self.dump_timeout = 500            # 倒土最长步数 (10s), 超时强制回位兜底
        self._dump_steps = 0               # DUMP 阶段累计步数
        self._dump_empty_steps = 0         # 连续"倒净"步数 (倒净再回位用)
        self.dump_settle_steps = 20        # 倒净后原地再停 0.4s, 让土落净再转回来
        self._cycle_count = 0              # 已完成的完整作业循环数 (挖→摆→倒)

        # ── ROS 节点 ──
        self._own_node = node is None
        self.node = node if node is not None else rclpy.create_node('unity_excavator_env')
        self._q = None                      # 最新 raw 关节角
        self._base_angvel = np.zeros(3)
        self._soil_volume = 0.0
        self._soil_mass = 0.0               # 铲斗内土质量 kg (getInnerSoilMass)
        self._new_state = False

        self.node.create_subscription(
            JointState, '/unity/joint_states', self._cb_joint, 10)
        self.node.create_subscription(
            Float64, '/unity/soil_volume', self._cb_volume, 10)
        self.node.create_subscription(
            Float64, '/unity/bucket_mass', self._cb_mass, 10)
        self._cmd_pub = self.node.create_publisher(
            JointState, '/unity/joint_command', 10)
        self._reset_pub = self.node.create_publisher(
            Bool, '/unity/reset_terrain', 10)

        self.stepper = APFStepper()
        self.home_q_raw = get_q_at_s(0.0) - self.ROS_OFFSET
        self._action_prev = np.zeros(4, dtype=np.float32)
        self._step_count = 0

        # ── Observation space: 17-dim (与离线 excavator_env 一致) ──
        self.observation_space = spaces.Box(
            low=np.array([
                -np.pi, -0.9, -3.0, -3.0,     # q
                3.0, -1.0, -5.0,               # p_tip
                0.0, 0.0, -2.0,                # s, d_pipe, α_margin
                -1.0, -1.0, -1.0,              # base_angvel
                -0.1, -0.1, -0.1, -0.1,        # action_prev
            ], dtype=np.float32),
            high=np.array([
                +np.pi, +1.2, +0.0, +1.0,
                10.0, +1.0, +10.0,
                1.0, 20.0, +3.0,
                +1.0, +1.0, +1.0,
                +0.1, +0.1, +0.1, +0.1,
            ], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=-action_max, high=action_max, shape=(4,), dtype=np.float32)

    # ────────────────────────────────────────────────
    #  ROS 回调
    # ────────────────────────────────────────────────
    def _cb_joint(self, msg):
        q = self._parse_q(msg)
        if q is not None:
            self._q = q
            self._new_state = True

    def _cb_volume(self, msg):
        self._soil_volume = float(msg.data)

    def _cb_mass(self, msg):
        self._soil_mass = float(msg.data)

    def _parse_q(self, msg) -> np.ndarray | None:
        if len(msg.position) < 4:
            return None
        q = np.zeros(4)
        for i, name in enumerate(self.JOINT_NAMES):
            try:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
            except ValueError:
                return None
        if len(msg.velocity) >= 4:
            self._base_angvel = np.array([float(x) for x in msg.velocity[1:4]])
        return q - self.ROS_OFFSET

    # ────────────────────────────────────────────────
    #  同步等待 / 发布
    # ────────────────────────────────────────────────
    def _publish_cmd(self, q_des_raw: np.ndarray):
        import rclpy
        from sensor_msgs.msg import JointState
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = (q_des_raw + self.ROS_OFFSET).tolist()
        self._cmd_pub.publish(msg)

    def _wait_for_state(self, timeout: float = 2.0) -> bool:
        """自旋等待下一帧关节状态, 返回是否在超时前等到."""
        import rclpy
        self._new_state = False
        start = time.monotonic()
        while not self._new_state and time.monotonic() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self._new_state

    def _home(self) -> None:
        """比例控制把 4 个关节拉回起点构型 q(s=0) (回位段, 无 RL)."""
        for _ in range(self.home_max_iters):
            if self._q is None:
                self._wait_for_state()
                continue
            err = self.home_q_raw - self._q
            if np.all(np.abs(err) < self.home_tol):
                break
            dq = np.clip(err, -0.05, 0.05)
            self._publish_cmd(self._q + dq)
            self._wait_for_state()

    # ────────────────────────────────────────────────
    #  Gymnasium 接口
    # ────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        q = self._q if self._q is not None else np.zeros(4)
        p_tip = self.stepper.p_tip if self.stepper.p_tip is not None \
            else np.zeros(3)
        return np.array([
            *q.astype(np.float32),
            *p_tip.astype(np.float32),
            np.float32(self.stepper.s_star),
            np.float32(self.stepper.d_pipe),
            np.float32(self.stepper.alpha_margin),
            *self._base_angvel.astype(np.float32),
            *self._action_prev.astype(np.float32),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from std_msgs.msg import Bool

        # 1. 回位 (先出坑, 再重置地形, 避免蹭坏新土)
        self._home()

        # 2. 重置地形
        msg = Bool()
        msg.data = True
        self._reset_pub.publish(msg)
        time.sleep(0.5)          # 等 Unity 执行 reset 并清零计数
        self._soil_volume = 0.0
        self._soil_mass = 0.0

        # 3. 重置挖掘段状态
        self.stepper.reset()
        self._action_prev = np.zeros(4, dtype=np.float32)
        self._step_count = 0
        self._dump_steps = 0
        self._dump_empty_steps = 0
        self._cycle_count = 0

        # 4. 等一帧新鲜状态
        self._wait_for_state()

        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(
            -self.action_max, self.action_max)

        q = self._q if self._q is not None else self.home_q_raw
        vol_before = self._soil_volume

        # 1. 挖掘控制一步
        q_des_raw, done = self.stepper.step(q, self._base_angvel, action)
        self._action_prev = action

        # 2. 发布指令
        if not done:
            self._publish_cmd(q_des_raw)

        # 3. 等下一帧关节状态
        self._wait_for_state()

        # 4. 倒土完成判定: DUMP 阶段且斗已倒空(或超时) → 进入回位阶段
        vol_after = self._soil_volume
        phase = self.stepper.phase
        if phase == CyclePhase.DUMP:
            self._dump_steps += 1
            # 判定: 铲斗内土质量倒到 0, 并连续停留片刻 → 才回位(转回来).
            # 不要求铲斗转到全开角 —— ZMP 倾覆保护会自然限制开斗幅度, 倒净即回.
            if self._soil_mass < self.dump_empty_mass_thresh:
                self._dump_empty_steps += 1
            else:
                self._dump_empty_steps = 0
            if (self._dump_empty_steps >= self.dump_settle_steps
                    or self._dump_steps > self.dump_timeout):
                self.stepper.start_return()
                self._dump_steps = 0
                self._dump_empty_steps = 0
                self._cycle_count += 1

        # 5. 奖励 = 装土(用铲斗内部土量) − 管线惩罚; 倒土不奖励也不惩罚
        d_pipe = self.stepper.d_pipe
        dv = vol_after - vol_before
        reward = 0.0
        if dv > 0:
            reward += self.reward_scale * dv   # 只奖励装土, 用铲斗内部土量(非堆土量)
        reward -= self.w_pipe * max(0.0, self.d_safe - d_pipe)
        if d_pipe < (self.r_pipe + self.margin):
            reward -= self.w_hit

        # 6. 观测
        obs = self._build_obs()

        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        info = {
            'soil_volume': vol_after,
            'd_pipe': d_pipe,
            'phase': phase.value,
            'dv': max(0.0, dv),          # 本步装土量 m³ (供累计)
            'cycle': self._cycle_count,
        }
        return obs, float(reward), bool(done), bool(truncated), info

    def close(self):
        if self._own_node:
            self.node.destroy_node()
