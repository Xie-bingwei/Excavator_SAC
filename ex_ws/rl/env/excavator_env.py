#!/usr/bin/env python3
"""
Gymnasium environment for 4-DOF hydraulic excavator RL.

Architecture
------------
   RL action a_t (Δq) → 叠加到 APF 的 q_des_apf → 关节动力学 → 下一状态

Two modes:
  - offline: kinematic simulator, α_margin computed, base_angvel=0
  - online:  Unity physics via ROS2, real base_angvel from RigidBody

State space (17-dim)
    0-3 : q[4]          joint angles, raw rad
    4-6 : p_tip[3]      齿尖 FK 坐标, m
    7   : s_star        轨迹进度 ∈ [0,1]
    8   : d_pipe        管线 XZ 距离, m
    9   : α_margin      ZMP 稳定裕度, m
  10-12 : base_angvel   底盘角速度, rad/s (offline=0)
  13-16 : action_prev   上一帧 RL 动作, rad

Action space (4-dim)
  Δq ∈ [-0.05, +0.05] rad per joint
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from pathlib import Path
import sys, os

_WS = Path(__file__).resolve().parents[2]
for _pkg in ['excavator_kinematics', 'excavator_trajectory',
             'excavator_controller', 'excavator_control']:
    _p = str(_WS / 'src' / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ['TRAJECTORY_GENERATING'] = '1'

from excavator_kinematics.mdh import (
    MDH_PARAMS, forward_kinematics, jacobian_position,
)
from excavator_trajectory.trajectory import (
    find_closest, find_closest_continuous, get_point, get_q_at_s,
    PIPE_POS, WAYPOINTS,
)
from excavator_controller.apf import (
    attractive_force, attractive_torque, joint_limit_torque, tipover_torque,
)
from excavator_control.reference_progress import advance_reference_progress
from excavator_control.terminal_state import (
    TerminalConfig, TerminalObservation, TerminalPhase, TerminalStateMachine,
)


@dataclass
class ExcavatorEnvConfig:
    """Environment hyperparameters — all tunable."""

    # ── Control gains (mirrors control.py) ──
    K_att: float = 300.0
    K_imp: float = 0.012
    K_joint: float = 60.0
    K_joint_bucket: float = 250.0
    dt: float = 0.02

    # ── Joint limits ──
    q_min: tuple = field(default_factory=lambda: tuple(
        MDH_PARAMS['joint_limits'][k][0] for k in ['q1', 'q2', 'q3', 'q4']))
    q_max: tuple = field(default_factory=lambda: tuple(
        MDH_PARAMS['joint_limits'][k][1] for k in ['q1', 'q2', 'q3', 'q4']))
    ros_offset: tuple = (0.0, 0.765, -0.743, -0.05)

    # ── Trajectory ──
    pipe_pos: tuple = tuple(float(x) for x in PIPE_POS)
    d_safe: float = 1.5
    carrot_lookahead_s: float = 0.05
    carrot_lookahead_star: float = 0.10
    s_ref_step: float = 0.01

    # ── Position delta limits ──
    max_delta_normal: tuple = (0.06, 0.04, 0.05, 0.08)    # s >= 0.68
    max_delta_dig: tuple = (0.06, 0.03, 0.04, 0.08)        # s < 0.68

    # ── Bucket curl (Z-based) ──
    z_curl_entry: float = -0.03
    q4_curl_in_soil: float = -1.4

    # ── RL action bounds ──
    action_max: float = 0.05       # ±0.05 rad per joint

    # ── Terminal config ──
    terminal_entry_s: float = 0.93
    terminal_entry_path_err: float = 0.35
    endpoint_tol: float = 0.30
    q4_tol: float = 0.08
    hold_timeout: float = 8.0

    # ── Episode limits ──
    max_steps: int = 600

    # ── Reward weights (tuned for offline training) ──
    w_progress: float = 15.0       # 轨迹前进 (主要回报源)
    w_cycle: float = 100.0         # 循环完成
    w_soil: float = 80.0           # 土中切卷奖励: 每米 X 后退位移 (兜土)
    w_stability: float = 3.0       # α_margin < 0.5 惩罚
    w_tipover: float = 10.0       # base angular vel 惩罚 (online only)
    w_pipe: float = 10.0           # 管线接近惩罚 (轨迹已安全, 追踪误差是主因)
    w_smooth: float = 0.1          # 动作平滑
    w_magnitude: float = 0.02      # 动作幅度惩罚
    alpha_thresh: float = 0.5

    @property
    def q_min_np(self) -> np.ndarray:
        return np.array(self.q_min)

    @property
    def q_max_np(self) -> np.ndarray:
        return np.array(self.q_max)

    @property
    def ros_offset_np(self) -> np.ndarray:
        return np.array(self.ros_offset)

    @property
    def max_delta_normal_np(self) -> np.ndarray:
        return np.array(self.max_delta_normal)

    @property
    def max_delta_dig_np(self) -> np.ndarray:
        return np.array(self.max_delta_dig)

    @property
    def pipe_pos_np(self) -> np.ndarray:
        return np.array(self.pipe_pos)


class ExcavatorEnv(gym.Env):
    """Gymnasium environment wrapping the APF control pipeline."""

    metadata = {"render_modes": []}
    # We declare observation/action specs here so SB3 can auto-detect them;
    # gymnasium will still require define() in __init__ or their auto-wrap.

    def __init__(self, config: Optional[ExcavatorEnvConfig] = None):
        super().__init__()
        self.cfg = config or ExcavatorEnvConfig()

        # ── Observation space: 17-dim Box ──
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -np.pi, -0.9, -3.0, -3.0,   # q
                3.0, -1.0, -5.0,              # p_tip
                0.0, 0.0, -2.0,               # s, d_pipe, α_margin
                -1.0, -1.0, -1.0,             # base_angvel
                -0.1, -0.1, -0.1, -0.1,       # action_prev
            ], dtype=np.float32),
            high=np.array([
                +np.pi, +1.2, +0.0, +1.0,     # q
                10.0, +1.0, +10.0,             # p_tip
                1.0, 20.0, +3.0,              # s, d_pipe, α_margin
                +1.0, +1.0, +1.0,             # base_angvel
                +0.1, +0.1, +0.1, +0.1,       # action_prev
            ], dtype=np.float32),
        )

        # ── Action space: 4-dim Box ──
        a = self.cfg.action_max
        self.action_space = gym.spaces.Box(
            low=-a, high=+a, shape=(4,), dtype=np.float32,
        )

        self.reset()

    # ────────────────────────────────────────────────────────
    #  Gymnasium interface
    # ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        # Start from q at trajectory start
        self._q = (get_q_at_s(0.0) - self.cfg.ros_offset_np).copy()
        self._q_smooth: Optional[np.ndarray] = None
        self._s_prev: Optional[float] = None
        self._s_ref: Optional[float] = None
        self._action_prev = np.zeros(4, dtype=np.float32)

        # Z-based curl state
        self._z_curl_active = False

        # Terminal state machine
        self._terminal = TerminalStateMachine(TerminalConfig(
            entry_s_threshold=self.cfg.terminal_entry_s,
            entry_path_error_threshold=self.cfg.terminal_entry_path_err,
            endpoint_error_threshold=self.cfg.endpoint_tol,
            q4_error_threshold=self.cfg.q4_tol,
            hold_timeout_sec=self.cfg.hold_timeout,
        ))
        self._q_terminal = np.clip(
            get_q_at_s(1.0) - self.cfg.ros_offset_np,
            self.cfg.q_min_np, self.cfg.q_max_np,
        )

        self._step_count = 0
        self._done = False
        self._cum_reward = 0.0
        self._soil_pull = 0.0        # 土中 X 向机身位移累积 (m, 兜土信号)
        self._p_tip_prev: Optional[np.ndarray] = None

        obs, info = self._build_obs(), {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(
            -self.cfg.action_max, self.cfg.action_max,
        )
        self._step_count += 1

        # ── 1. FK + 土中卷位移 ──
        p_tip, _ = forward_kinematics(self._q)
        # 齿尖在土中 (z < 0): 累计 X 向机身方向的位移 (X↓ = 兜土)
        if self._p_tip_prev is not None and p_tip[2] < 0 and self._p_tip_prev[2] < 0:
            dx = float(self._p_tip_prev[0] - p_tip[0])  # >0 表示向机身拉
            if dx > 0:
                self._soil_pull += dx
        self._p_tip_prev = p_tip.copy()

        # ── 2. Trajectory closest point ──
        if self._terminal.phase == TerminalPhase.TERMINAL_HOLD:
            s_star = self._s_prev or 0.0
        elif self._s_prev is None:
            s_star, _ = find_closest(p_tip)
        else:
            s_star, _ = find_closest_continuous(p_tip, self._s_prev)

        path_err = float(np.linalg.norm(p_tip - get_point(s_star)))
        endpoint_err = float(np.linalg.norm(p_tip - get_point(1.0)))
        self._s_ref = advance_reference_progress(
            self._s_ref, s_star, self.cfg.s_ref_step,
        )

        # ── 3. Terminal state machine ──
        decision = self._terminal.update(
            self._step_count * self.cfg.dt,
            TerminalObservation(
                s_star=s_star, path_error=path_err,
                endpoint_error=endpoint_err, q=self._q,
                q4_terminal_target=self._q_terminal[3],
            ),
        )

        # ── 4. Terminal handling ──
        if self._terminal.phase == TerminalPhase.TERMINAL_HOLD_TIMEOUT:
            self._done = True
            return self._build_obs(), 0.0, False, True, {"reason": "hold_timeout"}

        if decision.reason == 'terminal_complete':
            self._done = True
            reward = self._compute_reward(s_star, path_err, endpoint_err, action, 1.0)
            self._action_prev = action.copy()
            self._s_prev = s_star
            self._cum_reward += reward
            return self._build_obs(), reward, False, True, {"reason": "complete"}

        # ── 5. Compute APF target ──
        s_target = 1.0 if decision.force_terminal_target else min(
            self._s_ref + self.cfg.carrot_lookahead_s,
            s_star + self.cfg.carrot_lookahead_star, 1.0,
        )
        p_target = get_point(s_target)

        # 5a. Task space attraction
        F_att = attractive_force(p_tip, p_target, self.cfg.K_att)
        J_p = jacobian_position(self._q)
        tau_total = attractive_torque(F_att, J_p)
        tau_total[3] = 0.0

        # 5b. Null-space
        q_ref = self._q_terminal if decision.force_terminal_target else (
            get_q_at_s(self._s_ref) - self.cfg.ros_offset_np)
        N = np.eye(4) - J_p.T @ np.linalg.pinv(J_p).T
        tau_null = self.cfg.K_joint * (q_ref - self._q)
        tau_null[3] = 0.0
        tau_total += N @ tau_null

        # 5c. Bucket curl (Z-based)
        if decision.force_terminal_target:
            q4_target = self._q_terminal[3]
        else:
            if not self._z_curl_active and p_tip[2] < self.cfg.z_curl_entry:
                self._z_curl_active = True
            if self._z_curl_active:
                q4_target = (self.cfg.q4_curl_in_soil if p_tip[2] < 0.0
                             else self._q_terminal[3])
            else:
                q4_target = q_ref[3]
        q4_clip = float(np.clip(q4_target, self.cfg.q_min_np[3], self.cfg.q_max_np[3]))
        tau_total[3] = self.cfg.K_joint_bucket * (q4_clip - self._q[3])

        # 5d. Joint limits (q1-q3 only)
        tau_limit = joint_limit_torque(self._q, self.cfg.q_min_np, self.cfg.q_max_np)
        tau_limit[3] = 0.0
        tau_total += tau_limit

        # 5e. ZMP
        tau_tip, alpha_margin = tipover_torque(self._q)
        tau_total += tau_tip

        # 5f. Torque → desired position
        q_des_apf = self._q + self.cfg.K_imp * tau_total * self.cfg.dt

        # ── 6. Apply RL action: q_des = q_des_apf + Δq ──
        q_des = q_des_apf + action

        # ── 7. Position limit + smoothing (simplified physics) ──
        if s_star >= 0.68:
            max_delta = self.cfg.max_delta_normal_np
        else:
            max_delta = self.cfg.max_delta_dig_np

        delta_q = np.clip(q_des - self._q, -max_delta, max_delta)
        q_des = self._q + delta_q

        # q4 direct drive (same as control.py)
        if self._z_curl_active or decision.force_terminal_target:
            q4_err = q4_clip - self._q[3]
            q4_step = 0.10
            delta_q[3] = np.clip(q4_err, -q4_step, +q4_step)

        q_des = self._q + delta_q

        # Low-pass filter
        if self._q_smooth is None:
            self._q_smooth = q_des.copy()
        else:
            self._q_smooth = 0.5 * q_des + 0.5 * self._q_smooth
            if self._z_curl_active or decision.force_terminal_target:
                self._q_smooth[3] = q_des[3]

        q_next = np.clip(self._q_smooth, self.cfg.q_min_np, self.cfg.q_max_np)

        # ── 8. Reward + transition ──
        reward = self._compute_reward(s_star, path_err, endpoint_err, action, 0.0)
        self._action_prev = action.copy()
        self._s_prev = s_star
        self._q = q_next
        self._cum_reward += reward

        # ── 9. Termination: only step limit or task complete ──
        # 不做 d_pipe 早期终止 — 轨迹本身安全 (d≥1.2m), 追踪误差(0.5-0.8m)
        # 会临时让齿尖靠近管线, 但这和真实 control.py 行为一致.
        # 管线安全由奖励函数惩罚 (r_pipe = -100 × max(0, 1.5-d_pipe)).
        d_pipe = float(np.sqrt(
            (p_tip[0] - self.cfg.pipe_pos[0])**2 +
            (p_tip[2] - self.cfg.pipe_pos[2])**2))
        truncated = self._step_count >= self.cfg.max_steps

        obs = self._build_obs()
        info = {
            's_star': s_star, 'path_err': path_err,
            'endpoint_err': endpoint_err, 'd_pipe': d_pipe,
            'alpha_margin': alpha_margin,
            'cum_reward': self._cum_reward,
            'reason': ('truncated' if truncated else ''),
        }
        return obs, reward, False, truncated, info

    def _build_obs(self) -> np.ndarray:
        p_tip, _ = forward_kinematics(self._q)
        s_star = self._s_prev or 0.0
        d_pipe = float(np.sqrt(
            (p_tip[0] - self.cfg.pipe_pos[0])**2 +
            (p_tip[2] - self.cfg.pipe_pos[2])**2))
        from excavator_kinematics.zmp import zmp_alpha_margin
        alpha, _ = zmp_alpha_margin(self._q)

        return np.array([
            *self._q.astype(np.float32),
            *p_tip.astype(np.float32),
            np.float32(s_star),
            np.float32(d_pipe),
            np.float32(alpha),
            np.float32(0.0),  # base_angvel — offline mode = 0
            np.float32(0.0),
            np.float32(0.0),
            *self._action_prev.astype(np.float32),
        ], dtype=np.float32)

    def _compute_reward(
        self, s_star: float, path_err: float, endpoint_err: float,
        action: np.ndarray, cycle_bonus: float,
    ) -> float:
        cfg = self.cfg
        r = 0.0

        # Progress: reward forward movement along trajectory
        s_prev = self._s_prev or 0.0
        r += cfg.w_progress * max(0.0, s_star - s_prev)

        # Cycle complete: bonus + soil pull reward (土中切卷兜土)
        r += cfg.w_cycle * cycle_bonus
        r += cfg.w_soil * self._soil_pull * cycle_bonus

        # Stability: penalize ZMP margin below threshold
        from excavator_kinematics.zmp import zmp_alpha_margin
        alpha, _ = zmp_alpha_margin(self._q)
        r -= cfg.w_stability * max(0.0, cfg.alpha_thresh - alpha)

        # Pipe proximity: penalize approaching the pipeline
        p_tip, _ = forward_kinematics(self._q)
        d_pipe = float(np.sqrt(
            (p_tip[0] - cfg.pipe_pos[0])**2 + (p_tip[2] - cfg.pipe_pos[2])**2))
        r -= cfg.w_pipe * max(0.0, cfg.d_safe - d_pipe)

        # Action smoothness: penalize jerky control
        r -= cfg.w_smooth * float(np.linalg.norm(action - self._action_prev))

        # Action magnitude: slight penalty on large corrections
        r -= cfg.w_magnitude * float(np.linalg.norm(action))

        return r


# ── Quick smoke test ──
if __name__ == '__main__':
    env = ExcavatorEnv()
    print(f"obs_space = {env.observation_space}")
    print(f"act_space = {env.action_space}")
    obs, _ = env.reset()
    print(f"obs[0]    = {obs}")

    total_r, done = 0.0, False
    for i in range(env.cfg.max_steps):
        act = env.action_space.sample() * 0.1  # small random actions
        obs, r, terminated, truncated, info = env.step(act)
        total_r += r
        if terminated or truncated:
            print(f"  end at step {i+1}: reason={info.get('reason', '?')} "
                  f"total_r={total_r:.1f} endpoint_err={info.get('endpoint_err',-1):.3f}")
            break
        if (i + 1) % 100 == 0:
            print(f"  step {i+1:3d}: s={info['s_star']:.3f} "
                  f"path_err={info['path_err']:.3f} "
                  f"d_pipe={info['d_pipe']:.2f} r={total_r:.1f}")

    print(f"Done: {info.get('reason','?')} total_r={total_r:.1f}")
