"""
由 TipTrajectoryRecorder 录制、距离阈值 0.4m 降采样得到。
坐标系: FK Σ0 (X=前方, Y=右方, Z=上方)
"""
import numpy as np
import json as _json
import math
import math as _math
import os
from pathlib import Path as _Path

WAYPOINTS = np.array([
    # 下降段 (12)
    [ 8.389, 0.0,  4.375],
    [ 8.513, 0.0,  3.965],
    [ 8.615, 0.0,  3.549],
    [ 8.695, 0.0,  3.129],
    [ 8.753, 0.0,  2.705],
    [ 8.789, 0.0,  2.278],
    [ 8.802, 0.0,  1.850],
    [ 8.793, 0.0,  1.422],
    [ 8.761, 0.0,  0.995],
    [ 8.707, 0.0,  0.570],
    [ 8.631, 0.0,  0.148],
    [ 8.532, 0.0, -0.269],
    # 挖掘弧 (41 路点, 每 5 帧, 密集采样捕获铲斗卷曲)
    [ 8.557, 0.0, -0.173],
    [ 8.515, 0.0, -0.333],
    [ 8.470, 0.0, -0.490],
    [ 8.423, 0.0, -0.647],
    [ 8.372, 0.0, -0.802],
    [ 8.318, 0.0, -0.955],
    [ 8.262, 0.0, -1.106],
    [ 8.203, 0.0, -1.256],
    [ 8.165, 0.0, -1.347],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345],
    [ 8.127, 0.0, -1.380],
    [ 8.059, 0.0, -1.435],
    [ 7.988, 0.0, -1.485],
    [ 7.914, 0.0, -1.531],
    [ 7.837, 0.0, -1.573],
    [ 7.758, 0.0, -1.610],
    [ 7.676, 0.0, -1.642],
    [ 7.593, 0.0, -1.670],
    [ 7.509, 0.0, -1.692],
    [ 7.423, 0.0, -1.709],
    [ 7.336, 0.0, -1.721],
    [ 7.249, 0.0, -1.728],
    [ 7.162, 0.0, -1.729],
    [ 7.075, 0.0, -1.725],
    [ 6.988, 0.0, -1.717],
    [ 6.901, 0.0, -1.702],
    [ 6.816, 0.0, -1.683],
    [ 6.732, 0.0, -1.659],
    [ 6.650, 0.0, -1.629],
    [ 6.569, 0.0, -1.595],
    [ 6.491, 0.0, -1.556],
    [ 6.415, 0.0, -1.513],
    [ 6.342, 0.0, -1.465],
    # 提升 + 回位 (11)
    [ 6.380, 0.0, -0.989],
    [ 6.288, 0.0, -0.596],
    [ 6.168, 0.0, -0.211],
    [ 6.021, 0.0,  0.166],
    [ 5.847, 0.0,  0.533],
    [ 5.647, 0.0,  0.889],
    [ 5.395, 0.0,  1.232],
    [ 5.141, 0.0,  1.542],
    [ 4.919, 0.0,  1.883],
    [ 4.799, 0.0,  2.266],
    [ 4.790, 0.0,  2.328],
])

# Immutable teleoperation geometry. Safety processing never mutates this array.
ORIGINAL_WAYPOINTS = WAYPOINTS.copy()
PIPE_POS = np.array([6.6301, 0.0, -1.2])
PIPE_CLEARANCE = 1.2


def _lift_dangerous_points(path: np.ndarray, pipe_pos: np.ndarray,
                           clearance: float) -> np.ndarray:
    """Lift only dangerous points above the pipe, with local smooth blending."""
    safe = path.copy()
    n = len(path)
    required = np.zeros(n)
    for i, point in enumerate(path):
        dx = point[0] - pipe_pos[0]
        radial = abs(dx)
        if radial < clearance:
            required[i] = pipe_pos[2] + math.sqrt(clearance ** 2 - radial ** 2)
        else:
            required[i] = point[2]
    lift = np.maximum(0.0, required - path[:, 2])
    danger = lift > 1e-9
    if not np.any(danger):
        return safe
    lo, hi = np.where(danger)[0][[0, -1]]
    support_lo = max(0, lo - 3)
    support_hi = min(n - 1, hi + 3)
    for i in range(support_lo, support_hi + 1):
        if support_hi == support_lo:
            envelope = 1.0
        elif i <= lo:
            t = (i - support_lo) / max(1, lo - support_lo)
            envelope = 0.5 - 0.5 * math.cos(math.pi * t)
        elif i >= hi:
            t = (support_hi - i) / max(1, support_hi - hi)
            envelope = 0.5 - 0.5 * math.cos(math.pi * t)
        else:
            envelope = 1.0
        safe[i, 2] = path[i, 2] + envelope * lift[i]
    return safe


# 2026-08-04: _lift_dangerous_points 仅沿 Z 方向抬升路点, 保持 X 不变.
# 安全化后做局部平滑, 消除录制轨迹中下降→挖掘过渡段的向上拐折
# (WP11→12: z=-0.269→-0.173↑). 该拐折导致 carrot target 高于齿尖,
# 引力往上拉 → 臂被拉向机身方向, 挖掘段卡住+震荡.
SAFE_WAYPOINTS = _lift_dangerous_points(ORIGINAL_WAYPOINTS, PIPE_POS,
                                         PIPE_CLEARANCE)
# 局部平滑: 对前 15 个路点做 2 轮高斯平滑, 消除向上拐折而不改变深挖段.
_pipe_x, _pipe_z = PIPE_POS[0], PIPE_POS[2]
for _ in range(3):
    for _i in range(1, min(18, len(SAFE_WAYPOINTS) - 1)):
        _smoothed = 0.5 * SAFE_WAYPOINTS[_i] + 0.25 * SAFE_WAYPOINTS[_i - 1] + 0.25 * SAFE_WAYPOINTS[_i + 1]
        # 保证平滑后不进入管线危险区 (d ≥ 1.0m)
        _dx = _smoothed[0] - _pipe_x
        _dz = _smoothed[2] - _pipe_z
        _d = math.sqrt(_dx * _dx + _dz * _dz)
        if _d >= 1.0:
            SAFE_WAYPOINTS[_i] = _smoothed
WAYPOINTS = SAFE_WAYPOINTS


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    3D 空间中点 p 到线段 ab 的最近点。

    Args:
        p: 查询点 [x, y, z]
        a: 线段起点
        b: 线段终点

    Returns:
        closest: 最近点坐标
        dist: 最短距离
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a.copy(), float(np.linalg.norm(p - a)), 0.0
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    dist = float(np.linalg.norm(p - closest))
    return closest, dist, float(t)


def find_closest(p_tip: np.ndarray) -> tuple[float, np.ndarray]:
    """
    在轨迹上查找距离齿尖最近的 3D 点。

    Args:
        p_tip: 齿尖在 Σ0 系中的位置 [x, y, z], (m)

    Returns:
        s_star: 轨迹进度 ∈ [0, 1]
        p_target: 最近点 3D 坐标
    """
    n_segments = WAYPOINTS.shape[0] - 1

    best_dist = float('inf')
    best_s = 0.0
    best_point = WAYPOINTS[0].copy()

    for i in range(n_segments):
        a = WAYPOINTS[i]
        b = WAYPOINTS[i + 1]
        closest, dist, t = closest_point_on_segment(p_tip, a, b)
        if dist < best_dist:
            best_dist = dist
            best_s = (i + t) / n_segments
            best_point = closest

    return best_s, best_point


def find_closest_continuous(p_tip: np.ndarray, s_prev: float,
                            max_step: float = 0.08,
                            k_fwd: float = 8.0,
                            back_tol: float = 0.05) -> tuple[float, np.ndarray]:
    """
    带连续性的最近点搜索 — 防止相邻轨迹段之间跳变。

    全局 find_closest 在齿尖下压穿过地面附近时, 下降段(近)与挖掘段(深)的
    最近点距离接近, 会发生 s 从 0.14 → 0.56 的跳变, 导致 q4 卷斗参考跳过
    整个挖掘段 → 铲斗挖土时不动.

    2026-08-03 修复: 新安全化轨迹在挖掘弧存在几何回折 (x≈7.7 竖直段), 齿尖在
    挖掘深处 z<-0.5 时, 全局最近点天然偏向回折段 (几何更近), 线性惩罚挡不住
    (距离差 ~0.5m > 惩罚 3.0×(0.70-0.25-0.08)=1.11m). 改为:
      1. 前向二次惩罚 k_fwd·(Δs)²: 远超 max_step 的前向段产生二次代价, 阻止跳变
      2. 回退容差 back_tol: s 最多回退 0.05, 防止挖掘段反向跳变；一次作业模式
         在终点满足完成条件后进入 done，不会将 s 复位为下一循环的起点
    已验证: 理想跟踪下 s 单调递增, 0 反向跳变.

    Args:
        p_tip: 齿尖位置
        s_prev: 上一时刻的轨迹进度 (用于连续性约束)
        max_step: 单步允许的最大 s 前进量 (强制渐进)
        k_fwd: 前向/回退超步惩罚系数 (m per Δs²)
        back_tol: 允许的最大 s 回退量

    Returns:
        s_star, p_target
    """
    n_segments = WAYPOINTS.shape[0] - 1

    best_dist = float('inf')
    best_s = s_prev if s_prev is not None else 0.0
    best_point = WAYPOINTS[int(best_s * n_segments)].copy()

    for i in range(n_segments):
        a = WAYPOINTS[i]
        b = WAYPOINTS[i + 1]
        closest, dist, t = closest_point_on_segment(p_tip, a, b)
        seg_s = i / n_segments
        if s_prev is not None:
            # 前向超步: 二次惩罚 (挡几何回折导致的远距离大步跳变)
            if seg_s > s_prev + max_step:
                dist += k_fwd * (seg_s - s_prev - max_step) ** 2
            # 回退超容差: 二次惩罚 (挖掘段禁止反向跳变)
            elif seg_s < s_prev - back_tol:
                dist += k_fwd * (s_prev - seg_s) ** 2
        if dist < best_dist:
            best_dist = dist
            best_s = (i + t) / n_segments
            best_point = closest

    return best_s, best_point


def find_closest_bounded(p_tip: np.ndarray, s_min: float, s_max: float):
    """
    有界搜索: 仅在 [s_min*segs, s_max*segs] 窗口内找最近点。
    防止 find_closest 在弯曲轨迹上跳到不相邻路段 → 控制器振荡。

    Args:
        p_tip: 齿尖 Σ0 坐标 [x, y, z]
        s_min: 搜索起始进度 ∈ [0,1]
        s_max: 搜索终止进度 ∈ [0,1]

    Returns:
        s_star, p_target
    """
    n_segments = WAYPOINTS.shape[0] - 1
    i_min = max(0, int(s_min * n_segments))
    i_max = min(n_segments, int(s_max * n_segments) + 1)

    best_dist = float('inf')
    best_s = s_min
    best_point = WAYPOINTS[i_min].copy()

    for i in range(i_min, i_max):
        a = WAYPOINTS[i]; b = WAYPOINTS[i + 1]
        closest, dist, t = closest_point_on_segment(p_tip, a, b)
        if dist < best_dist:
            best_dist = dist; best_s = (i + t) / n_segments; best_point = closest

    return best_s, best_point


def certify_trajectory(pipe_pos: np.ndarray, d_safe: float,
                       lam: float = 0.3, max_iter: int = 200):
    """
    离线轨迹安全化 — 梯度下降最小化复合代价泛函 J(ξ) = J_obs + λ·J_smooth。

    J_obs(p_k)   = Σ_k max(0, d_safe - d(p_k))²      ← 障碍物接近代价
    J_smooth(ξ)  = Σ_k ||p_k - (p_{k-1}+p_{k+1})/2||² ← 轨迹不平滑代价

    Args:
        pipe_pos: 管线 Σ0 坐标 [x, y, z]
        d_safe:   安全距离, m (默认 0.8, 但实践中建议 1.5)
        lam:      平滑正则系数 λ (默认 0.3)
        max_iter: 最大迭代次数 (默认 200)

    Returns:
        safe_wps: 安全化后的路点 numpy array, shape (N, 3)
        history:  每轮迭代的 (J_obs, J_smooth, J_total) 列表, 用于收敛分析
    """
    safe = WAYPOINTS.copy()
    n = safe.shape[0]  # 路点总数64
    history = []       # 用于存放每一轮迭代的J值

    for _iter in range(max_iter):
        moved = False

        # ── Phase 1: 显式障碍物梯度下降 ──
        # ∇J_obs(p_k) = -2·(d_safe - d) · (p_k - p_pipe) / d
        # 更新: p_k ← p_k - α · ∇J_obs,  α = 0.25 / d_safe (自适应步长)
        for i in range(n):
            dx = safe[i, 0] - pipe_pos[0]
            dz = safe[i, 2] - pipe_pos[2]
            d = _math.sqrt(dx * dx + dz * dz)

            if d < d_safe and d > 1e-6:
                grad = -2.0 * (d_safe - d) * np.array([dx / d, 0.0, abs(dz) / d])
                alpha = 0.25 / d_safe
                safe[i] = safe[i] - alpha * grad
                # 对 Z 分量取绝对值确保向上推 (管线在地下)
                if safe[i, 2] < WAYPOINTS[i, 2]:
                    safe[i, 2] = WAYPOINTS[i, 2] + 0.5 * (safe[i, 2] - WAYPOINTS[i, 2])
                moved = True

        # ── Phase 2: 显式平滑梯度下降 ──
        # ∇J_smooth(p_k) = 2·(p_k - (p_{k-1} + p_{k+1}) / 2)
        # 更新: p_k ← p_k - α·λ · ∇J_smooth(p_k)
        for i in range(1, n - 1):
            avg = (safe[i - 1] + safe[i + 1]) * 0.5
            grad_s = 2.0 * (safe[i] - avg)
            safe[i] = safe[i] - (0.25 / d_safe) * lam * grad_s

        # ── 计算显式代价值 J(ξ) = J_obs + λ·J_smooth ──
        J_obs = 0.0
        J_smooth = 0.0
        for i in range(n):
            dx = safe[i, 0] - pipe_pos[0]
            dz = safe[i, 2] - pipe_pos[2]
            d = _math.sqrt(dx * dx + dz * dz)
            if d < d_safe:
                J_obs += (d_safe - d) ** 2
            if 1 <= i <= n - 2:
                dev = safe[i] - (safe[i - 1] + safe[i + 1]) * 0.5
                J_smooth += float(np.dot(dev, dev))

        J_total = J_obs + lam * J_smooth
        history.append((J_obs, J_smooth, J_total))

        # 收敛条件: J_obs < 1e-4 或没有路点被推动
        if not moved or J_obs < 1e-4:
            break

    return safe, history


def get_point(s: float) -> np.ndarray:
    """
    轨迹上参数 s ∈ [0, 1] 处的 3D 位置（逐段线性插值）。

    Args:
        s: 弧长参数, 0=起点, 1=终点

    Returns:
        3D 坐标 [x, y, z]
    """
    s = np.clip(s, 0.0, 1.0)
    n_segments = WAYPOINTS.shape[0] - 1
    s_scaled = s * n_segments
    i = int(s_scaled)
    if i >= n_segments:
        return WAYPOINTS[-1].copy()
    t = s_scaled - i
    return WAYPOINTS[i] + t * (WAYPOINTS[i + 1] - WAYPOINTS[i])

# 2026-08-04: 优先加载安全 IK 工件. 若不存在, 回退原始示教 q.
# 安全轨迹 (WAYPOINTS) 用于任务空间跟踪; 关节空间参考 (q) 只需接近即可.
# 原始示教 q 作为回退是安全的: 零空间投影确保它不干扰任务空间安全跟踪.
_q_path = _Path(__file__).parent / "recorded_trajectory_safe_ik.json"
_q_fallback = _Path(__file__).parent / "recorded_trajectory.json"
_q_source = None
if _q_path.exists():
    _q_source = _q_path
elif os.environ.get('TRAJECTORY_GENERATING') == '1':
    _q_source = _q_fallback
else:
    _q_source = _q_fallback   # 回退到原始示教, 不再崩溃
with open(_q_source) as _f:
    _data = _json.load(_f)
_Q_FRAMES = np.array([frm["q"] for frm in _data])  # (N, 4)

# 从录制数据加载参考 q 值 (850 帧 × 4 关节)，供 get_q_at_s 插值
def get_q_at_s(s: float) -> np.ndarray:
    """
    轨迹上参数 s ∈ [0, 1] 处的参考关节角度 (线性插值)。

    Args:
        s: 弧长参数, 0=起点, 1=终点

    Returns:
        q_ref: 4 关节角度 [swing, boom, arm, bucket], rad
    """
    s = np.clip(s, 0.0, 1.0)
    n = _Q_FRAMES.shape[0] - 1
    idx = s * n
    i = int(idx)
    if i >= n:
        return _Q_FRAMES[-1].copy()
    t = idx - i
    return _Q_FRAMES[i] + t * (_Q_FRAMES[i + 1] - _Q_FRAMES[i])


# 2026-08-04: q4 卷斗不再由 _Q_FRAMES 覆写控制, 改为 control.py 运行时
# 根据齿尖 Z 坐标触发 (入土即卷). 保留原始示教 q 作为关节空间参考,
# 零空间投影确保它不干扰任务空间跟踪. q4 由 Z-based 逻辑独立驱动.
