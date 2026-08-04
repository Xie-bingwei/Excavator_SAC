#!/usr/bin/env python3
"""
保存论文绘图所需数据，无需跑仿真。

输出文件 (保存到 ../data/):
  - traj_original.npy       原始示教路点 (64, 3)
  - traj_safe.npy            安全化后路点 (64, 3)
  - safety_history.npy       代价收敛历史 (N_iter, 3) 列: J_obs, J_smooth, J_total
  - pipe_info.npy            管线信息 [x, y, z, d_safe, lam, max_iter]
  - fig_traj_meta.json       轨迹图元信息 (方便画图脚本读取)

用法:
  cd /home/user/xie/AGX_Project/ex_ws
  python3 tools/save_plot_data.py
"""

import sys
import os
import json
import numpy as np

# ── 硬编码原始路点 (从 trajectory.py 源码提取, 避免 import 时被覆盖) ──
WAYPOINTS_ORIGINAL = np.array([
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


def certify_trajectory(waypoints: np.ndarray, pipe_pos: np.ndarray,
                       d_safe: float, lam: float = 0.3,
                       max_iter: int = 200, alpha_factor: float = 0.25):
    """
    与 trajectory.py 中完全一致的算法。
    额外返回原始路点不被覆盖。
    """
    import math

    safe = waypoints.copy()
    n = safe.shape[0]
    history = []

    for _iter in range(max_iter):
        moved = False

        # Phase 1: 障碍物梯度下降
        for i in range(n):
            dx = safe[i, 0] - pipe_pos[0]
            dz = safe[i, 2] - pipe_pos[2]
            d = math.sqrt(dx * dx + dz * dz)

            if d < d_safe and d > 1e-6:
                grad = -2.0 * (d_safe - d) * np.array([dx / d, 0.0, abs(dz) / d])
                alpha = alpha_factor / d_safe
                safe[i] = safe[i] - alpha * grad
                if safe[i, 2] < waypoints[i, 2]:
                    safe[i, 2] = waypoints[i, 2] + 0.5 * (safe[i, 2] - waypoints[i, 2])
                moved = True

        # Phase 2: 平滑梯度下降
        for i in range(1, n - 1):
            avg = (safe[i - 1] + safe[i + 1]) * 0.5
            grad_s = 2.0 * (safe[i] - avg)
            safe[i] = safe[i] - (alpha_factor / d_safe) * lam * grad_s

        # 计算代价值 J(ξ) = J_obs + λ·J_smooth
        J_obs = 0.0
        J_smooth = 0.0
        for i in range(n):
            dx = safe[i, 0] - pipe_pos[0]
            dz = safe[i, 2] - pipe_pos[2]
            d = math.sqrt(dx * dx + dz * dz)
            if d < d_safe:
                J_obs += (d_safe - d) ** 2
            if 1 <= i <= n - 2:
                dev = safe[i] - (safe[i - 1] + safe[i + 1]) * 0.5
                J_smooth += float(np.dot(dev, dev))

        J_total = J_obs + lam * J_smooth
        history.append((J_obs, J_smooth, J_total))

        if not moved or J_obs < 1e-4:
            break

    return safe, history


def main():
    # ── 配置 ──
    # ── 输出目录 ──
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)

    # ── 保存原始路点 (只写一次) ──
    np.save(os.path.join(out_dir, "traj_original.npy"), WAYPOINTS_ORIGINAL)

    # ── 跑三组 d_safe，覆盖论文消融实验 ──
    pipe_pos = np.array([7.5, 0.0, -1.2])
    lam = 0.3
    max_iter = 200

    for d_safe in [0.5, 1.0, 1.5]:
        print(f"\n{'='*60}")
        print(f"Running with d_safe={d_safe}, λ={lam}…")
        safe_wps, history = certify_trajectory(WAYPOINTS_ORIGINAL, pipe_pos, d_safe, lam, max_iter)
        print(f"  Converged in {len(history)} iterations")
        print(f"  Final J_obs={history[-1][0]:.6f}, J_smooth={history[-1][1]:.6f}, J_total={history[-1][2]:.6f}")

        # 保存每组
        tag = f"d{d_safe}"
        np.save(os.path.join(out_dir, f"traj_safe_{tag}.npy"), safe_wps)
        np.save(os.path.join(out_dir, f"safety_history_{tag}.npy"), np.array(history))

        # 统计
        orig_dist = np.sqrt((WAYPOINTS_ORIGINAL[:, 0] - pipe_pos[0])**2 +
                            (WAYPOINTS_ORIGINAL[:, 2] - pipe_pos[2])**2)
        safe_dist = np.sqrt((safe_wps[:, 0] - pipe_pos[0])**2 +
                            (safe_wps[:, 2] - pipe_pos[2])**2)
        n_unsafe_orig = int(np.sum(orig_dist < d_safe))
        n_unsafe_safe = int(np.sum(safe_dist < d_safe))
        print(f"  原始不安全路点数: {n_unsafe_orig}  →  安全化后: {n_unsafe_safe}")
        print(f"  原始 min(d_pipe): {np.min(orig_dist):.3f}m  →  安全化后: {np.min(safe_dist):.3f}m")

    # ── 保存管线 & 参数信息 ──
    np.save(os.path.join(out_dir, "pipe_info.npy"), np.array([*pipe_pos, lam, max_iter]))

    # ── 元信息 (以 d_safe=1.5 为默认) ──
    safe_wps_default, history_default = certify_trajectory(WAYPOINTS_ORIGINAL, pipe_pos, 1.5, lam, max_iter)
    meta = {
        "description": "CBF轨迹安全化数据 (三组d_safe)",
        "n_waypoints": int(WAYPOINTS_ORIGINAL.shape[0]),
        "pipe_pos": pipe_pos.tolist(),
        "lambda": lam,
        "runs": {
            "d0.5": {"d_safe": 0.5, "n_iters": np.load(os.path.join(out_dir, "safety_history_d0.5.npy")).shape[0]},
            "d1.0": {"d_safe": 1.0, "n_iters": np.load(os.path.join(out_dir, "safety_history_d1.0.npy")).shape[0]},
            "d1.5": {"d_safe": 1.5, "n_iters": np.load(os.path.join(out_dir, "safety_history_d1.5.npy")).shape[0]},
        },
        "files": {
            "traj_original":      "traj_original.npy         — shape (64,3)",
            "traj_safe_d0.5":     "traj_safe_d0.5.npy        — shape (64,3)",
            "traj_safe_d1.0":     "traj_safe_d1.0.npy        — shape (64,3)",
            "traj_safe_d1.5":     "traj_safe_d1.5.npy        — shape (64,3)",
            "safety_history_d*":  "safety_history_d*.npy     — shape (N_iter,3) cols=[J_obs, J_smooth, J_total]",
            "pipe_info":          "pipe_info.npy             — [pipe_x, pipe_y, pipe_z, lam, max_iter]",
        }
    }
    with open(os.path.join(out_dir, "fig_traj_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ── 打印摘要 ──
    print(f"\n{'='*60}")
    print(f"All saved to {out_dir}/:")
    for fname in ["traj_original.npy", "traj_safe_d0.5.npy", "traj_safe_d1.0.npy",
                  "traj_safe_d1.5.npy", "safety_history_d0.5.npy",
                  "safety_history_d1.0.npy", "safety_history_d1.5.npy",
                  "pipe_info.npy", "fig_traj_meta.json"]:
        fpath = os.path.join(out_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<30s}  {size:>8d} bytes")


if __name__ == "__main__":
    main()
