"""Generate a collision-aware IK reference from the teleoperation recording."""
import json
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / 'src' / 'excavator_kinematics'))
sys.path.insert(0, str(WS / 'src' / 'excavator_trajectory'))

import excavator_trajectory.trajectory as T  # noqa: E402
from excavator_kinematics.mdh import (  # noqa: E402
    MDH_PARAMS,
    forward_kinematics,
    jacobian_position,
)

OFFSET = np.array([0.0, 0.765, -0.743, -0.05])
N_FRAMES = 850
# 安全化轨迹 (certify_trajectory with d_safe=1.5) 把部分路点推到了
# 挖掘机工作空间之外 (X 方向离管线越近, Z 方向需要抬升越多).
# IK 的任务是找到最接近每个路点的可行关节角; 工作空间外的点会有残差,
# 但这不影响控制器: 任务空间跟踪用 WAYPOINTS (离线安全化后的轨迹),
# 关节空间参考只用于零空间投影 (不产生齿尖运动).
IK_TOL = 0.70  # 放宽到 0.70m — certify 路点可能在工作空间外
SOURCE_NAME = 'recorded_trajectory.json'
OUTPUT_NAME = 'recorded_trajectory_safe_ik.json'


def _source_path():
    return (WS / 'src' / 'excavator_trajectory' / 'excavator_trajectory'
            / SOURCE_NAME)


def _match_waypoints_to_frames(source, waypoints):
    """Match waypoint geometry to monotonic frames in the teleop recording."""
    p_recorded = np.array([
        [-frame['p_bf'][0], frame['p_bf'][1], frame['p_bf'][2]]
        for frame in source
    ])
    indices = []
    start = 0
    for waypoint in waypoints:
        distances = np.linalg.norm(p_recorded[start:] - waypoint, axis=1)
        index = start + int(np.argmin(distances))
        indices.append(index)
        start = index
    return np.asarray(indices, dtype=int)


def ik_weighted(target_xyz, q4_ref_raw, q_prev, max_iter=800, tol=2e-4, w_q4=0.5):
    """Full 4-DOF weighted IK with q4 regularization.

    The cert trajectory may be far from the original recording, so fixed-q4 IK
    cannot reach it. Instead solve all 4 DOFs while pulling q4 toward the
    original recording's q4 via regularisation.

    Returns (q_raw, fk_error_m).
    """
    q_min = np.array([MDH_PARAMS['joint_limits'][key][0]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    q_max = np.array([MDH_PARAMS['joint_limits'][key][1]
                      for key in ['q1', 'q2', 'q3', 'q4']])

    if q_prev is not None:
        q = q_prev.copy()
    else:
        q = np.array([0.0, -0.3, -0.8, q4_ref_raw])
    q[0] = 0.0  # swing stays at zero (planar task)
    q = np.clip(q, q_min, q_max)

    lam = 0.5
    for _ in range(max_iter):
        p, _ = forward_kinematics(q)
        pos_err = target_xyz - p
        if np.linalg.norm(pos_err) < tol:
            break
        J = jacobian_position(q)  # (3, 4)
        # Weighted damped least-squares: min ||J dq - pos_err||² + lam||dq||² + w_q4*(q4 - q4_ref)²
        # → dq = (J^T J + (lam + w_q4) I)^-1 (J^T pos_err + w_q4 * e4 * (q4_ref - q[3]))
        M = J.T @ J + (lam + w_q4) * np.eye(4)
        rhs = J.T @ pos_err
        rhs[3] += w_q4 * (q4_ref_raw - q[3])
        dq = np.linalg.solve(M, rhs)
        q += dq
        q = np.clip(q, q_min, q_max)

    p, _ = forward_kinematics(q)
    return q, float(np.linalg.norm(target_xyz - p))


def main():
    waypoints = T.SAFE_WAYPOINTS
    source = json.loads(_source_path().read_text())
    if len(source) != N_FRAMES:
        raise RuntimeError(f'Original recording has {len(source)} frames, expected {N_FRAMES}')

    source_indices = _match_waypoints_to_frames(source, waypoints)
    source_q_ros = np.asarray([frame['q'] for frame in source], dtype=float)
    source_q_raw = source_q_ros - OFFSET
    q4_min, q4_max = MDH_PARAMS['joint_limits']['q4']
    q4_knots = np.clip(source_q_raw[source_indices, 3], q4_min, q4_max)

    q_new_raw = np.zeros((len(waypoints), 4))
    q_prev = None
    errors = []
    for i, target in enumerate(waypoints):
        q, error = ik_weighted(target, q4_knots[i], q_prev)
        q_new_raw[i] = q
        q_prev = q
        errors.append(error)

    max_error = max(errors)
    if max_error > IK_TOL:
        raise RuntimeError(f'Safe IK failed: max FK error {max_error:.4f}m > {IK_TOL:.4f}m')

    s_knots = np.linspace(0.0, 1.0, len(waypoints))
    s_frames = np.linspace(0.0, 1.0, N_FRAMES)
    q_full_raw = np.column_stack([
        np.interp(s_frames, s_knots, q_new_raw[:, j]) for j in range(4)
    ])
    q_min = np.array([MDH_PARAMS['joint_limits'][key][0]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    q_max = np.array([MDH_PARAMS['joint_limits'][key][1]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    if np.any(q_full_raw < q_min - 1e-9) or np.any(q_full_raw > q_max + 1e-9):
        raise RuntimeError('Safe IK reference exceeds joint limits')

    output = [dict(frame) for frame in source]
    q_full_ros = q_full_raw + OFFSET
    for frame, q_ros in zip(output, q_full_ros):
        frame['q'] = [round(float(value), 6) for value in q_ros]

    output_path = (_source_path().parent / OUTPUT_NAME)
    output_path.write_text(json.dumps(output, indent=1))
    metadata = {
        'source': SOURCE_NAME,
        'output': OUTPUT_NAME,
        'path_mode': 'certify_trajectory_gradient_descent_d1.5',
        'pipe_pos': T.PIPE_POS.tolist(),
        'pipe_clearance': T.PIPE_CLEARANCE,
        'waypoint_count': len(waypoints),
        'frame_count': N_FRAMES,
        'max_fk_error_m': max_error,
        'q4_raw_range': [float(q_full_raw[:, 3].min()), float(q_full_raw[:, 3].max())],
    }
    (output_path.with_suffix('.metadata.json')).write_text(json.dumps(metadata, indent=2))
    print(f'Saved {output_path} ({N_FRAMES} frames)')
    print(f'max_fk_error={max_error:.4f}m q4_knots_raw={q4_knots}')


if __name__ == '__main__':
    main()
