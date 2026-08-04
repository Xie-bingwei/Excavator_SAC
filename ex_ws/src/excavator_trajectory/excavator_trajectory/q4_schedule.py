"""Prescribed bucket-curl schedule for the safety-shaped trajectory."""

import numpy as np

from excavator_kinematics.mdh import MDH_PARAMS

SCHEDULE_VERSION = 'p3-late-curl-v1'
EXPECTED_WAYPOINTS = 64


def q4_schedule_raw(n_waypoints: int) -> np.ndarray:
    """Return raw q4 knots, curling during the underground exit."""
    if n_waypoints != EXPECTED_WAYPOINTS:
        raise ValueError(
            f'P3 schedule expects {EXPECTED_WAYPOINTS} waypoints, got {n_waypoints}'
        )

    q4 = np.full(n_waypoints, 0.443, dtype=float)
    q4[41:48] = [0.360, 0.200, 0.000, -0.120, -0.250, -0.450, -0.650]
    q4[48:] = -0.725

    q4_min, q4_max = MDH_PARAMS['joint_limits']['q4']
    if np.any(q4 < q4_min) or np.any(q4 > q4_max):
        raise ValueError('P3 q4 schedule exceeds physical joint limits')
    if np.any(np.diff(q4[40:]) > 1e-9):
        raise ValueError('P3 q4 schedule must be non-increasing after WP40')
    return q4
