import json
from pathlib import Path

import numpy as np

from excavator_kinematics.mdh import MDH_PARAMS, forward_kinematics
from excavator_trajectory.q4_schedule import q4_schedule_raw
from excavator_trajectory.trajectory import WAYPOINTS, get_q_at_s


PACKAGE = Path(__file__).resolve().parents[1] / 'excavator_trajectory'


def test_p3_schedule_knots():
    q4 = q4_schedule_raw(len(WAYPOINTS))
    assert np.allclose(q4[:41], 0.443)
    assert np.allclose(q4[41:48], [0.360, 0.200, 0.000, -0.120, -0.250, -0.450, -0.650])
    assert np.allclose(q4[48:], -0.725)
    assert np.all(np.diff(q4[40:]) <= 0.0)


def test_p3_artifact_shape_limits_and_fk():
    artifact = PACKAGE / 'recorded_trajectory_ik.json'
    data = json.loads(artifact.read_text())
    assert len(data) == 850

    q_ros = np.asarray([frame['q'] for frame in data], dtype=float)
    offset = np.array([0.0, 0.765, -0.743, -0.05])
    q_raw = q_ros - offset
    q_min = np.array([MDH_PARAMS['joint_limits'][key][0]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    q_max = np.array([MDH_PARAMS['joint_limits'][key][1]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    assert np.all(np.isfinite(q_raw))
    assert np.all(q_raw >= q_min - 1e-6)
    assert np.all(q_raw <= q_max + 1e-6)

    errors = []
    for i, target in enumerate(WAYPOINTS):
        q = get_q_at_s(i / (len(WAYPOINTS) - 1)) - offset
        tip, _ = forward_kinematics(q)
        errors.append(np.linalg.norm(tip - target))
    assert max(errors) <= 0.02
