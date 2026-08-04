import math

import numpy as np

from excavator_controller.apf import joint_limit_torque
from excavator_kinematics.mdh import MDH_PARAMS


def _joint_limits():
    q_min = np.array([MDH_PARAMS['joint_limits'][key][0]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    q_max = np.array([MDH_PARAMS['joint_limits'][key][1]
                      for key in ['q1', 'q2', 'q3', 'q4']])
    return q_min, q_max


def test_q4_upper_limit_generates_opposing_soft_limit_torque():
    q_min, q_max = _joint_limits()
    q = np.zeros(4)
    q[3] = q_max[3]

    tau_limit = joint_limit_torque(q, q_min, q_max)

    assert tau_limit[3] < 0.0
    assert math.isclose(tau_limit[3], -5000.0)


def test_q4_exclusion_preserves_other_joint_limit_protection():
    q_min, q_max = _joint_limits()
    q = np.zeros(4)
    q[1] = q_max[1]
    q[3] = q_max[3]

    tau_limit = joint_limit_torque(q, q_min, q_max)
    raw_q4_limit = tau_limit[3]
    tau_limit[3] = 0.0

    assert tau_limit[1] < 0.0
    assert raw_q4_limit < 0.0
    assert tau_limit[3] == 0.0


def test_q4_command_target_stays_at_physical_limit():
    q_min, q_max = _joint_limits()
    q4_reference = q_max[3] + 0.2

    q4_target = np.clip(q4_reference, q_min[3], q_max[3])

    assert q_min[3] <= q4_target <= q_max[3]
    assert q4_target == q_max[3]
