import math
import numpy as np

# MDH 参数表
# 变换顺序: Rot_X(α_{i-1}) · Trans_X(a_{i-1}) · Rot_Z(θ_i) · Trans_Z(d_i)
#
# Joint 链:
#   base_footprint ──[Trans(0,0,H)]──> swing(body) ──> boom ──> arm ──> bucket ──> tip
#
# H     - 回转关节 (swing) 距地面高度 (m)
# alpha - 连杆扭转角 (rad): 绕 x_{i-1} 旋转，使 z_{i-1} 与 z_i 对齐
# a     - 连杆长度 (m):     沿 x_{i-1} 方向，从 z_{i-1} 到 z_i 的距离
# d     - 连杆偏距 (m):     沿 z_i 方向，从 x_{i-1} 到 x_i 的距离

MDH_PARAMS = {
    # 回转关节距地面高度
    # H = base_footprint → body_link 高度 (FK 标定值, 与 AGX 场景一致)
    'H': 1.0403,

    # 连杆扭转角 α_{i-1} (rad)
    'alpha_i-1_swing':  0,
    'alpha_i-1_boom':   math.pi / 2,        # Z轴 → -Y轴 (水平)
    'alpha_i-1_arm':    0,
    'alpha_i-1_bucket': 0,
    'alpha_i-1_tip':    0,

    # 连杆长度 a_{i-1} (m) 
    'a_i-1_swing':  0,
    'a_i-1_boom':   0.6509,               # 回转轴 → 大臂铰点 水平偏移
    'a_i-1_arm':    5.7101,               # 大臂铰点 → 斗杆铰点 (boom 长度)
    'a_i-1_bucket': 2.915,                # 斗杆铰点 → 铲斗铰点 (arm 长度)
    'a_i-1_tip':    1.4592,               # 铲斗铰点 → 铲斗齿尖

    # 连杆偏距 d_i (m)
    'd_i_swing':  0,                      # swing 的 d 归入 H
    'd_i_boom':   0.771,              # 回转平台 → 大臂铰点 竖直偏移
    'd_i_arm':    0,
    'd_i_bucket': 0,
    'd_i_tip':    0,

    'joint_limits': {
        'q1': [-math.pi, math.pi], 
        'q2': [-0.8033, 1.155],
        'q3': [-2.7594, -0.6084],
        'q4': [-2.614, 0.4433],
    },

    # 铲斗齿尖固定旋转变换
    'bucket_tip_rpy': (0.0, 0.0, 0.0),  # RPY: 绕Y轴下俯 ~39.1°

    # ROS指令的零位偏移，ROS 目标位置 = AGX 关节角度 + offset
    'theta_offset_swing':  0.0,
    'theta_offset_boom':   0.7854,       # = π/4, boom AGX零位→ROS零位的角度偏移
    'theta_offset_arm':   -0.7854,
    'theta_offset_bucket': 0.0,

    # 【FK 标定】关节零点偏置 (rad) — 由 AGX 场景关节树解析 + 实机标定得到
    # 每个关节实际角度 = q_i + joint_offset_i, 旋转轴绕连杆局部 Y 轴
    # 标定点验证: 4 个构型全部误差 0.000 m
    'joint_offset_swing':  0.0,
    'joint_offset_boom':   math.pi / 4,     # +45°
    'joint_offset_arm':   -math.pi / 4,     # -45°
    'joint_offset_bucket': 0.0,
}

def _mdh_T(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """
    MDH齐次变换矩阵
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct,     -st,   0,      a],
        [ca*st, ca*ct, -sa, -d*sa],
        [sa*st, sa*ct, ca,   d*ca],
        [  0,     0,    0,      1]
    ])

def _rot_y(theta: float) -> np.ndarray:
    """绕 Y 轴的旋转矩阵 (关节旋转轴)"""
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct,  0, st, 0],
        [0,   1,  0, 0],
        [-st, 0, ct, 0],
        [0,   0,  0, 1],
    ])


def forward_kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    4-DOF 正运动学 (按 AGX 真实几何重写)

    关节树 (来自 AGX 场景解析 + 4 构型实机标定, 误差 0.000m):
      base_footprint → body_link: (0, 0, 1.0403)         固定
      body_link     → boom_link:  (-0.6509, 0, 0.771)    固定
      boom_link     → arm_link:   (-5.7101, 0, 0)        绕 local Y 转 θ2 = q2 + π/4
      arm_link      → bucket_link:(-2.915, 0, 0)         绕 local Y 转 θ3 = q3 - π/4
      bucket_link   → bucket_tip: (-1.4592, 0, 0)        绕 local Y 转 θ4 = q4
    连杆沿 -X 方向伸出, 关节绕连杆局部 Y 轴旋转 (齿尖在 XZ 平面运动).

    Args:
        q:[q1, q2, q3, q4], (rad)

    Returns:
        p_tip: 齿尖在 base_footprint 系中的位置 [x, y, z], (m)
        d_tip: 齿尖朝向 [dx, dy, dz] (单位向量)
    """
    q1, q2, q3, q4 = q

    a_s = MDH_PARAMS['a_i-1_boom']
    a2 = MDH_PARAMS['a_i-1_arm']
    a3 = MDH_PARAMS['a_i-1_bucket']
    tx = MDH_PARAMS['a_i-1_tip']
    H = MDH_PARAMS['H']           # base_footprint → body_link 高度
    d_boom = MDH_PARAMS['d_i_boom']   # body_link → boom_link 垂直偏移
    off2 = MDH_PARAMS['joint_offset_boom']    # +π/4
    off3 = MDH_PARAMS['joint_offset_arm']     # -π/4

    th2, th3, th4 = q2 + off2, q3 + off3, q4

    # 累计齐次变换: 先绕 swing (Rz(q1)) 再沿链累乘
    # 标定: swing 旋转方向与物理相反, 取 -q1 修正
    T = np.eye(4)
    cz, sz = math.cos(-q1), math.sin(-q1)
    T_swing = np.array([
        [cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = T_swing @ T

    # 固定段: body (0,0,H) + boom 铰 (-a_s, 0, d_boom)
    T = T @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,H],[0,0,0,1]])
    T = T @ np.array([[1,0,0,-a_s],[0,1,0,0],[0,0,1,d_boom],[0,0,0,1]])
    # boom → arm: 绕 local Y 转 θ2, 平移 -a2
    T = T @ _rot_y(th2) @ np.array([[1,0,0,-a2],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # arm → bucket: 绕 local Y 转 θ3, 平移 -a3
    T = T @ _rot_y(th3) @ np.array([[1,0,0,-a3],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # bucket → tip: 绕 local Y 转 θ4, 平移 -tx
    T = T @ _rot_y(th4) @ np.array([[1,0,0,-tx],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    p_tip = T[:3, 3]
    d_tip = T[:3, 0]

    # 输出统一为轨迹坐标系 (WAYPOINTS 空间): 录制时 p_bf 的 X 取负,
    # 故翻转 X 使 FK 输出与轨迹一致
    p_tip = np.array([-p_tip[0], p_tip[1], p_tip[2]])
    d_tip = np.array([-d_tip[0], d_tip[1], d_tip[2]])

    return p_tip, d_tip

def _rot_x(alpha: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [1,  0,   0, 0],
        [0, ca, -sa, 0],
        [0, sa,  ca, 0],
        [0,  0,   0, 1],
    ])


def _trans_x(a: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def jacobian_position(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    位置雅可比 J_p ∈ R^{3×4} (数值微分, 与重写的 FK 一致)

    齿尖位置 p_tip = FK(q), J_p[i,j] = ∂p_i / ∂q_j.
    用中心差分数值求导, 避免手推符号错误.

    Returns:
        J_p: shape (3, 4)
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    J_p = np.zeros((3, n))
    p0, _ = forward_kinematics(q)
    for j in range(n):
        qp = q.copy(); qm = q.copy()
        qp[j] += eps; qm[j] -= eps
        pp, _ = forward_kinematics(qp)
        pm, _ = forward_kinematics(qm)
        J_p[:, j] = (pp - pm) / (2 * eps)
    return J_p


