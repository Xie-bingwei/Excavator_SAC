"""
ZMP (零力矩点) 稳定裕度计算

基于 FK 关节链 + 各连杆质量, 计算整机质心 (CoM) 的地面投影,
进而得到稳定裕度 α_margin (ZMP 到支撑多边形最近边界的距离).

坐标系: base_footprint 系, X 前, Y 左, Z 上.
支撑多边形: 履带接地矩形, X ∈ [-SUP_X, SUP_X] (前后), Y ∈ [-SUP_Y, SUP_Y] (左右).

倾覆判断 (准静态近似):
  ZMP ≈ CoM 地面投影
  α_margin = min(d_front, d_back, d_left, d_right)
  α_margin < α_thresh → 倾覆风险, 需要斥力保护
"""
import math

import numpy as np

from excavator_kinematics.mdh import _rot_y, forward_kinematics

# ── 各连杆质量 (kg) — 从 AGX 实测 ──
MASS = {
    'base': 5000.0,     # 底盘/履带
    'body': 9800.0,     # 上部结构 (swing 以上)
    'boom': 2200.0,     # 动臂
    'arm': 1500.0,      # 斗杆
    'bucket': 1000.0,   # 铲斗
}

# ── 支撑多边形 (m) — 履带接地范围, 可校准 ──
SUP_X = 1.82    # 前后半长
SUP_Y = 1.20    # 左右半宽

# FK 几何常量 (与 mdh.py 标定一致)
_H, _D_BOOM = 1.0403, 0.771
_A_S, _A2, _A3, _TX = 0.6509, 5.7101, 2.915, 1.4592


def _link_frames(q: np.ndarray) -> dict:
    """
    沿 FK 关节链计算各连杆的质心位置 (base_footprint 系, 臂朝 -X 方向).

    质心近似: 各连杆质心取其几何中点 (boom/arm/bucket 沿臂方向).
    这是工程近似; 若需精确, 可用 AGX 实测质心偏移校准.

    Args:
        q: 关节角 [q1,q2,q3,q4] (rad, raw 空间)

    Returns:
        dict: 各连杆质心 base 系坐标
    """
    q1, q2, q3, q4 = q
    off2, off3 = math.pi / 4, -math.pi / 4
    th2, th3, th4 = q2 + off2, q3 + off3, q4

    # swing 旋转 (绕 Z, 方向取反与 FK 一致)
    cz, sz = math.cos(-q1), math.sin(-q1)
    Rsw = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = Rsw.copy()
    p = np.zeros(3)
    frames = {}

    # 底盘质心 ≈ 原点附近 (base_link cm 实测 (0.52, 0, 0))
    frames['base'] = np.array([0.52, 0.0, 0.0])

    # body: base → body_link (0,0,H)
    p = p + R @ np.array([0, 0, _H])
    frames['body'] = p.copy()

    # boom: body → boom 铰 (-a_s, 0, d_boom), 再沿 -X 伸
    p = p + R @ np.array([-_A_S, 0, _D_BOOM])
    R = R @ _rot_y(th2)[:3, :3]
    arm_o = p + R @ np.array([-_A2, 0, 0])
    frames['boom'] = (p + arm_o) / 2

    # arm: arm 铰 → bucket 铰
    R = R @ _rot_y(th3)[:3, :3]
    bkt_o = arm_o + R @ np.array([-_A3, 0, 0])
    frames['arm'] = (arm_o + bkt_o) / 2

    # bucket: bucket 铰 → tip
    R = R @ _rot_y(th4)[:3, :3]
    tip = bkt_o + R @ np.array([-_TX, 0, 0])
    frames['bucket'] = (bkt_o + tip) / 2

    return frames


def compute_com(q: np.ndarray) -> np.ndarray:
    """
    整机质心 (CoM) 在 base 系中的位置.

    坐标约定: 车前方 = +X (ZMP/倾覆评估用). FK 链原始臂朝 -X, 故翻转 X.

    Args:
        q: 关节角 [q1,q2,q3,q4] (raw 空间)

    Returns:
        com: (3,) 整机质心 base 系坐标
    """
    frames = _link_frames(q)
    total = sum(MASS.values())
    com = sum(MASS[k] * frames[k] for k in MASS) / total
    # 翻转 X: 使车前方为 +X (与 FK 输出/轨迹空间一致)
    com = np.array([-com[0], com[1], com[2]])
    return com


def zmp_alpha_margin(q: np.ndarray) -> tuple[float, np.ndarray]:
    """
    ZMP 稳定裕度 α_margin 及 ZMP 位置.

    准静态近似: ZMP ≈ CoM 地面投影 (忽略动态).
    稳定裕度 = ZMP 到支撑多边形最近边界的距离.

    Args:
        q: 关节角 (raw 空间)

    Returns:
        alpha_margin: ZMP 到最近边界距离 (m), 负值=已越界
        zmp: (2,) ZMP 地面位置 [x, y] (base 系)
    """
    com = compute_com(q)
    x_zmp, y_zmp = com[0], com[1]

    d_front = SUP_X - x_zmp      # ZMP 到前边界
    d_back = x_zmp + SUP_X       # 到后边界
    d_left = SUP_Y - y_zmp       # 到左边界 (y+ 为左)
    d_right = y_zmp + SUP_Y      # 到右边界

    alpha = min(d_front, d_back, d_left, d_right)
    return alpha, np.array([x_zmp, y_zmp])


if __name__ == '__main__':
    # 自检: 零位与几个构型
    for label, q in [
        ('零位', [0, 0, 0, 0]),
        ('下压', [0, 0.3, -0.8, 0.4]),
        ('挖掘底', [0, -0.5, -0.65, -0.6]),
        ('提升', [0, -0.4, -0.9, -1.0]),
    ]:
        alpha, zmp = zmp_alpha_margin(np.array(q))
        print(f'{label}: ZMP=({zmp[0]:+.2f},{zmp[1]:+.2f}) α_margin={alpha:.3f}m')
