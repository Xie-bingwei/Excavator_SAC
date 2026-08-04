"""
管线 CBF 安全滤波

用离散时间控制障碍函数 (CBF) 替代 APF 斥力, 在速度空间中施加硬约束。
单个管线约束时 QP 有闭式解, 无需安装优化库。

原理:
  h(p_tip) = d_pipe - d_safe      ← CBF 候选函数, h < 0 表示危险
  ∇h·v ≥ -λ·h/Δt                  ← 离散时间 CBF 约束不等式
  若 v_des 满足约束 → 放行
  若不满足 → 把 v_des 投影到安全半空间边界上 (闭式解)
"""
import numpy as np
import math


def pipe_cbf_from_apf(p_tip: np.ndarray, pipe_pos: np.ndarray, 
                      rho: float = 1.0, K: float = 10000.0, delta: float = 0.05) -> tuple[float, np.ndarray]:
    """
    管线 CBF 候选函数及梯度。

    Args:
        p_tip:    齿尖 Σ0 系位置 [x, y, z], m
        pipe_pos: 管线 Σ0 系位置 [x, y, z], m (管线沿 Y 轴水平, 忽略 Y)
        d_safe:   安全距离, m (法规硬边界, 默认 0.3)

    Returns:
        h:    CBF 值 = d - d_safe, h < 0 进入危险区, m
        grad: dh/dp ∈ R³ (单位向量, 从管线指向齿尖), 无量纲
    """
    # XZ平面距离
    dx = p_tip[0] - pipe_pos[0]
    dz = p_tip[2] - pipe_pos[2]
    d = np.sqrt(dx**2 + dz**2)
    dir = np.array([dx / d, 0.0, abs(dz) / d])

    # APF势能及其对d的导数
    if d < rho:
        diff = 1/d - 1/rho
        u = 0.5 * K * diff**2
        du_dd = -K * diff / (d**2)
    else:
        u = 0.0
        du_dd = 0.0

    # 构造CBF
    h = 1.0 / (1.0 + u) - delta

    # 求梯度
    dh_du = -1.0 / (1.0 + u)**2
    grad = dh_du * du_dd * dir

    return h, grad


def cbf(v_des: np.ndarray, h: float, grad: np.ndarray,
                lam: float = 0.5, dt: float = 0.02) -> np.ndarray:
    """
    把期望齿尖速度投影到 CBF 安全半空间上 (单约束, 闭式解).

    QP: min ||v - v_des||²  s.t. grad·v ≥ -λ·h/dt

    Args:
        v_des: APF 期望齿尖线速度 ∈ R³, m/s
        h:     当前 CBF 值, m
        grad:  dh/dp ∈ R³ (单位向量)
        lam:   收敛率 ∈ (0,1], 默认 0.5
        dt:    控制周期, s (默认 0.02)

    Returns:
        v_safe: 安全齿尖速度 ∈ R³, m/s
    """
    # 计算约束下界
    bound = -lam * h / dt

    # 计算当前投影
    proj = float(np.dot(grad, v_des))

    if proj >= bound:
        return v_des
    if not np.any(grad):
        return v_des

    eta = (bound - proj) / float(np.dot(grad, grad))
    return v_des + eta * grad
