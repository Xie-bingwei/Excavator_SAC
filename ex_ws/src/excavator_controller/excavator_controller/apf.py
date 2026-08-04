import numpy as np

def attractive_force(p_tip: np.ndarray, p_target: np.ndarray, K_att: float) -> np.ndarray:
    """
    APF引力

    Args:
        p_tip: 末端位置[x, y, z]
        p_target: 期望位置, 来自遥操采集
        K_att: 引力参数

    Returns:
        F_att: 3D引力向量
    """
    return -K_att * (p_tip - p_target)
 
def attractive_torque(F_att: np.ndarray, J_p: np.ndarray) -> np.ndarray:
    """
    转化关节力矩

    Args:
        F_att: 引力
        J_p: 雅可比矩阵

    Returns:
        Tau_att: 关节力矩
    """
    return J_p.T @ F_att

def joint_limit_torque(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray,
                       rho: float = 0.15, K: float = 100.0,
                       tau_max: float = 5000.0) -> np.ndarray:
    """
    关节限位软斥力 —— 靠近限位时产生斥力矩将关节推回安全区。

    U(q_i) = 0.5 * K * (1/Δd - 1/ρ)²,  Δd = 距限位边界距离
    τ_i = -∂U/∂q_i

    修复: 原实现当 q 越过限位 (Δd<0) 时 1/Δd 为负, 斥力矩变号,
    反而把关节推离可行域 (越卷越狠, 且百万级爆炸淹没其他控制).
    现在: 越界时 Δd 钳到 ε>0 (斥力保持正方向, 推回可行域), 并加 tau_max 饱和.

    Args:
        q:       当前关节角度 [q1, q2, q3, q4], rad
        q_min:   关节下限, rad
        q_max:   关节上限, rad
        rho:     软限位激活窗口, rad (默认 0.15 ≈ 8.6°)
        K:       斥力增益
        tau_max: 斥力矩饱和上限, 防止越界时爆炸淹没其他控制项

    Returns:
        tau_joint: 4D 关节力矩, 远离限位时为零, N·m
    """
    # 越界时钳到小正数: 保证斥力方向永远"推离限位、推回可行域"
    d_low  = np.clip(q - q_min, 1e-4, None)   # 距下限, >0
    d_high = np.clip(q_max - q, 1e-4, None)   # 距上限, >0

    # 下限侧斥力: 正 (把关节推离下限); 上限侧斥力: 负 (把关节推离上限)
    tau_low  = np.where(d_low < rho,  K * (1/d_low - 1/rho) / d_low**2, 0.0)
    tau_high = np.where(d_high < rho, -K * (1/d_high - 1/rho) / d_high**2, 0.0)

    # 饱和, 防止越界时爆炸
    tau_low  = np.clip(tau_low, 0.0, tau_max)
    tau_high = np.clip(tau_high, -tau_max, 0.0)

    return tau_low + tau_high


def tipover_torque(q: np.ndarray,
                   alpha_thresh: float = 0.4,
                   K: float = 300.0,
                   tau_max: float = 3000.0) -> tuple[np.ndarray, float]:
    """
    倾覆风险斥力 (ZMP 稳定裕度保护, 安全优先级 L1).

    当 ZMP 稳定裕度 α_margin < α_thresh 时, 产生关节力矩把工作装置推回安全区.
    势能 U = 0.5*K*(1/α - 1/α_thresh)², 力矩 τ = -∂U/∂q.

    Args:
        q:           关节角 [q1,q2,q3,q4] (raw 空间)
        alpha_thresh: 稳定裕度阈值 (m), α < 此值开始斥力
        K:            斥力增益
        tau_max:      力矩饱和上限

    Returns:
        tau_tip: 4D 关节力矩 (直接叠加到 τ_cmd)
        alpha:   当前稳定裕度 (m), 用于调试
    """
    from excavator_kinematics.zmp import zmp_alpha_margin

    alpha, _ = zmp_alpha_margin(q)
    if alpha >= alpha_thresh:
        return np.zeros(4), alpha

    # 势能对 q 的梯度 (数值微分) → 斥力矩
    eps = 1e-5
    tau = np.zeros(4)
    for i in range(4):
        qp = q.copy(); qm = q.copy()
        qp[i] += eps; qm[i] -= eps
        alpha_p, _ = zmp_alpha_margin(qp)
        alpha_m, _ = zmp_alpha_margin(qm)
        dalpha_dq = (alpha_p - alpha_m) / (2 * eps)
        # U = 0.5*K*(1/α - 1/α_th)²,  dU/dq = -K*(1/α-1/α_th)/α² * dα/dq
        # τ = -dU/dq = K*(1/α-1/α_th)/α² * dα/dq
        tau[i] = K * (1.0 / alpha - 1.0 / alpha_thresh) / (alpha ** 2) * dalpha_dq

    tau = np.clip(tau, -tau_max, tau_max)
    return tau, alpha


def pipe_repulsive_force(p_tip: np.ndarray, pipe_pos: np.ndarray,
                          rho: float = 1.0, K: float = 10000.0) -> np.ndarray:
    """
    管线斥力 —— 齿尖靠近地下管线时产生的 3D 任务空间斥力。

    管线视为沿 Y 轴水平铺设的圆柱，距离取 XZ 平面上的投影距。
    F_rep = K·(1/d - 1/ρ) / d² · (p_tip - pipe_pos) / d

    Args:
        p_tip:    齿尖在 Σ0 系中的位置 [x, y, z], m
        pipe_pos: 管线在 Σ0 系中的位置 [x, y, z], m (Y 任意)
        rho:      斥力影响范围, m (默认 1.0)
        K:        斥力增益 (默认 10000, 极大确保管线是不可逾越的屏障)

    Returns:
        F_rep: 3D 斥力向量, d ≥ ρ 时为零
    """
    # 管线沿 Y 轴水平, 距离取 XZ 平面投影
    dx = p_tip[0] - pipe_pos[0]
    dz = p_tip[2] - pipe_pos[2]
    d = float(np.sqrt(dx * dx + dz * dz))

    if d >= rho or d < 1e-6:
        return np.zeros(3)

    f_mag = K * (1.0 / d - 1.0 / rho) / (d * d)
    return f_mag * np.array([dx, 0.0, dz]) / d
 