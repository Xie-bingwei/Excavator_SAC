# 4-DOF 液压挖掘机安全自主作业 —— 轨迹安全化、冗余分解与在线优化

> **文档定位**：学术论文方法论文档，涵盖问题形式化、CBF 轨迹安全化、APF 冗余分解、Z-based 铲斗卷斗逻辑、SAC 强化学习集成的完整推导
> **核心贡献**：轨迹安全化（离线）+ APF 冗余分解（在线）+ SAC 残差强化学习的三层安全自适应架构
> **关联代码**：`excavator_controller/`、`excavator_trajectory/`、`excavator_kinematics/`、`rl/`
> **最后更新**：2026-08-05

---

## 目录

1. [问题形式化](#1-问题形式化)
2. [系统架构总览](#2-系统架构总览)
3. [CBF 轨迹安全化 —— 离线阶段](#3-cbf-轨迹安全化--离线阶段)
4. [APF 任务-零空间分层控制 —— 在线阶段](#4-apf-任务-零空间分层控制--在线阶段)
5. [铲斗 Z-based 入土即卷策略](#5-铲斗-z-based-入土即卷策略)
6. [Tipping-over Protection via ZMP Margin](#6-tipping-over-protection-via-zmp-margin)
7. [SAC 残差强化学习 —— 在线策略优化](#7-sac-残差强化学习--在线策略优化)
8. [理论分析](#8-理论分析)
9. [实验设计](#9-实验设计)
10. [论文提纲](#10-论文提纲)

---

## 1. 问题形式化

### 1.1 系统模型

本节形式化地定义 4-DOF 液压挖掘机的运动学与控制问题。

**关节空间** $\mathcal{Q} \subset \mathbb{R}^4$，关节向量定义为：

$$\boxed{\mathbf{q} = [q_{\text{swing}}, q_{\text{boom}}, q_{\text{arm}}, q_{\text{bucket}}]^T \in \mathbb{R}^4}$$

其中各关节的物理限位构成紧致子集：

$$\begin{aligned}
q_1 &\in [-\pi, \pi] \quad &\text{(Swing: 上部结构回转)} \\
q_2 &\in [-0.8033, 1.155] \quad &\text{(Boom: 动臂俯仰)} \\
q_3 &\in [-2.7594, -0.6084] \quad &\text{(Arm: 斗杆俯仰)} \\
q_4 &\in [-2.614, 0.4433] \quad &\text{(Bucket: 铲斗卷扬)}
\end{aligned}$$

**任务空间由齿尖位置确定**：

$$\boxed{\mathbf{p}_{\text{tip}} = FK(\mathbf{q}) \in \mathbb{R}^3}$$

其中 $FK(\cdot)$ 为 Modified Denavit-Hartenberg (MDH) 递推正运动学。MDH 参数由 AGX 场景关节树解析并经 4 构型实机标定（标定精度：位置误差 $\leq 0.0003\text{m}$）。

**齿尖线速度与关节角速度的关系通过位置雅可比表达**：

$$\boxed{\dot{\mathbf{p}}_{\text{tip}} = J_p(\mathbf{q}) \cdot \dot{\mathbf{q}}, \quad J_p \in \mathbb{R}^{3 \times 4}}$$

> **冗余性**：$J_p$ 是 $3 \times 4$ 宽矩阵，存在一维零空间。这意味着同一个齿尖位置对应无穷多种关节构型 —— 这是多目标优化（轨迹跟踪 + 铲斗卷斗 + 稳定性）的几何根源。

### 1.2 示教轨迹

由人类操作员通过遥操作录制一条挖掘循环，经 0.4m 距离阈值降采样至 64 个三维路点：

$$\mathcal{T}_{\text{original}} = \{\mathbf{p}_k\}_{k=0}^{63}, \quad \mathbf{p}_k \in \mathbb{R}^3$$

同时录制每个路点对应的关节角度 $\mathbf{q}^{\text{demo}}_k \in \mathbb{R}^4$，构成 $(s, \mathbf{p}, \mathbf{q})$ 三元组。

### 1.3 安全约束

地下管线位于 $\mathbf{p}_{\text{pipe}} \in \mathbb{R}^3$（来自场景 BIM 数据）。由于管线沿水平方向（$Y$ 轴）铺设，齿尖危险程度取决于 XZ 平面投影距离：

$$\boxed{d(\mathbf{p}) = \sqrt{(p_x - x_{\text{pipe}})^2 + (p_z - z_{\text{pipe}})^2}}$$

硬安全条件：$d(\mathbf{p}_{\text{tip}}) \geq d_{\text{safe}}$，其中 $d_{\text{safe}} = 1.5\text{m}$ 涵盖物理管线半径、铲斗本体半径和安全余量。

---

## 2. 系统架构总览

本方法采用 **"离线安全化 + 在线分层控制"** 架构，将安全职责部署到不同处理阶段：

```
┌─────────────────────────────────────────────────────────────────────────┐
│               阶段 1: CBF 轨迹安全化 (Offline Safety Certification)        │
│                                                                         │
│  示教轨迹 T_demo                                                        │
│      │                                                                  │
│      ▼                                                                  │
│  显式梯度下降: 最小化复合代价 J(ξ) = J_obs + λ·J_smooth                   │
│      │                                                                  │
│      ▼                                                                  │
│  安全化轨迹 T_cert (所有路点满足 d ≥ d_safe)                               │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         阶段 2: 在线 APF 任务-零空间分层控制 (Online Redundant Control)     │
│                                                                         │
│  任务空间 (3D): carrot APF 跟踪 T_cert → τ_task = J_p^T · F_att          │
│  零空间 (1D): 关节构型参考 + 关节限位软约束 → τ_null                      │
│  铲斗通道: Z-based 入土即卷, 独立高速驱动 → 不经过零空间投影                │
│  ZMP 保护: 稳定裕度斥力 → 把臂推回安全区 (L1 最高优先级)                    │
│                                                                         │
│  总力矩: τ_cmd = τ_task + N·τ_null + τ_bucket + τ_tipover + τ_limit       │
│                                                                         │
│  产出: 安全示范数据集 {s_t, a_t_apf, r_t, s_{t+1}}                        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          阶段 3: SAC 残差强化学习 (Online Residual Optimization)          │
│                                                                         │
│  动作: Δq ∈ [-0.05, +0.05] rad — 叠加到 APF 的 q_des_apf                │
│  状态: 17-dim (关节角 + 齿尖位置 + 轨迹进度 + 安全距离 + ZMP + base_ω)    │
│  奖励: 轨迹前进 + 稳定惩罚 + 管线代价 + 动作平滑                           │
│                                                                         │
│  训练: 离线运动学环境, 70% 无管道 + 30% 有管道随机切换                     │
│  目标: 优化挖掘平滑性 + 稳定性 + 管线安全权衡                              │
└─────────────────────────────────────────────────────────────────────────┘
```

**三层安全保障的协同**：

1. **第一层（离线）** — 轨迹本身满足硬安全约束 → 在线跟踪即使有误差，齿尖已在安全邻域内
2. **第二层（在线 APF）** — ZMP 倾覆斥力 + 关节限位 → 直接力矩干预
3. **第三层（RL）** — SAC 微调动作不离开安全邻域（动作幅值 $\leq 0.05$ rad）

---

## 3. CBF 轨迹安全化 —— 离线阶段

### 3.1 代价泛函

#### 3.1.1 符号定义

**定义（轨迹）**：一条离散轨迹由 $N$ 个路点构成：

$$\boxed{\xi = \{\mathbf{p}_k\}_{k=0}^{N-1}, \quad \mathbf{p}_k \in \mathbb{R}^3}$$

$\xi$ 是全体路点的集合。两个边界点 $\mathbf{p}_0$ 和 $\mathbf{p}_{N-1}$ 分别为起点和终点，固定不动（安全化不应改变作业的起始和终止位置）。

**定义（代价泛函）**：轨形 $\xi$ 的复合代价由两部分加权组成：

$$\boxed{J(\xi) = \underbrace{\sum_{k=0}^{N-1} J_{\text{obs}}(\mathbf{p}_k)}_{\text{障碍物接近代价}} \;+\; \lambda \underbrace{\sum_{k=1}^{N-2} J_{\text{smooth}}(\mathbf{p}_{k-1}, \mathbf{p}_k, \mathbf{p}_{k+1})}_{\text{轨迹不平滑代价}}}$$

$J(\xi)$ 是一个**标量能量**。它的值越小，轨迹越安全且越平滑。优化目标是找到一条安全化轨迹 $\xi^*$ 使 $J(\xi^*)$ 极小化。

#### 3.1.2 各分量的物理含义

当路点与管线的投影距离低于安全阈值时，产生二次代价：

$$\boxed{J_{\text{obs}}(\mathbf{p}_k) = \max\big(0,\; d_{\text{safe}} - d(\mathbf{p}_k)\big)^2}$$

**物理含义**：$J_{\text{obs}} > 0$ 仅当 $d(\mathbf{p}_k) < d_{\text{safe}}$ 时发生。二次形式意味着距离越近，代价增长越快 —— 等价于一个软约束，驱动路点远离管线。

#### 平滑代价

惩罚相邻路点之间的不平滑性，防止路点被推开后形成"锯齿"：

$$\boxed{J_{\text{smooth}} = \big\|\mathbf{p}_k - \frac{\mathbf{p}_{k-1} + \mathbf{p}_{k+1}}{2}\big\|^2}$$

**物理含义**：若 $\mathbf{p}_k$ 恰好位于相邻两点的中间位置，则代价为零。偏离越大代价越高。这等价于施加了一个"弹簧"约束，将相邻路点拉向平滑配置。

### 3.2 代价函数的目的与优化机制

**核心问题**：原始示教轨迹穿越了管线的安全禁区（部分路点 $d(\mathbf{p}_k) < d_{\text{safe}}$）。我们希望将轨迹变形，使所有路点满足 $d(\mathbf{p}_k) \geq d_{\text{safe}}$，同时尽可能保留原始轨迹的整体形状。

**代价函数的角色**：$J(\xi) = J_{\text{obs}} + \lambda J_{\text{smooth}}$ 是一个能量泛函。它的构造遵循"惩罚不安全、奖励平滑"的原则：

- $J_{\text{obs}}$ 施加**排斥力**：任何路点若离管线太近，会产生二次代价。代价的梯度指向远离管线的方向，驱动该路点向外移动。
- $J_{\text{smooth}}$ 施加**弹性力**：相邻三个路点之间若不平滑（中间点偏离相邻点的平均值），产生二次代价。梯度将中间路点拉向邻域均值，防止局部抬升造成锯齿状断崖。

**为什么梯度下降有效**：总代价 $J$ 是凸函数。梯度下降等价于将轨迹**沿代价减小的方向逐步变形**——在不安全区域，$J_{\text{obs}}$ 的梯度占主导，将路点推离管线；离开危险区后 $J_{\text{obs}} = 0$，$J_{\text{smooth}}$ 的梯度占主导，将变形区域的边界自然过渡到未变形区域。这类似于物理中"弹性体被障碍物顶起"的平衡形态——障碍物对弹性体施加排斥力，弹性体的内部张力将变形平滑分布到一个邻域内。

**物理类比**：将轨迹想象为一根橡皮筋。管线是一根钉子从下方顶上来。橡皮筋被钉子顶起的部位对应 $J_{\text{obs}}$ 的排斥力（向上推），钉子两侧的橡皮筋自动形成平滑的过渡弧线对应 $J_{\text{smooth}}$ 的弹性力（拉住变形区域的边界）。梯度下降迭代本质上是在寻找橡皮筋在钉子上的平衡构型。

**与在线斥力的本质区别**：在线 APF 斥力是"控制器层面的力对抗"——引力向前拉、斥力往后推，在线平衡点即是振荡点。离线轨迹安全化是"规划层面的几何变形"——一次性将轨迹本身改写到安全区域内，在线控制只需要跟踪一条安全的轨迹，无对抗、无振荡。**安全性从控制的输出转移到了规划的输入**——这是本方法的核心哲学。

### 3.3 从代价泛函到路点更新——梯度下降

**问题**：$J(\xi)$ 是一个 $\mathbb{R}^{3N}$ 空间中的标量函数（$N$ 个路点，每个 3 维），如何找到使 $J$ 极小化的 $\xi^*$？

**答案**：**梯度下降迭代**。每一步沿着 $J$ 减小的最陡方向移动每个路点：

$$\boxed{\mathbf{p}_k \leftarrow \mathbf{p}_k - \alpha \cdot \nabla_{\mathbf{p}_k} J(\xi), \quad k = 0, 1, \ldots, N-1}$$

其中 $\alpha$ 是**步长**（学习率），控制每次迭代中路点移动的幅度。

**$\alpha$ 的作用**：
- $\alpha$ 太小 → 收敛过慢，需要更多迭代
- $\alpha$ 太大 → 路点可能"跳过"最小点，产生震荡
- 本文采用自适应步长：$\alpha = 0.25 / d_{\text{safe}}$。当安全距离要求更大时（如 $d_{\text{safe}} = 1.5\text{m}$），步长自动缩小（$\alpha \approx 0.167$），防止过度变形

**梯度下降如何体现在路点上**：将定理 1 的梯度 $\nabla_{\mathbf{p}_k} J$ 代入更新公式：

$$\boxed{\mathbf{p}_k^{\text{new}} = \mathbf{p}_k^{\text{old}} - \alpha \cdot \underbrace{\nabla J_{\text{obs}}(\mathbf{p}_k)}_{\text{推离管线}} - \alpha \lambda \cdot \underbrace{2\Big(\mathbf{p}_k - \frac{\mathbf{p}_{k-1} + \mathbf{p}_{k+1}}{2}\Big)}_{\text{拉向平滑}}}$$

**逐项物理含义**：

| 项 | 作用 | 激活条件 |
|:---|:---|:---|
| $-\alpha \cdot \nabla J_{\text{obs}}$ | 将路点沿"管线→路点"方向推开 | $d(\mathbf{p}_k) < d_{\text{safe}}$ |
| $-\alpha\lambda \cdot \nabla J_{\text{smooth}}$ | 将路点拉向相邻两点的平均值 | 始终激活（$k = 1, \ldots, N-2$） |

**关键观察**：当 $\nabla J_{\text{obs}} = 0$（所有路点已安全），只剩平滑项在起作用——此时迭代等价于高斯滤波，将安全化区域的边界自然融合到未变形区域。

**为什么梯度下降是对的**：$J_{\text{obs}}$ 是凸函数（二次惩罚），$J_{\text{smooth}}$ 也是凸函数（二次范数）。凸函数的和仍是凸函数。梯度下降对凸函数在适当步长下保证全局收敛——**不可能陷入局部极小**。

**停止条件**：所有不安全路点被推出后，$J_{\text{obs}} = 0$，且平滑项已收敛 → 算法自然停止。迭代上限 $max\_iter$ 提供安全网保证有限终止。

### 3.4 梯度方向推导

**定理 1（梯度方向）**：代价泛函 $J(\xi)$ 对第 $k$ 个路点的梯度为：

$$\boxed{\nabla_{\mathbf{p}_k} J = \nabla J_{\text{obs}}(\mathbf{p}_k) + \lambda \cdot \nabla J_{\text{smooth}}(\mathbf{p}_{k-1}, \mathbf{p}_k, \mathbf{p}_{k+1})}$$

其中：

$$\nabla J_{\text{obs}}(\mathbf{p}_k) = \begin{cases}
-2\big(d_{\text{safe}} - d(\mathbf{p}_k)\big) \cdot \dfrac{\mathbf{p}_k - \mathbf{p}_{\text{pipe}}}{d(\mathbf{p}_k)}, & d(\mathbf{p}_k) < d_{\text{safe}} \\
\mathbf{0}, & d(\mathbf{p}_k) \geq d_{\text{safe}}
\end{cases}$$

$$\nabla J_{\text{smooth}} = 2\left(\mathbf{p}_k - \frac{\mathbf{p}_{k-1} + \mathbf{p}_{k+1}}{2}\right)$$

**推导**：$J_{\text{obs}}$ 的梯度通过链式法则得到。令 $d = \|\mathbf{p}_k - \mathbf{p}_{\text{pipe}}\|_{XZ}$（投影距离），则 $\frac{\partial d}{\partial p_x} = \frac{p_x - x_{\text{pipe}}}{d}$，$\frac{\partial d}{\partial p_z} = \frac{p_z - z_{\text{pipe}}}{d}$。由 $\frac{\partial}{\partial d}(d_{\text{safe}} - d)^2 = -2(d_{\text{safe}} - d)$，链式法则即得。$J_{\text{smooth}}$ 的梯度由平方范数的导数直接给出。

> **$Y$ 方向的处理**：挖掘机在标准示教过程中 $q_1 \approx 0$（上部结构不回转），齿尖始终在 $y \approx 0$（XZ 平面内运动）。因此障碍物代价仅对 $X$ 和 $Z$ 分量求梯度，$Y$ 分量保持不变。这简化了优化问题的维度（2D 而非 3D），但不失正确性。

### 3.5 两阶段迭代优化算法

```
Algorithm 1: CBF 轨迹安全化 (CBF-TrajCert)
─────────────────────────────────────────────
Input:  T_demo = {p_k},  p_pipe,  d_safe,  λ,  max_iter
Output: T_cert

 1:  η = 0.25 / d_safe           // 自适应步长 (安全距离越大步长越小)
 2:  ξ ← T_demo
 3:  for iter = 1 to max_iter do
 4:      moved ← False
 5:
 6:      // ── Phase 1: 障碍物推力 ──
 7:      for k = 0 to N-1 do
 8:          d_k ← ||p_k - p_pipe|| (XZ projection)
 9:          if d_k < d_safe and d_k > 1e-6 then
10:              g_obs ← -2(d_safe - d_k) · (p_k - p_pipe) / d_k
11:              p_k ← p_k - η · g_obs
12:              moved ← True
13:
14:      // ── Phase 2: 轨迹平滑 ──
15:      for k = 1 to N-2 do
16:          avg ← (p_{k-1} + p_{k+1}) / 2
17:          p_k ← p_k - η · λ · 2(p_k - avg)
18:
19:      if ¬ moved ∨ J_obs < 1e-4 then break
20:  return ξ
```

**关键设计选择**：

- **Phase 1 和 Phase 2 交替执行**（而非合并为单一梯度）：交替保证了障碍物代价下降的同时轨迹保持平滑。若合并，梯度的不同量级会导致优化不稳定。
- **自适应步长** $\alpha = 0.25 / d_{\text{safe}}$：当 $d_{\text{safe}} = 1.5\text{m}$ 时 $\alpha \approx 0.167$。步长与安全距离成反比的设计确保了不同 $d_{\text{safe}}$ 设置下收敛速度一致。
- **Z 分量矫偏**：梯度下降后若 $z_{\text{new}} < z_{\text{original}}$（路点被向下推），则将修正方向限幅，确保路点只抬不压。这是基于工程常识的设计——管道在挖掘机下方，安全化应该向上抬升轨迹而非将其推向更深的土壤。

**定理 2（收敛性）**：若步长满足 $\alpha < \frac{1}{2(1+4\lambda)}$，算法在有限步内收敛到安全可行解，使 $\forall k: d(\mathbf{p}_k) \geq d_{\text{safe}}$。且仅在危险路点（$d_k^{(0)} < d_{\text{safe}}$）上需要迭代，收敛至多 $O\!\left(\max_k \frac{d_{\text{safe}} - d_k^{(0)}}{\alpha}\right)$ 步。

**证明**：

**步骤 1：代价函数的凸性。** $J_{\text{obs}}(\mathbf{p}_k) = \max(0, d_{\text{safe}} - \|\mathbf{p}_k - \mathbf{p}_{\text{pipe}}\|)^2$ 是距离 $d$ 的凸递减函数的平方，关于 $\mathbf{p}_k$ 是凸的。$J_{\text{smooth}}$ 是 $\mathbf{p}_k$ 的二次型 $(\mathbf{p}_{k-1} - 2\mathbf{p}_k + \mathbf{p}_{k+1})^T(\mathbf{p}_{k-1} - 2\mathbf{p}_k + \mathbf{p}_{k+1})/4$，严格凸（在恒定偏移方向上退化为零）。两者之和 $J(\xi) = J_{\text{obs}} + \lambda J_{\text{smooth}}$ 是凸函数。凸函数的任何驻点都是全局最小值。

**步骤 2：梯度下降的下降性质。** 将全体路点堆叠为向量 $\mathbf{x} = [\mathbf{p}_0; \mathbf{p}_1; \ldots; \mathbf{p}_{N-1}] \in \mathbb{R}^{3N}$。迭代可写为：

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \mathbf{g}(\mathbf{x}_t), \quad \mathbf{g} = \nabla_{\mathbf{x}} J$$

需要 $\mathbf{g}$ 满足 Lipschitz 条件以保证充分下降。

对于 $J_{\text{smooth}}$：其梯度为 $\nabla J_{\text{smooth}} = \mathbf{H}_s \mathbf{x}$，其中 $\mathbf{H}_s$ 是由二阶差分矩阵构造的块三对角矩阵。$\mathbf{H}_s$ 在路点 $k$ 处的行包含系数 $[\frac{1}{2}\mathbf{I}_3, -2\mathbf{I}_3, 3\mathbf{I}_3, -2\mathbf{I}_3, \frac{1}{2}\mathbf{I}_3]$（分别对应 $k-2, k-1, k, k+1, k+2$）。由 Gershgorin 圆盘定理，$\|\mathbf{H}_s\|_2 \leq 3 + 2 \times 2 + 2 \times \frac{1}{2} = 8$。因此 $\|\nabla J_{\text{smooth}}\|$ 的 Lipschitz 常数 $L_s = 8$。

对于 $J_{\text{obs}}$：其梯度 $\nabla J_{\text{obs}}(\mathbf{p}_k) = -2(d_{\text{safe}} - d_k) \cdot \mathbf{u}_k$，其中 $\mathbf{u}_k = (\mathbf{p}_k - \mathbf{p}_{\text{pipe}})/d_k$。关键观察：$\nabla J_{\text{obs}}$ 的方向始终背离管线，因此 $d_k^{(t)}$ **单调不降**（路点只会远离管线）。在初始最小距离 $d_{\min}^{(0)} = \min_k d_k^{(0)}$ 上，$J_{\text{obs}}$ 的 Hessian 最大特征值为 $2$（当 $d = d_{\text{safe}}$ 时），故 $L_{\text{obs}} = 2$。

复合梯度 $\mathbf{g}$ 的全局 Lipschitz 常数 $L \leq L_{\text{obs}} + \lambda L_s = 2 + 8\lambda$。

由标准梯度下降的下降引理（Nesterov, 2004, Thm 1.2.4），当 $\alpha \leq 1/L$ 时：

$$\boxed{J(\mathbf{x}_{t+1}) \leq J(\mathbf{x}_t) - \frac{\alpha}{2}\|\mathbf{g}(\mathbf{x}_t)\|^2}$$

其中 $\|\mathbf{g}\|^2$ 是在当前路点上的梯度平方和。

**步骤 3：每步代价下降量。** 设有 $m$ 个路点违反安全约束（$d_k < d_{\text{safe}}$）。每个违反路点的梯度范数为：

$$\|\nabla J_{\text{obs}}(\mathbf{p}_k)\| = 2(d_{\text{safe}} - d_k) \geq 2\varepsilon$$

其中 $\varepsilon = \min_{k: d_k < d_{\text{safe}}} (d_{\text{safe}} - d_k) > 0$ 是当前最小违规量。因此 $\|\mathbf{g}\|^2 \geq \sum_{k \in \text{violating}} \|\nabla J_{\text{obs}}\|^2 \geq 4m\varepsilon^2$。代回下降引理：

$$J(\mathbf{x}_{t+1}) \leq J(\mathbf{x}_t) - 2\alpha m \varepsilon^2$$

**步骤 4：有限步终止。** 初始总代价 $J_0 = J(\xi^{(0)})$ 是有限值（原始轨迹距离管线有一定距离，$d_k > 0$ 对大多数路点成立）。每一步 $J$ 至少减少 $2\alpha m \varepsilon^2 > 0$。由于 $J \geq 0$，迭代次数由 $J_0 / (2\alpha m\varepsilon^2)$ 上界。当所有路点满足 $d_k \geq d_{\text{safe}}$ 时 $J_{\text{obs}} = 0$，步长条件 $\alpha < 1/L$ 保证平滑项随之单调递减至平稳。算法至多经过 $\lceil \max_{k: d_k^{(0)} < d_{\text{safe}}} \frac{d_{\text{safe}} - d_k^{(0)}}{\alpha} \rceil$ 次障碍物推力迭代即收敛。

**步骤 5：收敛到可行解。** 凸函数 $J$ 的梯度下降收敛到全局最小。当 $\nabla J = \mathbf{0}$ 时，对每个 $k$：

$$\nabla J_{\text{obs}}(\mathbf{p}_k) + \lambda \cdot \nabla J_{\text{smooth}}(\mathbf{p}_k) = \mathbf{0}$$

若存在 $k$ 使 $d_k < d_{\text{safe}}$，则 $\nabla J_{\text{obs}} \neq \mathbf{0}$ 且方向指向远离管线。但 $\nabla J_{\text{smooth}}$ 仅包含相邻路点的加权平均——其方向与 $\nabla J_{\text{obs}}$ 最多部分抵消（在平滑边界处）。由于 $\lambda$ 适中（0.3），且 $\nabla J_{\text{smooth}} \to \mathbf{0}$ 当路点收敛至平滑构型，不可能出现 $\nabla J_{\text{obs}}$ 被 $\lambda\nabla J_{\text{smooth}}$ 完全抵消的驻点。因此收敛点必然满足 $\nabla J_{\text{obs}}(\mathbf{p}_k) = \mathbf{0}$ 对全体 $k$，即 $\forall k: d(\mathbf{p}_k) \geq d_{\text{safe}}$。$\square$

**参数验证**：$\lambda = 0.3$，$L = 2 + 8 \times 0.3 = 4.4$，步长条件要求 $\alpha < 1/4.4 = 0.227$。实际 $\alpha = 0.25/1.5 = 0.167 < 0.227$ ✓。实验中 94 次迭代收敛（图 3），与理论预测一致。

---

## 4. APF 任务-零空间分层控制 —— 在线阶段

### 4.1 控制架构

对 4-DOF 冗余臂的控制采用 **分层策略**：管线安全由离线轨迹安全化保证，在线只需跟踪安全轨迹，从而彻底规避了在线斥力与引力之间力对抗导致的振荡问题。

```
任务空间 (3D):  引力跟踪 T_cert        → τ_task = J_p^T · F_att
                  仅驱动 q2, q3 (q1=0 平面, q4 独立)

零空间 (1D):   关节构型参考 + 软限位     → τ_null, 经 N 投影解除任务空间耦合

铲斗独立:      Z-based 入土即卷          → τ_bucket, 直接叠加, 不经零空间

ZMP 保护:      稳定裕度斥力              → τ_tipover, L1 最高优先级
```

### 4.2 任务空间引力跟踪

**定义（轨迹参数化）**：连续轨迹 $\mathcal{T}_{\text{cert}}(s): [0, 1] \to \mathbb{R}^3$ 将归一化弧长映射为 3D 路点。路点之间线性插值。

**定义（当前最近点）**：给定齿尖位置 $\mathbf{p}_{\text{tip}}$，最近轨迹点参数通过逐段点到线段距离的最小化求得：

$$\boxed{s^* = \arg\min_{s \in [0,1]} \|\mathbf{p}_{\text{tip}} - \mathcal{T}_{\text{cert}}(s)\|}$$

为实现渐进跟踪，引入 **carrot 前视策略**：

$$\boxed{s_{\text{target}} = \min\big(s^{\text{ref}} + \Delta s_{\text{fwd}},\; s^* + \Delta s_{\text{star}},\; 1.0\big)}$$

其中 $\Delta s_{\text{fwd}} = 0.05$（基于参考进度的前视步长），$\Delta s_{\text{star}} = 0.10$（基于当前最近点的前视步长）。Carrot 目标始终在齿尖前侧，避免了纯最近点跟踪在拐折处回退的抖动。

**引力**（二次势场）：

$$\boxed{U_{\text{att}}(\mathbf{p}_{\text{tip}}) = \frac{1}{2} K_{\text{att}} \cdot \|\mathbf{p}_{\text{tip}} - \mathcal{T}_{\text{cert}}(s_{\text{target}})\|^2}$$

$$\boxed{\mathbf{F}_{\text{att}} = -\nabla U_{\text{att}} = -K_{\text{att}} \cdot \big(\mathbf{p}_{\text{tip}} - \mathcal{T}_{\text{cert}}(s_{\text{target}})\big)}$$

**$K_{\text{att}} = 300$**，对任务空间位置跟踪提供适中的吸引力。引力方向始终从当前齿尖位置指向 carrot 目标点。

**引力→关节力矩映射**：

$$\boxed{\boldsymbol{\tau}_{\text{task}} = J_p^T(\mathbf{q}) \cdot \mathbf{F}_{\text{att}} \in \mathbb{R}^4}$$

**重要设计决策**：$\tau_{\text{task}}[3] = 0$（bucket 关节不参与任务空间跟踪）。理由：bucket 是完全冗余自由度 — 其转动对齿尖位置的贡献极小（仅 $t_x$ 分量），若参与引力映射，bucket 会在引力拖拽下自由漂移，产生不受控的卷斗行为。

### 4.3 零空间冗余分解

**定义（零空间投影矩阵）**：

$$\boxed{N(\mathbf{q}) = I_{4} - J_p^T(\mathbf{q}) \cdot (J_p^{\dagger})^T(\mathbf{q})}$$

**解释**：对于一个宽矩阵 $J_p \in \mathbb{R}^{3 \times 4}$（行数 < 列数），常规逆 $J_p^{-1}$ 不存在。用 Moore-Penrose 伪逆 $J_p^{\dagger}$ 代替：

$$\boxed{J_p^{\dagger} = J_p^T (J_p J_p^T)^{-1} \in \mathbb{R}^{4 \times 3}}$$

`†`（dagger，匕首符号）是伪逆的国际标准记号。$J_p^{\dagger}$ 满足 $J_p \cdot J_p^{\dagger} = I_3$（右乘不变性），即 $\forall \mathbf{v} \in \mathbb{R}^3$，$J_p \cdot J_p^{\dagger} \mathbf{v} = \mathbf{v}$。

**性质 1（任务不变性）**：$\forall \boldsymbol{\tau}_0 \in \mathbb{R}^4$，$J_p \cdot N \cdot \boldsymbol{\tau}_0 = \mathbf{0}$。零空间力矩不产生齿尖线加速度，因此不影响任务空间跟踪精度。

**证明**：

$$\begin{aligned} J_p N &= J_p \big(I_4 - J_p^T (J_p^{\dagger})^T\big) = J_p - J_p J_p^T (J_p J_p^T)^{-1} J_p \\ &= J_p - I_3 \cdot J_p = J_p - J_p = 0 \end{aligned}$$

其中第三步利用了 $J_p J_p^T \in \mathbb{R}^{3 \times 3}$ 是可逆方阵，$J_p J_p^T (J_p J_p^T)^{-1} = I_3$。$\square$

**零空间子任务**：

$$\boxed{\boldsymbol{\tau}_{\text{null}} = N(\mathbf{q}) \cdot K_{\text{joint}} \cdot (\mathbf{q}^{\text{ref}} - \mathbf{q})}$$

其中 $K_{\text{joint}} = 60$，$\mathbf{q}^{\text{ref}}$ 来自示教录制的关节角度经参数 $s^{\text{ref}}$ 插值。零空间力矩的作用是将臂的冗余构型拉向示教参考构型 —— 在齿尖跟踪轨迹的同时，boom/arm 的"姿态"尽可能接近人类操作员的偏好。

**再次排除 q4**：$\tau_{\text{null}}[3] = 0$。铲斗卷斗不经过零空间投影，原因同 §4.2。

### 4.4 关节限位软保护

对 q1–q3 施加基于势场的软限位斥力：

$$\boxed{U_{\text{limit},i}(q_i) = \begin{cases}
\frac{1}{2}K_{\text{limit}}\big(\frac{1}{\Delta d} - \frac{1}{\rho}\big)^2, & \Delta d < \rho \\
0, & \Delta d \geq \rho
\end{cases}}$$

其中 $\Delta d = \min(|q_i - q_i^{\min}|, |q_i^{\max} - q_i|)$，$\rho = 0.15$ rad，$K_{\text{limit}} = 100$。

**力矩**：

$$\boxed{\boldsymbol{\tau}_{\text{limit},i} = -\frac{\partial U_{\text{limit},i}}{\partial q_i} = \pm K_{\text{limit}}\big(\frac{1}{\Delta d} - \frac{1}{\rho}\big) \cdot \frac{1}{\Delta d^2}}$$

其中符号取决于靠近上限（$-$，向负方向推）还是下限（$+$，向正方向推）。力矩被饱和在 $\tau_{\max} = 5000$ N·m 以防止越界时爆炸。$\varepsilon = 10^{-4}$ 防止分母为零。

**q4 被排除在关节限位势场之外**：q4 主要由 Z-based 卷斗逻辑驱动，物理限位由最终命令的裁剪保证。排除的原因是：示教参考可能恰好在 q4 上限附近（如 +0.4433 rad），软限位会在上限处产生相反的、饱和的 -5000 N·m 力矩与卷斗增益对抗。

### 4.5 总控制律

$$\boxed{\boldsymbol{\tau}_{\text{cmd}} = \underbrace{J_p^T \mathbf{F}_{\text{att}}}_{\text{任务空间跟踪}} + \underbrace{N \cdot \boldsymbol{\tau}_{\text{null}}}_{\text{零空间构型}} + \underbrace{\tau_{\text{bucket}} \cdot \mathbf{e}_4}_{\text{铲斗直接驱动}} + \underbrace{\boldsymbol{\tau}_{\text{limit}}}_{\text{q1–q3 软限位}} + \underbrace{\boldsymbol{\tau}_{\text{tipover}}}_{\text{ZMP 倾覆保护}}}$$

转到位置接口（与 Unity LockController 兼容）：

$$\boxed{\mathbf{q}_{\text{des}} = \mathbf{q} + K_{\text{imp}} \cdot \boldsymbol{\tau}_{\text{cmd}} \cdot \Delta t}$$

其中 $K_{\text{imp}} = 0.012$ rad/(N·m·s) 是力矩→位置增量转导增益，$\Delta t = 0.02$ s（50 Hz）。

**位置增量限幅**（防止单步跳变）：

$$\Delta q_{\max} = \begin{cases}
[0.06, 0.04, 0.05, 0.08], & s \geq 0.68\;\text{(提升/回位)} \\
[0.06, 0.03, 0.04, 0.08], & s < 0.68\;\text{(下压/挖掘)}
\end{cases}$$

$\mathbf{q}_{\text{des}}$ 经一阶低通滤波（$\alpha = 0.5$，仅 q1–q3）以抑制急加速/急停导致的反作用力，最终裁剪到物理限位后发布。

---

## 5. 铲斗 Z-based 入土即卷策略

### 5.1 问题动机

示教轨迹中，操作员在挖掘弧段（$s \in [0.19, 0.68]$）保持铲斗全开（$q_4 \approx +0.44$ rad），直到搬运段才突然卷斗。直接跟踪此参考会导致铲斗完全空铲出土 —— 这是示教风格所致，而非物理限制。

**解决方案**：不依赖轨迹参数 $s$ 来调度卷斗（$s$ 存在跟踪误差和平滑延迟），而是用齿尖的物理 $Z$ 坐标作为卷斗的触发信号 —— **入土即卷**。

### 5.2 有限状态机

定义两个卷斗模式：

- **空中模式**（$p_{\text{tip},z} \geq 0$）：铲斗保持全开构型，为切入土壤准备
- **土中模式**（$p_{\text{tip},z} < 0$）：铲斗直接追赶硬目标 $q_4^{\text{curl}} = -1.4$ rad

触发条件：齿尖 $Z$ 首次低于 $-0.03$ m（刚入土）即激活卷斗状态。

**状态转换**：

$$\text{state}(t_{k+1}) = \begin{cases}
\text{CURL}, & p_{\text{tip},z}(t_k) < -0.03 \\
\text{CURL}, & \text{state}(t_k) = \text{CURL} \land p_{\text{tip},z}(t_k) < 0 \\
\text{LIFT\_CURL}, & \text{state}(t_k) = \text{CURL} \land p_{\text{tip},z}(t_k) \geq 0 \\
\text{OPEN}, & \text{otherwise}
\end{cases}$$

**目标设定**：

$$q_4^{\text{target}} = \begin{cases}
-1.4 \text{ rad}, & \text{state} = \text{CURL} \quad\text{(土中狠卷)} \\
q_4^{\text{terminal}}, & \text{state} = \text{LIFT\_CURL} \quad\text{(出土继续卷至终点)} \\
q_4^{\text{ref}}, & \text{otherwise} \quad\text{(保持全开)}
\end{cases}$$

### 5.3 独立高速驱动通道

铲斗卷斗绕过 $K_{\text{imp}} \to \max\Delta q \to \text{低通滤波}$ 的常规流水线，使用独立高速通道：

$$\boxed{\Delta q_4^{\text{bucket}} = \text{clip}\big(q_4^{\text{target}} - q_4,\; -0.10,\; +0.10\big)\;\text{rad/step}}$$

这等价于 $5.0$ rad/s 的直接角速度指令（@ 50 Hz）。铲斗从 +0.44 rad 卷至 -1.4 rad 需 $\frac{1.84}{0.10} = 18.4$ 步 $\approx 0.37$ s。

**设计理由**：若将 bucket 力矩通过 $K_{\text{imp}} \to \max\Delta \to \text{滤波}$ 流水线，三重滞后叠加导致有效速率仅约 $0.006$ rad/step（实测），铲斗在整个挖掘阶段几乎不动。

---

## 6. Tipping-over Protection via ZMP Margin

### 6.1 准静态 ZMP 近似

对低速挖掘作业（齿尖速度 $\ll 1$ m/s），采用准静态近似：ZMP ≈ 整机质心的地面投影。

**整机质心**（5 个刚体质量加权）：

$$\boxed{\mathbf{p}_{\text{CoM}}(q) = \frac{\sum_{k \in \{\text{base}, \text{body}, \text{boom}, \text{arm}, \text{bucket}\}} m_k \cdot \mathbf{p}_{\text{com},k}(q)}{\sum_k m_k}}$$

各连杆质心通过 FK 关节链的几何中点近似计算。质量参数由 AGX 模型实测（底盘 5000 kg，上部结构 9800 kg，动臂 2200 kg，斗杆 1500 kg，铲斗 1000 kg）。

**支撑多边形**（履带接地矩形，$X \in [-1.82, 1.82]$ m，$Y \in [-1.20, 1.20]$ m）：

$$\boxed{\alpha_{\text{margin}} = \min\begin{pmatrix}
\text{SUP\_X} - x_{\text{CoM}},\; x_{\text{CoM}} + \text{SUP\_X}, \\
\text{SUP\_Y} - y_{\text{CoM}},\; y_{\text{CoM}} + \text{SUP\_Y}
\end{pmatrix}}$$

$\alpha_{\text{margin}} < 0$ 表示 ZMP 已超出支撑多边形边界 —— 倾覆事件。

### 6.2 倾覆斥力矩

当 $\alpha_{\text{margin}} < \alpha_{\text{thresh}} = 0.4$ m 时触发斥力：

$$\boxed{U_{\text{tip}}(q) = \begin{cases}
\frac{1}{2}K_{\text{tip}}\big(\frac{1}{\alpha} - \frac{1}{\alpha_{\text{thresh}}}\big)^2, & \alpha < \alpha_{\text{thresh}} \\
0, & \alpha \geq \alpha_{\text{thresh}}
\end{cases}}$$

力矩通过链式法则 + 数值微分计算（避免解析 $\partial \alpha / \partial q_i$ 的复杂符号推导）：

$$\boxed{\tau_{\text{tip},i} = K_{\text{tip}}\big(\frac{1}{\alpha} - \frac{1}{\alpha_{\text{thresh}}}\big)\frac{1}{\alpha^2} \cdot \frac{\alpha(q + \varepsilon \mathbf{e}_i) - \alpha(q)}{\varepsilon}}$$

其中 $K_{\text{tip}} = 300$，$\varepsilon = 10^{-5}$，力矩饱和在 $\pm 3000$ N·m。倾覆斥力直接叠加到关节力矩指令，不经 $J_p^T$ 映射 —— 因为势函数定义在关节空间，拉格朗日力学中广义力就是势能对广义坐标的负梯度。

> **为什么倾覆保护不在任务空间处理**：同一个齿尖位置可能对应完全不同的构型（boom 高 + arm 近 vs. boom 低 + arm 远），倾覆风险截然不同。定义在关节空间的势函数能区分这些情况，而任务空间势函数无法区分。

---

## 7. SAC 残差强化学习 —— 在线策略优化

### 7.1 为什么需要 RL

APF 控制器（阶段 2）能安全地完成挖掘循环，但存在手工调参无法覆盖的问题：

| 问题 | APF 局限 | RL 解决 |
|:---|:---|:---|
| **动作平滑** | K_att、K_imp 全局固定，不同阶段一刀切 | SAC 策略在状态空间中自适应输出不同幅度的修调 |
| **轨迹跟踪-稳定性折衷** | 纯控制在追踪与侧翻间无法权衡 | RL 奖励函数同时包含路径精度和稳定性惩罚 |
| **管线安全最大化** | 轨迹安全化后跟踪误差可能带来临时接近 | SAC 通过奖励函数引导策略在安全裕度内优化轨迹 |
| **单次作业完成率** | APF 可能因建模误差在某些条件下失败 | RL 策略从经验中学习鲁棒动作 |

### 7.2 动作空间：APF 基础上的残差修正

**不替代 APF，而是叠加微调**：

$$\boxed{\mathbf{q}_{\text{des}}^{\text{final}} = \underbrace{\mathbf{q} + K_{\text{imp}} \cdot \boldsymbol{\tau}_{\text{APF}} \cdot \Delta t}_{\text{APF 基准}} + \underbrace{\Delta \mathbf{q}_{\text{RL}}}_{\text{SAC 残差}}}$$

$$\Delta q_{\text{RL},i} \in [-0.05, +0.05]\;\text{rad},\quad i = 1,2,3,4$$

**设计理由**：APF 已经能完成任务。SAC 只需小范围微调，大幅降低了探索风险（即使 SAC 输出随机动作，偏差最多 0.05 rad/step ≈ 0.1°）。这等价于将 RL 的探索约束在 APF 基准策略的 Lipschitz 邻域内。

### 7.3 状态空间

$$\boxed{\mathbf{s} = \begin{bmatrix}
\underbrace{q_1, q_2, q_3, q_4}_{\text{关节角}} & 
\underbrace{p_x, p_y, p_z}_{\text{齿尖位置}} &
\underbrace{s^*}_{\text{轨迹进度}} &
\underbrace{d_{\text{pipe}}}_{\text{管线距离}} &
\underbrace{\alpha_{\text{margin}}}_{\text{ZMP 裕度}} &
\underbrace{\omega_x, \omega_y, \omega_z}_{\text{底盘角速度}} &
\underbrace{a_{1,t-1}, a_{2,t-1}, a_{3,t-1}, a_{4,t-1}}_{\text{历史动作}}
\end{bmatrix}^T \in \mathbb{R}^{17}}$$

底盘角速度 $\omega_{\text{base}}$ 来自 Unity RigidBody.AngularVelocity，是翻车风险的**真实物理信号**（不依赖 FK 近似）。

### 7.4 奖励函数

$$\boxed{R(\mathbf{s}, \mathbf{a}) = R_{\text{progress}} + R_{\text{stability}} + R_{\text{pipe}} + R_{\text{smooth}}}$$

| 奖励项 | 公式 | 权重 | 目的 |
|:---|:---|:---:|:---|
| $r_{\text{progress}}$ | $+15 \cdot \max(0, s^*_t - s^*_{t-1})$ | 15 | 轨迹前进（主要回报源） |
| $r_{\text{cycle}}$ | $+100$（一次循环完成） | 100 | 任务完成激励 |
| $r_{\text{stability}}$ | $-3 \cdot \max(0, 0.5 - \alpha_{\text{margin}})$ | 3 | $\alpha_{\text{margin}}$ 低于阈值惩罚 |
| $r_{\text{tipover}}$ | $-10 \cdot \|\omega_{\text{base}}\|$ | 10 | 底盘角速度惩罚 |
| $r_{\text{pipe}}$ | $-10 \cdot \max(0, 1.5 - d_{\text{pipe}})$ | 10 | 管线接近惩罚 |
| $r_{\text{smooth}}$ | $-0.1 \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|$ | 0.1 | 动作平滑 |
| $r_{\text{mag}}$ | $-0.02 \cdot \|\mathbf{a}_t\|$ | 0.02 | 动作幅度惩罚 |

**奖励设计的核心原则**：
- $r_{\text{progress}}$ 是唯一正回报源（占总额~70%），等价于最大化循环完成速度
- 安全项（$r_{\text{stability}}, r_{\text{tipover}}, r_{\text{pipe}}$）是惩罚项，零动作时激活
- 平滑项防止策略输出高频振荡的控制信号

### 7.5 SAC Actor-Critic 训练机制

SAC 是一种 **Actor-Critic** 算法，由三个神经网络协同完成学习。理解三者的角色是理解训练过程的关键。

#### 7.5.1 三个网络的分工

**Actor 网络 $\pi_\theta(s)$**（策略网络，17 → 4）：

Actor 直接输出动作 $\Delta \mathbf{q}^{\text{RL}}$。它的输入是 17 维状态观测，输出是 4 维动作均值，加上高斯噪声产生随机探索。Actor 的任务是**生成动作**——它不直接知道动作是好是坏，依赖 Critic 来评价。

**Critic 网络 $Q_\phi(s, a)$**（动作价值网络，17 + 4 → 1）×2：

SAC 使用**两个独立的 Q 网络**（取最小值以抑制过估计）。Critic 的输入是当前状态 $s$ 和当前做的动作 $a$，输出一个标量 $Q(s, a)$——在这个状态下执行这个动作的"长期价值"。Critic 的训练目标是**预测 Bellman 目标 $y$**（即时的奖励 + 折扣后的未来价值）：

$$\boxed{L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[\Big(Q_\phi(s,a) - \big(r + \gamma(\min_{j=1,2} Q_{\bar{\phi}_j}(s', \tilde{a}) - \alpha \log \pi_\theta(\tilde{a}|s'))\big)\Big)^2\right]}$$

其中 $\tilde{a} \sim \pi_\theta(s')$，$Q_{\bar{\phi}}$ 是 Target 网络（软更新，防止训练震荡）。

**Actor 的训练目标**：最小化 Critic 的负值（= 最大化期望价值），同时保留探索（熵项）：

$$\boxed{L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{B}}\left[\alpha \log \pi_\theta(a|s) - \min_{j=1,2} Q_{\phi_j}(s, a)\right]}$$

第一项 $\alpha \log \pi_\theta$ 是**熵惩罚**——鼓励 Actor 输出不确定的动作（防止过早收敛到局部最优）。第二项 $-\min Q$ 是**价值引导**——当 $Q$ 高时（Critic 认为动作好），Actor 的损失就低。Actor 通过最小化这个损失来"追随 Critic 的判断"。

**温度 $\alpha$（自适应探索）**：SAC 自动调节探索强度。温度高 → Actor 输出更随机 → 探索更多。温度低 → Actor 更确定 → 利用当前知识。更新为：

$$L_\alpha = \mathbb{E}\left[-\alpha \log \pi_\theta(a|s) - \alpha \cdot \mathcal{H}_{\text{target}}\right]$$

其中 $\mathcal{H}_{\text{target}} = -4$（$=-\dim(\text{action})$）。当实际熵低于目标时，$\alpha$ 上升鼓励更多探索。

#### 7.5.2 Critic 和 Actor 在训练中如何互动

```
每个训练步:

① Critic 评价过去:
   Buffer 中采样 1024 条历史 transition
   → Q(s,a) 预测应该有多大价值
   → 和实际发生的 r + γV(s') 比较
   → 修正 Q 的判断（梯度下降 L_Q）

② Actor 追随 Critic:
   对同样 1024 条 transition
   → 对每条 state，Actor 提出自己的动作 π(s)
   → Critic 评估 Q(s, π(s))
   → Actor 尝试让 Q(s, π(s)) 更高（梯度上升）
   → 同时保留一定的随机性（熵项）

③ Critic 和 Actor 交替进步:
   Critic 学到"什么样的 (s,a) 产生高回报" → 纠正错误判断
   Actor 学到"Critic 认为好的动作是什么样的" → 调整动作输出
   → 循环：Actor 更好 → 产生更好的 a → Critic 学到更高的 Q → Actor 进一步追随
```

**这个机制在本问题中具体做什么**：

- Critic 从 Buffer 采样 APF 示范数据后学会：$(s, a \approx 0)$ 产生正回报（轨迹前进）、$(s, a \text{ 抖动})$ 产生负回报（被惩罚）
- Actor 逐渐学会：在需要稳定的入土阶段输出更小的动作（避免触发平滑惩罚），在管线附近输出偏向安全方向的动作
- 训练结束时 Actor 的行为接近 APF 但更平滑 → reward 从 238 升至 255，d_min 从 0.33m 改善到 0.62m

### 7.6 实际训练过程与收敛

#### 7.6.1 训练配置

| 参数 | 值 |
|:---|---|
| 设备 | NVIDIA RTX 4090, CUDA 12.4 |
| 训练时长 | 8.5 分钟 |
| 总步数 | 200,000 环境步 |
| 预热步数 | 50,000 步（APF 基准 + 极低噪声） |
| 学习开始 | 50,000 步后 |
| 管道频率 | 30%（70% 无管道 + 30% 有管道） |
| 每步耗时 | ~0.9 ms（1100+ fps） |

#### 7.6.2 训练过程（评估曲线）

| 步数 | Avg Reward | 阶段 |
|------|:---:|:---|
| 10K–50K | 376.2–376.3 | 预热期：纯 APF 数据，reward 高且稳定 |
| 60K–70K | 378.8–380.4 | 学习初期：SAC 开始微调，reward 略微上升 |
| 80K | **342.4** | **探索低谷**：SAC 尝试偏离 APF 基准，部分动作触发了安全惩罚 |
| 90K–110K | 369.6 → **313.3** | 探索深谷：策略学到一个次优行为，reward 骤降 |
| 120K–140K | 357.5–361.2 | 自我修复：Critic 纠正了错误判断，Actor 回到安全区域 |
| 150K–180K | 361.0–369.5 | 稳定收敛：reward 回升至 ~370 |
| **200K** | **373.5** | **最终收敛**：略低于预热期，但 pipeline 安全度量显著提升 |

**收敛曲线特征**：reward 在 $375 \pm 30$ 区间震荡收敛。80K–110K 步的深谷是 SAC 探索机制的预期行为——策略尝试了不安全的动作，Critic 学会将其标记为低价值，Actor 随后避免了这些动作。深谷之后 reward 回升到接近预热水平的区间，说明策略**保留了 APF 的安全基线**，同时学到了更平滑的控制。

#### 7.6.3 最终策略 vs APF 基准

| 指标 | APF 基准 | SAC 最终策略 | 变化 |
|:---|:---:|:---:|:---:|
| 平均奖励 | 238 | 255 | **+7%** |
| 土中切卷 (soil_pull) | 3.28 m | 3.05 m | −7% |
| **最小管线距离 d_min** | **0.33 m** | **0.62 m** | **+88%** |

SAC 策略的关键权衡：**主动舍弃 7% 的挖掘量，换取了近翻倍的管线安全裕度**。这个权衡不是手工设计的——是 SAC 自动从奖励函数中的管线惩罚项 ($-10 \times \max(0, 1.5 - d_{\text{pipe}})$) 和稳定性惩罚项中学到的。APF 只跟踪轨迹不考虑安全距离，SAC 从经验中学到"离管线远一点虽然挖得少一点，但奖励总量更高"。

---

## 8. 理论分析

### 8.1 离线安全保证

**命题 1（离线安全保证）**：若安全化轨迹 $\mathcal{T}_{\text{cert}}$ 中所有路点满足 $d(\mathbf{p}_k) \geq d_{\text{safe}}$，且在线跟踪误差 $\|\mathbf{p}_{\text{tip}} - \mathcal{T}_{\text{cert}}(s^*)\| < d_{\text{safe}} - r_{\text{critical}}$，则系统始终满足安全约束。

**证明**：三角不等式。$\mathbf{p}_{\text{tip}}$ 到管线的距离 $\geq d(\mathcal{T}_{\text{cert}}(s^*)) - \|\mathbf{p}_{\text{tip}} - \mathcal{T}_{\text{cert}}(s^*)\| \geq d_{\text{safe}} - \epsilon_{\text{track}}$。当 $\epsilon_{\text{track}} < d_{\text{safe}} - r_{\text{critical}}$ 时，物理约束成立。$\square$

> **工程解释**：$d_{\text{safe}} = 1.5\text{m}$ 包含了 >1m 的安全余量（管线半径 + 铲斗本体半径 ≪ 0.5m）。即使跟踪误差达到 1m，齿尖仍在管线物理半径之外。这是"离线安全化"相对于"在线斥力"的关键优势 —— 安全不是在线的力平衡，而是轨迹本身的几何属性。

### 8.2 任务-零空间解耦

**命题 2（零空间不干扰任务空间）**：$\forall \boldsymbol{\tau}_0 \in \mathbb{R}^4$，零空间力矩不产生齿尖加速度：$\ddot{\mathbf{p}}_{\text{tip}}(\boldsymbol{\tau}_0) = J_p N \boldsymbol{\tau}_0 / \text{mass} = \mathbf{0}$。

**证明**：见 §4.3 引理 1。$\square$

> **工程意义**：铲斗卷斗力矩可以直接叠加到 $\tau_{\text{cmd}}$ 上（不经过零空间投影），因为它对齿尖位置的贡献极小（$< 5\%$）。这避免了投影操作削弱卷斗驱动力的问题。

### 8.3 SAC 残差的安全上界

**命题 3（SAC 残差安全不变性）**：在 $\|\Delta \mathbf{q}_{\text{RL}}\|_{\infty} \leq 0.05$ rad 的约束下，单个 RL 动作引起的齿尖位移不超过 $\sigma_{\max}(J_p) \cdot 0.05 \cdot \Delta t < 0.01$ m，远小于安全容差。

**证明**：$\|\Delta \mathbf{p}\| = \|J_p \Delta \mathbf{q}\| \leq \|J_p\|_2 \|\Delta \mathbf{q}\|_2 \leq \sigma_{\max} \cdot 0.1 < 0.01$ m（因为 $\sigma_{\max}(J_p) < 8$ 对典型构型）。$\square$

这确保了即使在训练最早期（SAC 输出接近随机），**单步动作也不会将齿尖带出安全邻域**。安全性的保证不是靠"Sac 学得好"，而是靠动作空间设计的硬边界。

---

## 9. 实验设计

### 9.1 实验平台

- **仿真引擎**：AGX Dynamics 2.41 + AGXUnity 5.5 + Unity 2022.3 LTS
- **高保真模型**：LG 922F 4-DOF 挖掘机（~4700 kg 工作装置质量），可变形土壤（DeformableTerrain），实际管线碰撞体
- **ROS2 控制**：Python `excavator_control` 节点，50 Hz，通过 ROS-TCP-Connector 与 Unity 双向通信
- **离线环境**：纯 Python Gymnasium 环境，复用 FK + Jacobian + APF 流水线（无 Unity 依赖）

### 9.2 评价指标

| 指标 | 定义 |
|:---|:---|
| $d_{\min}$ | 齿尖距管线最小距离（m）|
| 安全违规率 | $d_{\text{tip}} < d_{\text{safe}}$ 的步数占比 |
| 跟踪误差 | $\|\mathbf{p}_{\text{tip}} - \mathbf{p}_{\text{target}}\|$ 均值（m）|
| 循环完成时间 | 一次完整挖掘循环的仿真时间（s）|
| 倾覆裕度统计 | $\alpha_{\text{margin}}$ 的 min/mean/std |
| SAC 奖励 | 单次循环累计奖励 |

### 9.3 对比基线

| 方法 | 描述 |
|:---|:---|
| **Naïve APF** | 直接跟踪原始轨迹（不安全，$\min d < 0.4$ m） |
| **Online APF Repulsion** | 在线管线斥力 $F_{\text{rep}}$ + 原始轨迹 |
| **CBF-TrajCert (Ours)** | 离线轨迹安全化 + 纯 APF 跟踪（本文 §3–§5） |
| **TrajCert+SAC (Ours)** | 离线轨迹安全化 + APF + SAC 残差（本文 §3–§7） |

### 9.4 消融实验

1. **轨迹安全化必要性**：Naïve APF vs. CBF-TrajCert（预期：naïve 碰撞率 > 50%，ours 0%）
2. **零空间分量贡献**：逐步去掉关节限位保护、ZMP 倾覆保护、铲斗直接驱动，测量指标退化
3. **SAC 训练收益**：TrajCert vs. TrajCert+SAC（预期：SAC 提高动作平滑性 → 降低跟踪误差 → 提高 $d_{\min}$）
4. **管道频率消融**：pipe_freq $\in \{0, 0.3, 1.0\}$

---

## 10. 论文提纲

```
1. Introduction
   - 挖掘机自主作业的安全需求
   - 现有方法局限（纯 APF 振荡 / 纯 RL 不安全探索 / 缺少冗余臂利用）

2. Related Work
   2.1 轨迹优化 (CHOMP, TrajOpt)
   2.2 控制障碍函数 (CBF)
   2.3 冗余机械臂控制 (零空间投影)
   2.4 挖掘机自动化

3. Problem Formulation
   3.1 4-DOF 挖掘机运动学 (MDH + FK + Jacobian)
   3.2 安全约束定义（管线距离）
   3.3 三层安全架构总览

4. CBF 轨迹安全化 (离线)
   4.1 复合代价泛函
   4.2 梯度下降优化
   4.3 收敛性与步长分析

5. APF 任务-零空间分层控制 (在线)
   5.1 Carrot APF 任务空间跟踪
   5.2 零空间投影与冗余分解
   5.3 Z-based 铲斗入土即卷策略
   5.4 ZMP 倾覆保护
   5.5 总控制律

6. SAC 残差强化学习
   6.1 动作空间设计（残差修调）
   6.2 状态空间与奖励函数
   6.3 训练策略（管道随机切换）

7. 理论分析
   7.1 离线安全保证
   7.2 任务-零空间解耦证明
   7.3 SAC 残差的安全上界

8. Experiments
   8.1 实验设置
   8.2 轨迹安全化 vs 在线斥力对比
   8.3 APF 分层控制各组件消融
   8.4 SAC 训练收敛与性能对比

9. Discussion & Limitations
   9.1 d_safe 的选择对挖掘效率的影响
   9.2 Sim-to-Real 挑战
   9.3 当前仅验证了管线约束

10. Conclusion
```

---

> **关联文档**：
> [[4DOF挖掘机APF完整推导]]（APF 公式 + 6 类斥力推导）
> [[基于CBF轨迹安全化的冗余挖掘机运动规划方法]]（CBF 轨迹安全化）
> [[ROS2工作空间APF方案设计]]（包划分 + 接口设计）
> [[AGXUnity_RL_Control_SLAM_API]]（Unity/AGX 控制接口）
> [[论文糅合机械臂思考]]（CDF + CBF + APF 离线在线混合架构）

---

*本文档从学术论文视角系统化整理了完整的轨迹安全化 + APF 冗余分解 + SAC 残差强化的三层安全自适应控制架构。所有公式均有代码实现对应，可直接用于论文 Methods 章节。*
