# 挖掘机 TIE 路线图 —— 现状盘点 → 下一步行动

> 定位：这是"第二步"交付文档 —— 把你**已经有的东西**和**该做的事**对照清楚。
> 数学推导不在这里重复，见 [rl/docs/SAC_excavator_theory.md](rl/docs/SAC_excavator_theory.md)（你已经写好的方法论全文）。
> 本文只回答两个问题：**(1) 现在代码里到底有什么；(2) 从"能跑"到"能发 TIE"还差哪几步。**
> 最后更新：2026-09-01

---

## 0. TL;DR（一句话结论）

你已经有一个**能跑通的完整闭环**：Unity/AGX 真实土壤 → ROS2 → APF 标称轨迹跟踪 + SAC 残差学习，铲斗内质量/体积已能量测。

但离 TIE 还差四件**硬差距**（详见第 2 节）：

| # | 差距 | 一句话 |
|---|------|--------|
| G1 | 在线奖励**没有管线惩罚** | `unity_env.py` 只给 `Δvolume`，和离线训练目标不一致 |
| G2 | 离线环境**没有土壤物理** | 离线"土量"奖励是运动学代理量（齿尖入土 X 向拉距），不是真实挖土量 → 残差语义错位 |
| G3 | **CBF 和 APF 管线斥力都是死代码** | `cbf.py`、`pipe_repulsive_force` 写好了但从未被 import，在线安全只靠"离线安全轨迹 + 学出来的避让"，**没有硬安全保证** |
| G4 | **MPPI 还没动手** | 你选的 TIE 方案二是 MPPI 规划 + 残差，现在标称规划器还是 APF |

---

## 1. 现状盘点（代码 ↔ 理论 对照）

### 1.1 运动学与稳定性

| 文件 | 职责 | 关键定义 |
|------|------|----------|
| `src/excavator_kinematics/excavator_kinematics/mdh.py` | MDH 正运动学 | `MDH_PARAMS`（H=1.0403, a_boom=0.6509, a_arm=5.7101, a_bucket=2.915, a_tip=1.4592）；`forward_kinematics(q)→(p_tip, d_tip)`；`jacobian_position(q)`（数值，3×4）；关节限位 `q2∈[-0.8033,1.155]` `q3∈[-2.7594,-0.6084]` `q4∈[-2.614,0.4433]`。**FK 已标定到 0.0003 m 误差** |
| `src/excavator_kinematics/excavator_kinematics/zmp.py` | ZMP 倾覆裕度 | 各连杆质量（base 5000 / body 9800 / boom 2200 / arm 1500 / bucket 1000 kg，AGX 实测）；`zmp_alpha_margin(q)→(α_margin, zmp)`，支撑多边形 SUP_X=1.82 / SUP_Y=1.20，准静态 CoM 投影近似 |

### 1.2 轨迹与安全

| 文件 | 职责 | 关键定义 |
|------|------|----------|
| `src/excavator_trajectory/excavator_trajectory/trajectory.py` | 示教轨迹 + 离线安全化 | `PIPE_POS=[6.6301, 0, -1.2]`（Σ0 系，对应 Unity world (-7.5,-1.2,0)）；`PIPE_CLEARANCE=1.2`；`WAYPOINTS`（64 点，遥操采集）；`SAFE_WAYPOINTS`（抬升/平滑后的安全轨迹）；`find_closest / find_closest_continuous`；`get_point(s) / get_q_at_s(s)`；`certify_trajectory(...)`（离线梯度下降安全化） |

### 1.3 控制器

| 文件 | 职责 | 关键定义 |
|------|------|----------|
| `src/excavator_controller/excavator_controller/apf.py` | APF 各项力/矩 | `attractive_force/torque`；`joint_limit_torque`（软限位斥力，已修符号 bug）；`tipover_torque`（ZMP L1 保护，α_thresh=0.4）；`pipe_repulsive_force`（管线屏障斥力，**未被调用**，见 G3） |
| `src/excavator_controller/excavator_controller/cbf.py` | 在线 CBF 滤波 | `pipe_cbf_from_apf(...)`（h=1/(1+u)−δ）；`cbf(v_des,h,grad)`（离散单约束闭式 QP 投影，**未被调用**，见 G3） |
| `src/excavator_control/excavator_control/control.py` | 在线 ROS 控制主循环 | 增益 `K_att=300, K_imp=0.012, K_joint=60, K_joint_bucket=250`；`ros_offset=[0,0.765,-0.743,-0.05]`；终端状态机；Z-based 卷斗（`_Q4_IN_SOIL=-1.4`）；SAC 残差接入点 `q_des += _dq_sac` |
| `src/excavator_control/excavator_control/terminal_state.py` | 一次作业终端判定 | 进保持：`s≥0.93` 且 `path_err≤0.35`；完成：`endpoint_err≤0.30` 且 `|q4−q4_terminal|≤0.08`；保持超时 8 s |
| `src/excavator_control/excavator_control/reference_progress.py` | 参考进度推进 | `advance_reference_progress`（单调、限速 step=0.01） |

### 1.4 RL 训练与推理

| 文件 | 职责 | 关键定义 |
|------|------|----------|
| `rl/env/excavator_env.py` | **离线**运动学环境 | 17 维 obs（q×4, p_tip×3, s, d_pipe, α, base_angvel×3, action_prev×4），**d_pipe 在第 8 维**；动作 `action_max=0.01`；奖励含 `w_pipe=3·max(0, 1.5−d_pipe)`、`w_soil=80·土壤拉距·cycle`、`w_tipover=10`、`w_track=3·max(0,path_err−0.70)` 等。**土壤奖励是运动学代理量 `_soil_pull`**（齿尖 z<0 时 X 向拉距），不是真实挖土量（见 G2） |
| `rl/env/unity_env.py` | **在线**真实土壤环境 | 同一 17 维 obs；`APFStepper` 复用 control.py 逻辑；奖励 `reward_scale·Δsoil_volume`，**没有管线惩罚**（见 G1）；默认 `action_max=0.03`（与离线 0.01 不一致） |
| `rl/train/train_sac.py` | SAC 离线训练 | SB3 SAC，`lr=5e-5`、`batch=1024`、`gamma=0.999`、`target_entropy=-4`；50k APF 预热 buffer；`PipeSwitchWrapper`（70% 无管 / 30% 有管）；导出 ONNX + `obs_norm.npz` |
| `rl/train/train_online.py` | SAC 在线微调 | `--dry-run / --total-steps / --reward-scale / --action-max / --warmup-steps` |
| `rl/inference/sac_wrapper.py` | 推理包装 | onnxruntime，`action_clip=0.01`，自动加载 obs_norm，`build_obs(...)` 与两个 env 的 17 维对齐 |
| `rl/models/sac_run/` | 已训练模型 | `sac_final.onnx/zip`、obs_norm、vec_normalize、各 checkpoint、convergence.png |

### 1.5 Unity 侧（excavator_xie，**改动需先经你同意**）

| 文件 | 职责 |
|------|------|
| `Assets/.../Scripts/MassVolumeCounter.cs` | 发布 `/unity/soil_volume`（`getInnerSoilBulkVolume`）、`/unity/bucket_mass`（`getInnerSoilMass`）；订阅 `/unity/reset_terrain` → `ResetEpisode()` → `ResetHeights()` |
| `Assets/.../ExcavatorE85_Terrain.asset` | 1.9 MB 地形高度图（git 已还原，勿再触碰） |

---

## 2. 三个关键发现（你务必先知道）

### 发现 A：在线控制回路里**没有显式管线避障**，安全全靠"轨迹 + 学出来的避让"

`control.py` 的 import 列表里**只有** `attractive_force, attractive_torque, joint_limit_torque, tipover_torque`。

- `pipe_repulsive_force`（APF 管线屏障斥力）——**没被 import**。
- `cbf.py` 的在线 CBF 滤波——**没有任何文件 import**。

所以现在在线跑的时候，管线安全实际来自两处：
1. `SAFE_WAYPOINTS`（离线抬升过的安全轨迹）——跟踪它就不会撞。
2. SAC 残差在离线训练时学到的避让（`w_pipe` 软惩罚）。

这两者都**不是硬保证**。TIE 审稿人会问"你怎么保证永不碰管"，现在答案是"轨迹安全 + 概率性的学习避让"——这个答案发 TIE 不够硬。

### 发现 B：离线环境的"土量奖励"是假的（运动学代理），在线才是真的

- 离线 `excavator_env.py`：`_soil_pull` = 齿尖 z<0 时 X 向后退位移的累积。这是**运动学代理量**，没算任何土壤力学/挖掘阻力。
- 在线 `unity_env.py`：`/unity/soil_volume` = AGX `getInnerSoilBulkVolume()`，是**真实挖土体积**。

→ 离线训练的 SAC，其"土量"奖励和真实挖土量不是一回事。残差学习的语义现在是"在运动学代理上微调"，不是"补偿土壤动力学建模误差"。这是方法学上最需要想清楚的一条（见第 5 节决策点 D1）。

### 发现 C：离线/在线目标不一致 + 动作范围不一致

- 离线训练目标 = 跟踪 + 避管 + 挖土代理量；在线奖励 = 纯 `Δvolume`。
- 离线 `action_max=0.01`，在线默认 `0.03`，推理 wrapper 又 `clip=0.01`。

→ 在线微调会朝"挖得多"单向跑，丢掉避管信号；三个地方的动作范围不一致会让微调结果和部署时行为对不上。

---

## 3. 分阶段执行计划（从"能跑"到"能发 TIE"）

### Phase 0 —— 对齐基线（本周，纯工程，不碰 Unity）

**目标**：让离线/在线目标一致，形成可复现的"APF+SAC 基线"。

- [ ] **P0-1** 在线 `unity_env.py` 加管线惩罚 + 撞管惩罚（对齐离线 `_compute_reward`）：
  ```python
  d_safe_hard = r_pipe + margin          # = 0.2 + 0.3 = 0.5 m
  reward = reward_scale * (vol_after - vol_before) \
         - w_pipe  * max(0.0, d_safe - d_pipe) \      # 软惩罚, 1.5 m 内开始扣
         - w_hit   * (d_pipe < d_safe_hard)           # 撞管, < 0.5 m 重罚
  ```
  `r_pipe=0.2`、`margin=0.3` 已确定（见第 5 节 D2）。
- [ ] **P0-2** 统一 `action_max`（三处统一到 0.01）。
- [ ] **P0-3** 跑一遍 `train_online.py --dry-run`，确认在线奖励/obs 管道通。

**验收**：在线一集能同时看到"挖土量"和"d_pipe"两个量，撞管时奖励明显为负。

### Phase 1 —— 基线实验（补论文对比表）

**目标**：拿到论文第一章实验表格所需的基线数据。

- [ ] **P1-1** 三组基线对比：
  1. 纯 APF 轨迹跟踪（`--sac-model ''`，禁 SAC）；
  2. APF + SAC 残差（现状）；
  3. 纯 RL（可选，若时间紧可只做前两组）。
- [ ] **P1-2** 每集记录指标：`Σ挖土量(kg)`、`min d_pipe`、`min α_margin`、`周期数`、`成功率`。`control.py` 已有 `cycle_data` 落盘（`data/cycle_data_*.npz`），把挖土量也接进记录。

**验收**：一张"挖得多 vs 离管远 vs 稳定"的三方对比表。

### Phase 2 —— 把 CBF 从死代码变成在线硬约束（安全保证）

**目标**：给"永不碰管"一个硬保证，这是 TIE 安全故事的核心。

- [ ] **P2-1** 定管线模型：把 `PIPE_POS` 扩展成圆柱 `(中心, 半径 r_pipe)`，用**距离型 CBF** `h = d(p_tip, 管轴) − (r_pipe + margin)` 替代现在 `cbf.py` 里基于 APF 势函数的 `h=1/(1+u)−δ`（后者难证 `h≥0 ⟹ 不碰管`）。
- [ ] **P2-2** 把 CBF 接进 `control.py`：在 `q_des = q + delta_q + dq_sac` 之后、滤波之前，对**齿尖速度**做闭式 QP 投影（`cbf.py` 已有单约束闭式解，改 h 即可复用）。
- [ ] **P2-3** 从齿尖速度投影反解回关节增量（`Δq = J_p⁺ · Δv_safe` + 零空间补全），保持不破坏挖掘动作。

**验收**：故意把 SAC 残差放大/给一个朝管的扰动，齿尖仍不越过 `r_pipe+margin`。

### Phase 3 —— MPPI 替代 APF 作标称规划器（TIE 核心贡献）

**目标**：把"标称规划"从 APF 升级为 MPPI（模型预测路径积分），这是你方案二区别于方案三（纯 APF+SAC）的关键卖点。

- [ ] **P3-1** 先建一个**运动学 + 土壤阻力项**的简化预测模型（可先纯运动学起步，后加 `F_soil ∝ 挖土体积 × 阻力系数`）。
- [ ] **P3-2** 实现 MPPI 滚动优化：代价 = `−挖土量 + 碰管代价 + 能量/抖动 + 末端误差`，采样轨迹，重要性加权求期望控制。
- [ ] **P3-3** SAC 残差从"叠在 APF 上"改成"叠在 MPPI 上"（`q_des_MPPI + Δq_SAC`），残差学习 MPPI 预测模型未建模的土壤动力学。
- [ ] **P3-4** CBF 继续压在 MPPI+SAC 输出之后，作为最后一层硬安全。

**验收**：MPPI 标称 + SAC 残差 + CBF 三层跑通，挖土量 ≥ APF 基线，且 d_pipe 全程 ≥ 安全阈。

### Phase 4 —— 理论补强 + 实验 + 论文

- [ ] **P4-1** 理论：CBF 安全性证明（前向不变集）、MPPI 的样本复杂度/收敛、残差学习的有界性论证（这些骨架 `SAC_excavator_theory.md` 里已有雏形，按 Phase 2/3 的实现回填）。
- [ ] **P4-2** 消融实验（论文核心贡献表）：`APF vs MPPI`、`+CBF vs −CBF`、`+残差 vs −残差`，每格 20+ 集统计。
- [ ] **P4-3** 真实挖掘机**图片** demo（定性展示，不做真 sim2real，符合导师要速度的要求）。
- [ ] **P4-4** 按 `SAC_excavator_theory.md` §10 的论文提纲成文。

---

## 4. 立即可做（本周内，无需等任何决策）

1. **P0-1 + P0-2**：给 `unity_env.py` 加管线惩罚、统一 `action_max`。改的是 `ex_ws/rl/`，不碰 Unity，可直接做。
2. **读一遍 `SAC_excavator_theory.md` §3**：确认你心里的"CBF 安全化"目前只实现了**离线轨迹**那一半，在线 CBF（`cbf.py`）还没接——这决定 Phase 2 的工作量。

---

## 5. 需要你拍板的决策点

| # | 决策 | 我的建议 | 影响 |
|---|------|----------|------|
| **D1** | 离线环境的"土壤"怎么处理？ | **保持运动学代理 + 在线用真实土量微调**（符合导师"要速度"），论文里把残差学习的语义写成"补偿真实土壤动力学与标称模型的偏差"，而不是声称离线就有土壤模型 | 决定 G2 是"接受"还是"补模型"，影响论文方法学表述 |
| **D2** ✅ 已定 | 管线半径 `r_pipe` 与安全裕度 `margin` | `r_pipe = 0.2 m`（Unity `underground_pipe` 的 AGX Cylinder `m_radius: 0.2`，高 6 m，沿 Y 轴）；`margin = 0.3 m` → **硬安全阈 `d_safe_hard = 0.5 m`**（即 `d_pipe` ≥ 0.5 m 不碰管）。软惩罚半径 `d_safe=1.5` 更早、硬阈值更底，不冲突 | 决定 Phase 0 撞管惩罚和 Phase 2 CBF 的 `h = d − 0.5` |
| **D3** | MPPI 预测模型先用纯运动学还是直接带土壤阻力？ | **先纯运动学起步跑通，再逐步加土壤阻力项**（别一开始就啃 AGX 土壤接触力学） | 决定 Phase 3 的难度与时间 |
| **D4** | 三组基线里"纯 RL"要不要做？ | 时间紧可只做 APF vs APF+SAC，纯 RL 作为"补充对比"后补 | 决定实验表规模 |

---

## 附：文件改动权限备忘

- **`ex_ws/`（本工作区）**：可直接改（本路线图涉及的所有 Python 改动都在这里）。
- **`excavator_xie/`（Unity 工程）**：**任何改动先告知你、经你同意后再动手**。Phase 0–4 里唯一可能碰 Unity 的是 P0 之后的"撞管检测"若需要在 Unity 里加碰撞传感器，届时单独跟你确认。
