#!/usr/bin/env python3
"""
SAC 在线真实土壤训练 — 直接驱动 Unity/AGX (通过 ROS).

用法:
  # 1. 先 dry-run: 跑几个 APF 基线 episode, 验证 ROS 布线 + 观察挖土量/奖励量级
  python rl/train/train_online.py --dry-run --episodes 3

  # 2. 在线训练 (每个 episode 都是一次真实挖掘, 非常耗时)
  python rl/train/train_online.py --total-steps 100000 --save-dir rl/models/sac_online

说明:
  - 训练是实时的: 每步 ≈ 20ms (Unity 50Hz), 一个 episode ≈ 600~1500 步.
    100k 步 ≈ 30+ 分钟. 先 dry-run 确认正常再长时间跑.
  - 奖励 = reward_scale * 累计挖土体积增量. 若 dry-run 显示单 episode 挖土体积
    很小/很大, 用 --reward-scale 把单 episode 总奖励调到 ~10~100 量级.
"""
import os
import sys
import time
import json
import warnings
from pathlib import Path

_WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WS / 'rl'))
sys.path.insert(0, str(_WS / 'rl/env'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import rclpy  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback, EvalCallback, CheckpointCallback,
)

from unity_env import UnityExcavatorEnv  # noqa: E402


SAC_CONFIG = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], qf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    ),
    learning_rate=3e-4,          # 在线数据稀缺, lr 略高于离线
    batch_size=256,
    tau=0.005,
    gamma=0.995,
    ent_coef='auto',
    target_entropy=-4.0,
    buffer_size=200_000,
    learning_starts=2_000,
    gradient_steps=4,            # 每步多学几次, 榨干稀缺数据
    train_freq=4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)


class ProgressCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 500 == 0:
            buf = self.model.replay_buffer
            ep = self.model.ep_info_buffer
            avg_r = np.mean([e['r'] for e in ep[-5:]]) if len(ep) else float('nan')
            print(f"  step {self.num_timesteps:>7d}: avg_r(last5)={avg_r:>7.1f} "
                  f"buffer={buf.size()}", flush=True)
        return True


def dry_run(env: UnityExcavatorEnv, episodes: int = 3):
    """零动作跑连续作业循环基线, 打印累计挖土量 / 完成循环数 / 最小离管距."""
    print(f"[dry-run] 运行 {episodes} 个连续作业 episode (零动作)...", flush=True)
    for ep in range(episodes):
        obs, _ = env.reset()
        steps = 0
        done = truncated = False
        total_dug = 0.0
        min_d_pipe = float('inf')
        total_reward = 0.0
        n_cycles = 0
        last_phase = 'dig'
        while not (done or truncated):
            obs, rew, done, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            total_reward += rew
            total_dug += info.get('dv', 0.0)
            min_d_pipe = min(min_d_pipe, info.get('d_pipe', float('inf')))
            n_cycles = info.get('cycle', 0)
            last_phase = info.get('phase', 'dig')
            steps += 1
        print(f"  episode {ep + 1}: steps={steps}  累计挖土={total_dug:.4f} m³ "
              f"完成循环={n_cycles}  min_d_pipe={min_d_pipe:.2f} m "
              f"总奖励={total_reward:.1f}  末阶段={last_phase}", flush=True)
    print("[dry-run] 完成. 累计挖土>0 说明装到土; "
          "完成循环≥1 说明挖→摆90°→倒→回位全流程通; "
          "min_d_pipe 应 ≥ 0.5 m (r_pipe+margin).", flush=True)


def train(total_steps: int, warmup_steps: int, save_dir: str,
          action_max: float, reward_scale: float,
          r_pipe: float, margin: float, d_safe: float,
          w_pipe: float, w_hit: float, max_steps: int):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = rclpy.create_node('sac_online_trainer')

    def make_env():
        return UnityExcavatorEnv(
            node=node, action_max=action_max, reward_scale=reward_scale,
            r_pipe=r_pipe, margin=margin, d_safe=d_safe,
            w_pipe=w_pipe, w_hit=w_hit, max_steps=max_steps,
        )

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    action_noise = NormalActionNoise(
        mean=np.zeros(4, dtype=np.float32),
        sigma=np.full(4, action_max * 0.3, dtype=np.float32),
    )

    model = SAC(env=venv, verbose=1, action_noise=action_noise, seed=42, **SAC_CONFIG)

    # ── 预灌 buffer: 零动作 APF 基线数据 (可选, 帮助 SAC 先学会基线奖励) ──
    if warmup_steps > 0:
        print(f"[warmup] 预灌 {warmup_steps} 步 APF 基线数据...", flush=True)
        obs = venv.reset()
        n = 0
        while n < warmup_steps:
            act = np.random.normal(0, action_max * 0.05, size=(1, 4)).astype(np.float32)
            act = np.clip(act, -action_max, action_max)
            next_obs, rew, done, info = venv.step(act)
            model.replay_buffer.add(
                obs=obs.copy(), next_obs=next_obs.copy(), action=act.copy(),
                reward=rew.copy(), done=np.array([done[0]], dtype=bool),
                infos=[info[0]],
            )
            n += 1
            obs = next_obs
            if done[0]:
                obs = venv.reset()
        obs = venv.reset()

    callbacks = [ProgressCallback()]
    if total_steps >= 10000:
        callbacks.append(CheckpointCallback(
            save_freq=max(10000, total_steps // 5),
            save_path=str(save_dir), name_prefix='online_ckpt',
        ))

    t0 = time.time()
    print(f"[train] 开始在线训练 {total_steps:,} 步 (实时, 请耐心)...", flush=True)
    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=False)
    print(f"[train] 完成, 用时 {(time.time() - t0) / 60:.1f} min", flush=True)

    # ── 保存 ──
    model.save(str(save_dir / 'sac_final'))
    venv.save(str(save_dir / 'vec_normalize.pkl'))
    _obs_rms = venv.obs_rms
    np.savez(str(save_dir / 'obs_norm.npz'),
             mean=_obs_rms.mean.astype(np.float32),
             std=np.sqrt(_obs_rms.var).clip(1e-6).astype(np.float32))

    # ── 导出 ONNX (供 control.py 推理) ──
    try:
        _obs_sample = torch.randn(1, 17)
        torch.onnx.export(
            model.policy.actor, _obs_sample, str(save_dir / 'sac_final.onnx'),
            input_names=['obs'], output_names=['action_mean'], opset_version=17,
        )
        print(f"[export] ONNX → {save_dir / 'sac_final.onnx'}", flush=True)
    except Exception as e:
        print(f"[export] ONNX 导出失败: {e}", flush=True)

    (save_dir / 'train_config.json').write_text(json.dumps({
        'total_steps': total_steps, 'warmup_steps': warmup_steps,
        'action_max': action_max, 'reward_scale': reward_scale,
        'r_pipe': r_pipe, 'margin': margin, 'd_safe': d_safe,
        'w_pipe': w_pipe, 'w_hit': w_hit, 'max_steps': max_steps,
        'device': SAC_CONFIG['device'],
    }, indent=2))

    venv.close()
    node.destroy_node()
    rclpy.shutdown()
    print(f"[done] 模型已保存 → {save_dir / 'sac_final.zip'}", flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='SAC online training on real soil')
    ap.add_argument('--total-steps', type=int, default=100_000)
    ap.add_argument('--warmup-steps', type=int, default=2_000)
    ap.add_argument('--save-dir', type=str, default=str(_WS / 'rl/models/sac_online'))
    ap.add_argument('--action-max', type=float, default=0.01)
    ap.add_argument('--reward-scale', type=float, default=100.0)
    ap.add_argument('--r-pipe', type=float, default=0.2)
    ap.add_argument('--margin', type=float, default=0.3)
    ap.add_argument('--d-safe', type=float, default=1.5)
    ap.add_argument('--w-pipe', type=float, default=3.0)
    ap.add_argument('--w-hit', type=float, default=100.0)
    ap.add_argument('--max-steps', type=int, default=3000, help='每 episode 最大步数')
    ap.add_argument('--dry-run', action='store_true', help='零动作跑基线, 不训练')
    ap.add_argument('--episodes', type=int, default=3, help='dry-run 的 episode 数')
    args = ap.parse_args()

    if args.dry_run:
        rclpy.init()
        node = rclpy.create_node('sac_online_dryrun')
        env = UnityExcavatorEnv(
            node=node, action_max=args.action_max, reward_scale=args.reward_scale,
            r_pipe=args.r_pipe, margin=args.margin, d_safe=args.d_safe,
            w_pipe=args.w_pipe, w_hit=args.w_hit, max_steps=args.max_steps)
        try:
            dry_run(env, args.episodes)
        finally:
            env.close()
            node.destroy_node()
            rclpy.shutdown()
    else:
        train(args.total_steps, args.warmup_steps, args.save_dir,
              args.action_max, args.reward_scale,
              args.r_pipe, args.margin, args.d_safe,
              args.w_pipe, args.w_hit, args.max_steps)
