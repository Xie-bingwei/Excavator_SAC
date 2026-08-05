#!/usr/bin/env python3
"""
SAC 离线训练 — 不依赖 Unity, 纯运动学环境.

用法:
  # 1. 先灌 buffer (可选, 但推荐)
  python rl/train/warmup.py -n 100000

  # 2. 训练
  python rl/train/train_sac.py --total-steps 200000 --warmup-buffer rl/models/warmup.npz

  # 3. 导出 onnx (推理)
  python rl/train/train_sac.py --export rl/models/sac_final.zip
"""
import sys
from pathlib import Path
_WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WS / 'rl'))
sys.path.insert(0, str(_WS / 'rl/env'))

import os
import time
import json
import warnings
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

os.environ['TRAJECTORY_GENERATING'] = '1'
import torch

# Suppress SB3 cleanup warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from excavator_env import ExcavatorEnv, ExcavatorEnvConfig


# ═══════════════════════════════════════════════════
#  超参数 — 保守设置, 确保收敛
# ═══════════════════════════════════════════════════

SAC_CONFIG = dict(
    # ── 网络 ──
    policy="MlpPolicy",
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], qf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    ),
    # ── 优化 ──
    learning_rate=5e-5,           # 极低 lr → 不偏离 APF 安全基线
    batch_size=1024,              # 更大 batch → 梯度稳定
    tau=0.005,                    # 软更新
    gamma=0.99,
    # ── 探索 ──
    ent_coef='auto',
    target_entropy=-4.0,          # -dim(action)
    # ── Buffer ──
    buffer_size=500_000,
    learning_starts=50_000,       # 先充分收集数据再开始学习
    gradient_steps=2,             # 每步学 2 次
    train_freq=4,                 # 每 4 步学一次
    # ── 设备 ──
    device='cuda' if torch.cuda.is_available() else 'cpu',
)


class ProgressCallback(BaseCallback):
    """Print training status every 10k steps."""
    def _on_step(self):
        if self.n_calls % 10000 == 0:
            buf = self.model.replay_buffer
            ep_buf = self.model.ep_info_buffer
            if len(ep_buf) > 0:
                avg_r = np.mean([e['r'] for e in ep_buf[-10:]])
                print(f"  step {self.num_timesteps:>7d}: "
                      f"avg_r(last10)={avg_r:>7.1f}  buffer={buf.size()}")
        return True


def make_env(cfg: ExcavatorEnvConfig, eval_mode: bool = False) -> gym.Env:
    """Factory for VecEnv."""
    env = ExcavatorEnv(cfg)
    if not eval_mode:
        env = Monitor(env)  # logs episode stats
    return env


def train(
    total_steps: int = 200_000,
    warmup_buffer: str | None = None,
    save_dir: str | None = None,
    pipe_freq: float = 0.3,       # 30% episodes have real pipe
):
    """
    Train SAC with conservative hyperparameters.

    Parameters
    ----------
    total_steps : int
        Total environment steps.
    warmup_buffer : str or None
        Path to warmup.npz from warmup.py.
    save_dir : str or None
        Output directory for models, logs, config.
    pipe_freq : float
        Fraction of episodes with real pipe position.
    """
    save_dir = Path(save_dir or _WS / 'rl/models/sac_run')
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Configs for both pipe modes ──
    cfg_no_pipe = ExcavatorEnvConfig(pipe_pos=(0.0, 0.0, -99.0))   # pipe far away
    cfg_with_pipe = ExcavatorEnvConfig()                           # real pipe pos

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    print(f"Total steps: {total_steps:,}")
    print(f"Pipe frequency: {pipe_freq * 100:.0f}%")
    print(f"Save dir: {save_dir}")

    # ── Create vec env ──
    # Use random pipe switching via env wrapper
    venv = DummyVecEnv([
        lambda: PipeSwitchWrapper(cfg_no_pipe, cfg_with_pipe, pipe_freq)
    ])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── Action noise (for exploration) ──
    action_noise = NormalActionNoise(
        mean=np.zeros(4, dtype=np.float32),
        sigma=np.full(4, 0.01, dtype=np.float32),
    )

    # ── Build SAC ──
    model = SAC(
        env=venv,
        verbose=1,
        action_noise=action_noise,
        seed=42,
        **SAC_CONFIG,
    )

    # ── Pre-fill replay buffer with APF baseline data ──
    # 跑 APF 基准 (零动作 + 极低噪声), 灌 50,000 条 transition.
    # SAC 从 buffer 里采样学习 → 初始就在安全行为附近, 不会乱挖.
    print("Collecting APF warmup data (50,000 steps)...")
    obs = venv.reset()
    n_warmup = 0
    while n_warmup < 50_000:
        act = np.random.normal(0, 0.005, size=(1, 4)).astype(np.float32)
        act = np.clip(act, -0.05, 0.05)
        next_obs, rew, done, info = venv.step(act)
        model.replay_buffer.add(
            obs=obs.copy(),
            next_obs=next_obs.copy(),
            action=act.copy(),
            reward=rew.copy(),
            done=np.array([done[0]], dtype=bool),
            infos=[info[0]],
        )
        n_warmup += 1
        obs = next_obs
        if done[0]:
            obs = venv.reset()
        if n_warmup % 10_000 == 0:
            print(f"  warmup: {n_warmup}/50000")
    print(f"  Buffer filled: {model.replay_buffer.size()} transitions ✓")
    obs = venv.reset()

    # ── Eval env (no pipe, for validation) ──
    eval_venv = DummyVecEnv([
        lambda: Monitor(ExcavatorEnv(cfg_no_pipe))
    ])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Share VecNormalize stats from training
    eval_venv.obs_rms = venv.obs_rms

    eval_callback = EvalCallback(
        eval_venv,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    ckpt_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(save_dir),
        name_prefix='sac_ckpt',
    )

    # ── Train ──
    t0 = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=[eval_callback, ckpt_callback, ProgressCallback()],
        progress_bar=False,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete: {elapsed / 60:.1f} min")

    # ── Save final ──
    model_path = save_dir / 'sac_final'
    model.save(str(model_path))
    venv.save(str(save_dir / 'vec_normalize.pkl'))

    # ── Convergence plot ──
    _plot_convergence(save_dir)

    # Save config
    config_json = {
        'total_steps': total_steps,
        'warmup_buffer': warmup_buffer,
        'pipe_freq': pipe_freq,
        'device': device_str,
        'elapsed_min': round(elapsed / 60, 1),
    }
    (save_dir / 'train_config.json').write_text(json.dumps(config_json, indent=2))

    print(f"Model saved → {model_path}.zip")
    return model_path


def _plot_convergence(save_dir: Path):
    """Generate convergence plot from EvalCallback CSV."""
    csv_path = save_dir / 'evaluations.npz'
    if not csv_path.exists():
        print("  No eval log found, skipping plot.")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.load(str(csv_path))
    steps = data['timesteps']
    mean_r = data['results'].mean(axis=1)  # mean over eval episodes

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(steps, mean_r, color='#0F4D92', linewidth=1.0)
    ax.axhline(y=0, color='#B64342', linewidth=0.6, linestyle='--', alpha=0.6)
    ax.set_xlabel('Total timesteps', fontsize=9)
    ax.set_ylabel('Mean episode reward', fontsize=9)
    ax.set_title('SAC Training Convergence — Excavator RL', fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.8)

    plot_path = save_dir / 'convergence.png'
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)
    print(f"Convergence plot saved → {plot_path}")


class PipeSwitchWrapper(gym.Wrapper):
    """
    Randomly switch pipe position each episode.
    70% no pipe (stable digging), 30% real pipe (safe avoidance).
    """
    def __init__(self, cfg_no_pipe, cfg_with_pipe, pipe_prob=0.3):
        super().__init__(ExcavatorEnv(cfg_no_pipe))
        self._cfg_no_pipe = cfg_no_pipe
        self._cfg_with_pipe = cfg_with_pipe
        self._pipe_prob = pipe_prob

    def reset(self, **kwargs):
        # Choose pipe mode randomly
        use_pipe = np.random.random() < self._pipe_prob
        if use_pipe:
            self.env = ExcavatorEnv(self._cfg_with_pipe)
        else:
            self.env = ExcavatorEnv(self._cfg_no_pipe)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='SAC training for excavator RL')
    ap.add_argument('--total-steps', type=int, default=200_000)
    ap.add_argument('--warmup-buffer', type=str, default=None)
    ap.add_argument('--pipe-freq', type=float, default=0.3)
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--export', type=str, default=None,
                    help='Export model to ONNX (give path to .zip)')
    args = ap.parse_args()

    if args.export:
        # Export mode
        _model = SAC.load(args.export, device='cpu')
        _obs_sample = torch.randn(1, 17)
        _path = args.export.replace('.zip', '.onnx')
        torch.onnx.export(
            _model.policy.actor, _obs_sample, _path,
            input_names=['obs'], output_names=['action_mean'],
            opset_version=17,
        )
        print(f"ONNX model → {_path}")
    else:
        train(
            total_steps=args.total_steps,
            warmup_buffer=args.warmup_buffer,
            save_dir=args.save_dir,
            pipe_freq=args.pipe_freq,
        )
