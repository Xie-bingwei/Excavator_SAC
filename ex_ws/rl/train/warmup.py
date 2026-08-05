#!/usr/bin/env python3
"""
灌 replay buffer: 跑 APF + 小幅噪声, 收集 safe transitions 存入 .npz.
训练时加载灌入 SAC replay buffer → 初始就有安全行为先验.
"""
import sys
from pathlib import Path
_WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WS / 'rl'))
sys.path.insert(0, str(_WS / 'rl/env'))

import numpy as np
from excavator_env import ExcavatorEnv, ExcavatorEnvConfig


def collect_warmup_transitions(
    total_transitions: int = 100_000,
    noise_std: float = 0.015,
    save_dir: str | None = None,
):
    """
    Collect transitions with APF + exploratory noise.

    Returns paths to saved .npz files.
    """
    cfg = ExcavatorEnvConfig()
    save_dir = Path(save_dir or _WS / 'rl/models')
    save_dir.mkdir(parents=True, exist_ok=True)

    env = ExcavatorEnv(cfg)

    obs_buf = np.empty((total_transitions, 17), dtype=np.float32)
    act_buf = np.empty((total_transitions, 4), dtype=np.float32)
    nxt_buf = np.empty((total_transitions, 17), dtype=np.float32)
    rew_buf = np.empty(total_transitions, dtype=np.float32)
    don_buf = np.empty(total_transitions, dtype=bool)

    obs, _ = env.reset()
    collected = 0
    episodes = 0
    ep_rewards = []

    print(f"Collecting {total_transitions} transitions (σ={noise_std})...")

    while collected < total_transitions:
        # APF baseline is zero action. Add noise to explore nearby.
        action = np.random.normal(0, noise_std, size=4).astype(np.float32)
        action = np.clip(action, -cfg.action_max, cfg.action_max)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        obs_buf[collected] = obs
        act_buf[collected] = action
        nxt_buf[collected] = next_obs
        rew_buf[collected] = reward
        don_buf[collected] = done

        collected += 1
        obs = next_obs

        if done:
            episodes += 1
            ep_rewards.append(env._cum_reward)
            obs, _ = env.reset()

        if collected % 20_000 == 0 and episodes > 0:
            avg_r = np.mean(ep_rewards[-10:]) if ep_rewards else 0
            print(f"  {collected}/{total_transitions} "
                  f"(ep {episodes}, avg_last10_r={avg_r:.1f})")

    # ── Save ──
    data = dict(
        obs=obs_buf, actions=act_buf, next_obs=nxt_buf,
        rewards=rew_buf, dones=don_buf,
    )
    path = save_dir / 'warmup.npz'
    np.savez_compressed(str(path), **data)

    print(f"Saved {collected} transitions → {path}")
    print(f"  {episodes} episodes, "
          f"avg_r/ep={np.mean(ep_rewards):.1f}, "
          f"best_r/ep={np.max(ep_rewards):.1f}, "
          f"worst_r/ep={np.min(ep_rewards):.1f}")
    return str(path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--transitions', type=int, default=100_000)
    ap.add_argument('--noise', type=float, default=0.015)
    ap.add_argument('-o', '--output', type=str, default=None)
    args = ap.parse_args()
    collect_warmup_transitions(args.transitions, args.noise, args.output)
