#!/usr/bin/env python3
"""
SAC 推理包装器 — 纯 onnxruntime + numpy, 零 SB3/PyTorch 依赖.
自动加载 obs_norm.npz (mean/std) 做观测归一化.
"""
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


class SACWrapper:
    """ONNX inference + obs normalization (from pre-exported .npz)."""

    action_clip = 0.05

    def __init__(self, model_path: str):
        # ── Load ONNX session ──
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        self._session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # ── Load normalization stats (.npz) ──
        _npz = Path(model_path).parent / 'obs_norm.npz'
        self._obs_mean = np.zeros(17, dtype=np.float32)
        self._obs_std = np.ones(17, dtype=np.float32)

        if _npz.exists():
            _d = np.load(str(_npz))
            self._obs_mean = _d['mean'].astype(np.float32)
            self._obs_std = _d['std'].astype(np.float32).clip(1e-6)
        else:
            print(f"[SACWrapper] WARNING: {_npz} not found, using raw obs")

        self._action_prev = np.zeros(4, dtype=np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        obs shape (17,) — raw (not normalized).
        Returns delta_q ∈ [-0.05, +0.05].
        """
        # Normalize
        norm = (obs.astype(np.float32) - self._obs_mean) / self._obs_std
        out = self._session.run(
            [self._output_name],
            {self._input_name: norm.reshape(1, -1)},
        )[0]
        delta = np.clip(out.flatten(), -self.action_clip, self.action_clip)
        self._action_prev = delta.copy()
        return delta

    def build_obs(
        self,
        q: np.ndarray,
        p_tip: np.ndarray,
        s_star: float,
        d_pipe: float,
        alpha_margin: float,
        base_angvel: np.ndarray,
    ) -> np.ndarray:
        """Build 17-dim raw observation vector (will be normalized in predict)."""
        return np.array([
            *q.astype(np.float32),
            *p_tip.astype(np.float32),
            np.float32(s_star),
            np.float32(d_pipe),
            np.float32(alpha_margin),
            *base_angvel.astype(np.float32),
            *self._action_prev.astype(np.float32),
        ], dtype=np.float32)


# Smoke test
if __name__ == '__main__':
    import sys, os
    _ws = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ws / 'src/excavator_kinematics'))
    sys.path.insert(0, str(_ws / 'src/excavator_trajectory'))
    os.environ['TRAJECTORY_GENERATING'] = '1'

    from excavator_kinematics.mdh import forward_kinematics
    from excavator_kinematics.zmp import zmp_alpha_margin

    sac = SACWrapper(str(_ws / 'rl/models/sac_run/sac_final.onnx'))
    q = np.array([0.0, 0.055, -0.609, 0.443])
    p_tip, _ = forward_kinematics(q)
    alpha, _ = zmp_alpha_margin(q)
    obs = sac.build_obs(q, p_tip, 0.0, 5.8, alpha, np.zeros(3))
    dq = sac.predict(obs)
    print(f'delta_q = {np.array2string(dq, precision=4)}')
    print(f'mean range = [{self._obs_mean.min():.2f}, {self._obs_mean.max():.2f}]')
    print(f'std  range = [{self._obs_std.min():.4f}, {self._obs_std.max():.4f}]')
    print('Wrapper OK')
