#!/usr/bin/env python3
"""
论文图生成脚本 —— 运行后自动在 VNC 桌面弹出图窗口。

用法:
    DISPLAY=:1 python3 scripts/plot_figures.py          # 弹窗交互
    DISPLAY=:1 python3 scripts/plot_figures.py --save   # 弹窗 + 保存到 document/AGX/
"""
import os, sys, argparse

# ── 切到交互式后端 (VNC 桌面弹窗) ──
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# 项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'excavator_trajectory'))
from excavator_trajectory.trajectory import WAYPOINTS, _SAFETY_HISTORY, certify_trajectory

# ═══════════ Nature Skills PALETTE ═══════════
PALETTE = {
    "blue_main":      "#0F4D92",
    "red_strong":     "#B64342",
    "teal":           "#42949E",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
}

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

# ── Build original waypoints from recorded JSON ──
import json as _json
from pathlib import Path as _Path

_q_path = _Path(__file__).parent.parent / 'src' / 'excavator_trajectory' / 'excavator_trajectory' / 'recorded_trajectory.json'
with open(_q_path) as _f:
    _data = _json.load(_f)
all_pts = [np.array([-frm['p_bf'][0], 0.0, frm['p_bf'][2]]) for frm in _data]
orig_wps = [all_pts[0]]
for p in all_pts[1:]:
    if np.linalg.norm(p - orig_wps[-1]) >= 0.4:
        orig_wps.append(p.copy())
orig_wps = np.array(orig_wps)

# ── Current certified data ──
cert_wps = WAYPOINTS
history  = np.array(_SAFETY_HISTORY)
pipe_pos = np.array([7.5, 0.0, -1.2])

# ═══════════════════════════════════════════
# Figure 1: Cost Functional Convergence
# ═══════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(3.0, 3.0), num='Fig 1 — Cost Convergence')
iters = np.arange(len(history))

p_total,  = ax1.plot(iters, history[:, 2], color=PALETTE["blue_main"],  linewidth=1.2)
p_obs,    = ax1.plot(iters, history[:, 0], color=PALETTE["red_strong"], linewidth=0.7, linestyle='--')
p_smooth, = ax1.plot(iters, history[:, 1], color=PALETTE["teal"],       linewidth=0.7, linestyle=':')

ax1.set_xlabel('Iteration', fontsize=8, labelpad=3)
ax1.set_ylabel('Cost', fontsize=8, labelpad=3)
ax1.set_yscale('log')
ax1.set_xlim(0, len(history) - 1)
ax1.tick_params(labelsize=7, pad=2)

ax1.legend([p_total, p_obs, p_smooth],
           [r'$J_{\mathrm{total}}$', r'$J_{\mathrm{obs}}$', r'$J_{\mathrm{smooth}}$'],
           ncol=3, loc='upper right', fontsize=6.5,
           handlelength=1.0, handletextpad=0.4)

# Annotations
ax1.annotate(f'{history[0,2]:.3f}', xy=(0, history[0,2]),
             xytext=(15, history[0,2] * 2.5), fontsize=6.5, color=PALETTE["blue_main"],
             arrowprops=dict(arrowstyle='->', color=PALETTE["neutral_mid"], lw=0.5))
ax1.annotate(f'{history[-1,2]:.1e}', xy=(len(history) - 1, history[-1,2]),
             fontsize=6.5, color=PALETTE["blue_main"], ha='right', va='bottom')

fig1.tight_layout(pad=0.5)

# ═══════════════════════════════════════════
# Figure 2: Original vs Certified Trajectory
# ═══════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(3.8, 3.8), num='Fig 2 — Trajectory Certification')

# Safe zone
zone = plt.Circle((pipe_pos[0], pipe_pos[2]), 0.5,
                   facecolor=PALETTE["red_strong"], alpha=0.10,
                   edgecolor=PALETTE["red_strong"], linewidth=0.6,
                   linestyle='--', zorder=1)
ax2.add_patch(zone)
ax2.plot(pipe_pos[0], pipe_pos[2], 'o', color=PALETTE["red_strong"], markersize=4.5, zorder=4)
ax2.text(pipe_pos[0] + 0.35, pipe_pos[2] + 0.25, 'Pipe', fontsize=6.5,
         color=PALETTE["red_strong"], va='bottom', ha='left')

# Trajectories
p1, = ax2.plot(orig_wps[:, 0], orig_wps[:, 2], 'o-',
               color=PALETTE["neutral_mid"], markersize=2.2,
               linewidth=0.8, markerfacecolor='white', markeredgewidth=0.4,
               zorder=2)
p2, = ax2.plot(cert_wps[:, 0], cert_wps[:, 2], 's-',
               color=PALETTE["blue_main"], markersize=1.8,
               linewidth=1.1, alpha=0.85, zorder=3)

# Deformation arrow
od = orig_wps[np.argmin(orig_wps[:, 2])]
cd = cert_wps[np.argmin(cert_wps[:, 2])]
ax2.annotate('', xy=(cd[0], cd[2]), xytext=(od[0], od[2]),
             arrowprops=dict(arrowstyle='->', color=PALETTE["neutral_dark"],
                             lw=0.5, linestyle='--'))

# Phase labels
ax2.text(8.58, 3.55, 'Descent', fontsize=6, color=PALETTE["neutral_mid"], ha='center')
ax2.text(7.95, -0.05, 'Dig', fontsize=6, color=PALETTE["neutral_mid"], ha='center')
ax2.text(5.7, 1.75, 'Lift & return', fontsize=6, color=PALETTE["neutral_mid"], ha='center')

ax2.legend([p1, p2], ['Original', 'Certified'], loc='lower right',
           fontsize=6.5, handlelength=1.0, handletextpad=0.3)

ax2.set_xlabel('X — forward distance (m)', fontsize=8, labelpad=3)
ax2.set_ylabel('Z — height (m)', fontsize=8, labelpad=3)
ax2.set_aspect('equal')
ax2.tick_params(labelsize=7, pad=2)

fig2.tight_layout(pad=0.5)

# ── Save ──
save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'document', 'AGX')
save_dir = os.path.abspath(save_dir)
fig1.savefig(os.path.join(save_dir, 'fig_cost_convergence.pdf'), bbox_inches='tight')
fig1.savefig(os.path.join(save_dir, 'fig_cost_convergence.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(save_dir, 'fig_trajectory_cert.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(save_dir, 'fig_trajectory_cert.png'), dpi=300, bbox_inches='tight')
print(f"Saved to {save_dir}/")

# ── Show ──
plt.show()
