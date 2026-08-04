#!/usr/bin/env python3
"""
图 4: APF 跟踪性能 (部署配置: d_safe=0.5)

三面板:
  (a) d_pipe 时序 — 安全距离随时间变化, d_safe=0.5 阈值
  (b) 跟踪误差 — 齿尖到安全化轨迹(safe_d0.5)的最近点距离
  (c) 齿尖 XZ 轨迹 — 叠加安全化轨迹 + 管线 + d_safe 安全圈

输出: ../plots/fig4_tracking.{png,pdf,svg}
"""

import matplotlib.pyplot as plt
import numpy as np
import os, glob
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
OUT_DIR  = TOOLS_DIR.parent / "plots"
DATA_DIR = TOOLS_DIR.parent / "data"
os.makedirs(OUT_DIR, exist_ok=True)

D_SAFE = 0.5   # 与 trajectory.py 第252行一致

# ── 加载仿真数据 ──
npz_files = sorted(glob.glob(str(DATA_DIR / "cycle_data_*.npz")))
if not npz_files:
    print("WARNING: No cycle_data_*.npz found. Generating synthetic demo.")
    np.random.seed(42)
    N = 800
    t = np.arange(N) * 0.02
    s = np.linspace(0, 0.95, N) + np.random.normal(0, 0.01, N)
    s = np.clip(np.abs(s), 0, 1)
    d_pipe = 2.5 + 0.8 * np.sin(np.linspace(0, 4*np.pi, N)) + np.random.normal(0, 0.03, N)
    d_pipe[300:500] -= 1.0
    p_x = np.linspace(8.4, 4.8, N) + np.random.normal(0, 0.02, N)
    p_z = np.linspace(4.4, -1.7, N) + np.random.normal(0, 0.03, N)
    pipe_pos = np.array([7.5, 0.0, -1.2])
    cycle_count = 1
    use_synthetic = True
else:
    fpath = npz_files[-1]
    print(f"Loading: {fpath}")
    data = np.load(fpath)
    t = data["t"]
    s = data["s"]
    d_pipe = data["d_pipe"]
    p_x = data["p_x"]
    p_z = data["p_z"]
    pipe_pos = data["pipe_pos"]
    cycle_count = int(data["cycle_count"])
    use_synthetic = False

# ── 加载 d_safe=0.5 安全化轨迹 (controller 实际跟踪的目标) ──
try:
    safe_wps = np.load(str(DATA_DIR / "traj_safe_d0.5.npy"))
    orig_wps = np.load(str(DATA_DIR / "traj_original.npy"))
    px_pipe, pz_pipe = float(pipe_pos[0]), float(pipe_pos[2])
    HAS_TRAJ = True
except FileNotFoundError:
    safe_wps = None
    orig_wps = None
    px_pipe, pz_pipe = 7.5, -1.2
    HAS_TRAJ = False

# ── 样式 ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.7, "axes.unicode_minus": False,
})

C_BLUE   = "#0F4D92"
C_RED    = "#B64342"
C_GREY   = "#767676"
C_GREEN  = "#42949E"
C_ORANGE = "#D55E00"

# ═══════════════════════════════════════════════════════════════
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    3, 1, figsize=(5.5, 7.2),
    gridspec_kw={"hspace": 0.52, "top": 0.93, "bottom": 0.07,
                  "left": 0.15, "right": 0.93}
)

# ── Panel (a): d_pipe 时序 ──
ax_a.plot(t, d_pipe, color=C_BLUE, linewidth=0.7, alpha=0.9)

ax_a.axhline(y=D_SAFE, color=C_RED, linewidth=0.6, linestyle="--", alpha=0.6)
ax_a.fill_between([0, t[-1]], 0, D_SAFE, color=C_RED, alpha=0.06, zorder=0)

ax_a.set_xlabel("Time (s)", fontsize=8, labelpad=2)
ax_a.set_ylabel(r"$d_{\rm pipe}$ (m)", fontsize=8, labelpad=2)
ax_a.set_title("(a)  Distance to underground pipe", fontsize=9, loc="left", pad=4)
ax_a.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_a.spines["right"].set_visible(False)
ax_a.spines["top"].set_visible(False)

# ── Panel (b): 跟踪误差 (齿尖到 d_safe=0.5 安全化轨迹) ──
if HAS_TRAJ:
    tracking_err = np.zeros(len(p_x))
    for i in range(len(p_x)):
        pt = np.array([p_x[i], 0.0, p_z[i]])
        min_d = float("inf")
        for j in range(safe_wps.shape[0] - 1):
            a, b = safe_wps[j], safe_wps[j + 1]
            ab = b - a
            tv = np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-10)
            tv = np.clip(tv, 0.0, 1.0)
            closest = a + tv * ab
            dd = float(np.linalg.norm(pt - closest))
            if dd < min_d:
                min_d = dd
        tracking_err[i] = min_d
else:
    tracking_err = np.abs(d_pipe - np.mean(d_pipe))

ax_b.plot(t, tracking_err, color=C_ORANGE, linewidth=0.7, alpha=0.9)

mean_err = np.mean(tracking_err)
ax_b.axhline(y=mean_err, color=C_GREY, linewidth=0.5, linestyle="--", alpha=0.7)

ax_b.set_xlabel("Time (s)", fontsize=8, labelpad=2)
ax_b.set_ylabel("Tracking error (m)", fontsize=8, labelpad=2)
ax_b.set_title("(b)  Tip-to-trajectory tracking error", fontsize=9, loc="left", pad=4)
ax_b.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_b.spines["right"].set_visible(False)
ax_b.spines["top"].set_visible(False)

# ── Panel (c): 齿尖 XZ 轨迹 ──
if HAS_TRAJ:
    ax_c.plot(safe_wps[:, 0], safe_wps[:, 2], color=C_GREEN, linewidth=1.0, alpha=0.7,
              linestyle="--", label="CBF-certified traj. (d=0.5)")
    ax_c.plot(orig_wps[:, 0], orig_wps[:, 2], color=C_GREY, linewidth=0.5, alpha=0.4,
              linestyle=":", label="Demo traj.")

sc = ax_c.scatter(p_x, p_z, c=t, s=3, cmap="Blues", alpha=0.6, linewidths=0,
                  zorder=3, label="Tip actual")

ax_c.scatter([px_pipe], [pz_pipe], s=80, color=C_RED, marker="o", zorder=5,
             edgecolors="white", linewidths=1.2)

theta_c = np.linspace(0, 2*np.pi, 200)
ax_c.plot(px_pipe + D_SAFE * np.cos(theta_c),
          pz_pipe + D_SAFE * np.sin(theta_c),
          color=C_RED, linewidth=0.5, linestyle="--", alpha=0.45)

ax_c.scatter(p_x[0], p_z[0], s=30, color="#333333", marker="s", zorder=4)
ax_c.scatter(p_x[-1], p_z[-1], s=30, color="#333333", marker="D", zorder=4)

cbar = fig.colorbar(sc, ax=ax_c, orientation="vertical", pad=0.02, aspect=30)
cbar.set_label("Time (s)", fontsize=7, labelpad=1)
cbar.ax.tick_params(labelsize=6)

ax_c.set_xlabel("X — forward (m)", fontsize=8, labelpad=2)
ax_c.set_ylabel("Z — height (m)", fontsize=8, labelpad=2)
ax_c.set_title("(c)  Tip trajectory in XZ-plane", fontsize=9, loc="left", pad=4)
ax_c.legend(fontsize=6, framealpha=0.85, edgecolor="#CCCCCC", labelspacing=0.2,
            borderpad=0.3, loc="upper right")
ax_c.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_c.set_aspect("equal")

fig.suptitle(f"APF Tracking Performance — {cycle_count} cycle(s), {len(t)} frames",
             fontsize=11, fontweight="bold", y=0.98)

if use_synthetic:
    for ax in [ax_a, ax_b, ax_c]:
        ax.text(0.5, 0.5, "SYNTHETIC DEMO DATA", transform=ax.transAxes,
                fontsize=16, color="red", alpha=0.15, ha="center", va="center",
                fontweight="bold", rotation=25)

for fmt in ["png", "pdf", "svg"]:
    fpath = OUT_DIR / f"fig4_tracking.{fmt}"
    fig.savefig(str(fpath), dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved: {fpath}")

plt.close(fig)
print("Done — Figure 4: Tracking (d_safe=0.5)")
