#!/usr/bin/env python3
"""
图 2: CBF 轨迹安全化对比 (部署配置: d_safe=0.5, λ=0.3)

四面板:
  (a) 完整轨迹 XZ — 原始 vs 安全化 (主推 d_safe=0.5)
  (b) 管线区域放大 — 原始 vs d=0.5/1.0/1.5 三组
  (c) d_pipe 剖面 — 沿路点索引的距离变化
  (d) 直方图 — 安全化前后对比 (d_safe=0.5)

输出: ../plots/fig_traj_safety.{png,pdf,svg}
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT_DIR, exist_ok=True)

orig    = np.load(os.path.join(DATA_DIR, "traj_original.npy"))
safe_05 = np.load(os.path.join(DATA_DIR, "traj_safe_d0.5.npy"))
safe_10 = np.load(os.path.join(DATA_DIR, "traj_safe_d1.0.npy"))
safe_15 = np.load(os.path.join(DATA_DIR, "traj_safe_d1.5.npy"))
pipe    = np.load(os.path.join(DATA_DIR, "pipe_info.npy"))
px, py, pz = pipe[0], pipe[1], pipe[2]

def dist_to_pipe(wps):
    return np.sqrt((wps[:, 0] - px)**2 + (wps[:, 2] - pz)**2)

orig_d = dist_to_pipe(orig)
safe_d_05 = dist_to_pipe(safe_05)
safe_d_10 = dist_to_pipe(safe_10)
safe_d_15 = dist_to_pipe(safe_15)

D_SELF = 0.5          # 当前部署值
D_PAPER = 1.5         # 论文推荐值, 缩放时对比展示

# ── 样式 ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.7, "axes.unicode_minus": False,
})

C_PIPE    = "#B64342"
C_ORIG    = "#767676"
C_MAIN    = "#0F4D92"   # 主推 d=0.5
C_SAFE_10 = "#42949E"
C_SAFE_15 = "#7BB5A8"
C_ZONE    = "#F5E6E0"

# ═══════════════════════════════════════════════════════════════
fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
    2, 2, figsize=(7.2, 6.0),
    gridspec_kw={"hspace": 0.48, "wspace": 0.42, "top": 0.93, "bottom": 0.10,
                  "left": 0.10, "right": 0.95}
)

# ── Panel (a): 完整 XZ (主推 d_safe=0.5) ──
circle_r = D_SELF
theta = np.linspace(0, 2*np.pi, 200)
ax_a.fill(px + circle_r * np.cos(theta), pz + circle_r * np.sin(theta),
          color=C_ZONE, alpha=0.5, zorder=0)
ax_a.plot(px + circle_r * np.cos(theta), pz + circle_r * np.sin(theta),
          color=C_PIPE, linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

ax_a.plot(orig[:, 0], orig[:, 2], color=C_ORIG, linewidth=0.6, alpha=0.8, zorder=2)
ax_a.scatter(orig[:, 0], orig[:, 2], s=6, color=C_ORIG, alpha=0.5, zorder=2, linewidths=0)

ax_a.plot(safe_05[:, 0], safe_05[:, 2], color=C_MAIN, linewidth=1.0, zorder=3)
ax_a.scatter(safe_05[:, 0], safe_05[:, 2], s=10, color=C_MAIN, alpha=0.8, zorder=3,
             marker="^", linewidths=0)

ax_a.scatter([px], [pz], s=60, color=C_PIPE, marker="o", zorder=5, edgecolors="white", linewidths=0.8)
ax_a.text(px + 0.25, pz + 0.15, "Pipe", fontsize=7, color=C_PIPE, fontweight="bold", ha="left", va="center")

ax_a.scatter(orig[0, 0], orig[0, 2], s=30, color="#333333", marker="s", zorder=4)
ax_a.text(orig[0, 0] + 0.1, orig[0, 2] - 0.35, "Start", fontsize=6.5, color="#333333")
ax_a.scatter(orig[-1, 0], orig[-1, 2], s=30, color="#333333", marker="D", zorder=4)
ax_a.text(orig[-1, 0] - 0.05, orig[-1, 2] + 0.35, "End", fontsize=6.5, color="#333333", ha="center")

n_orig_unsafe = int(np.sum(orig_d < D_SELF))
n_safe_unsafe = int(np.sum(safe_d_05 < D_SELF))
legend_a = [
    mpatches.Patch(color=C_ORIG, alpha=0.7,
                   label=f"Demo traj. (%d unsafe)" % n_orig_unsafe),
    mpatches.Patch(color=C_MAIN, alpha=0.8,
                   label=f"CBF-cert. $d_{{\\rm safe}}$=%s (%d unsafe)" % (str(D_SELF), n_safe_unsafe)),
    mpatches.Patch(color=C_ZONE, alpha=0.5, label="Safety zone"),
]
ax_a.legend(handles=legend_a, loc="upper right", fontsize=6.5, framealpha=0.85,
            edgecolor="#CCCCCC", handlelength=1.5, borderpad=0.4, labelspacing=0.3)

ax_a.set_xlabel("X — forward (m)", fontsize=8, labelpad=2)
ax_a.set_ylabel("Z — height (m)", fontsize=8, labelpad=2)
ax_a.set_title("(a)  XZ-plane overview  ($d_{\\rm safe}$=%s m)" % str(D_SELF), fontsize=9, loc="left", pad=4)
ax_a.set_aspect("equal")
ax_a.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)

# ── Panel (b): 管线区域放大 ──
ax_b.fill(px + D_PAPER * np.cos(theta), pz + D_PAPER * np.sin(theta),
          color=C_ZONE, alpha=0.3, zorder=0)
ax_b.plot(px + D_PAPER * np.cos(theta), pz + D_PAPER * np.sin(theta),
          color=C_PIPE, linewidth=0.5, linestyle="-.", alpha=0.35, zorder=1)
ax_b.fill(px + D_SELF * np.cos(theta), pz + D_SELF * np.sin(theta),
          color=C_ZONE, alpha=0.5, zorder=0)
ax_b.plot(px + D_SELF * np.cos(theta), pz + D_SELF * np.sin(theta),
          color=C_PIPE, linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

ax_b.plot(orig[:, 0], orig[:, 2], color=C_ORIG, linewidth=0.5, alpha=0.6, zorder=2)
ax_b.plot(safe_05[:, 0], safe_05[:, 2], color=C_MAIN, linewidth=1.2, zorder=3, label="d=0.5 (deployed)")
ax_b.plot(safe_10[:, 0], safe_10[:, 2], color=C_SAFE_10, linewidth=0.8, zorder=3, alpha=0.6, label="d=1.0")
ax_b.plot(safe_15[:, 0], safe_15[:, 2], color=C_SAFE_15, linewidth=0.6, zorder=3, alpha=0.4, label="d=1.5")

ax_b.scatter([px], [pz], s=80, color=C_PIPE, marker="o", zorder=5, edgecolors="white", linewidths=1.0)

unsafe_mask = orig_d < D_SELF
ax_b.scatter(orig[unsafe_mask, 0], orig[unsafe_mask, 2], s=22, color=C_PIPE, marker="x", zorder=4, linewidths=1.3, alpha=0.9)

ax_b.set_xlim(5.8, 8.4)
ax_b.set_ylim(-2.6, 0.4)
ax_b.set_xlabel("X — forward (m)", fontsize=8, labelpad=2)
ax_b.set_ylabel("Z — height (m)", fontsize=8, labelpad=2)
ax_b.set_title("(b)  Zoom — pipe vicinity", fontsize=9, loc="left", pad=4)
ax_b.legend(fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC", labelspacing=0.2, borderpad=0.3)
ax_b.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)

# ── Panel (c): 距离剖面 ──
idx = np.arange(len(orig))
ax_c.plot(idx, orig_d, color=C_ORIG, linewidth=0.7, marker="o", ms=2.5, alpha=0.7, label="Demo")
ax_c.plot(idx, safe_d_05, color=C_MAIN, linewidth=1.2, alpha=0.9, label="CBF d=0.5")
ax_c.plot(idx, safe_d_10, color=C_SAFE_10, linewidth=0.7, alpha=0.5, label="d=1.0")
ax_c.plot(idx, safe_d_15, color=C_SAFE_15, linewidth=0.6, alpha=0.35, label="d=1.5")

ax_c.axhline(y=D_SELF, color=C_PIPE, linewidth=0.8, linestyle="--", alpha=0.7)
ax_c.text(len(idx) - 1.5, D_SELF * 1.06, r"$d_{\rm safe}$=%s" % str(D_SELF),
          fontsize=7, color=C_PIPE, ha="right", va="bottom", fontweight="bold")

ax_c.fill_between([0, len(idx)-1], 0, D_SELF, color=C_ZONE, alpha=0.3, zorder=0)

ax_c.set_xlabel("Waypoint index", fontsize=8, labelpad=2)
ax_c.set_ylabel(r"$d_{\rm pipe}$ (m)", fontsize=8, labelpad=2)
ax_c.set_title("(c)  Distance-to-pipe profile", fontsize=9, loc="left", pad=4)
ax_c.set_xlim(0, len(idx) - 1)
ax_c.legend(fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC", labelspacing=0.2, borderpad=0.3, ncol=2)
ax_c.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)

# ── Panel (d): 直方图 (d_safe=0.5 前后) ──
bins = np.linspace(0, 6.0, 25)
ax_d.hist(orig_d, bins=bins, color=C_ORIG, alpha=0.55, edgecolor="white", lw=0.3,
          label="Demo (min=%.2fm, %d unsafe)" % (orig_d.min(), n_orig_unsafe))
ax_d.hist(safe_d_05, bins=bins, color=C_MAIN, alpha=0.55, edgecolor="white", lw=0.3,
          label="CBF-cert. (min=%.2fm, %d unsafe)" % (safe_d_05.min(), n_safe_unsafe))

ax_d.axvline(x=D_SELF, color=C_PIPE, linewidth=0.8, linestyle="--", alpha=0.7)
ax_d.text(D_SELF + 0.08, ax_d.get_ylim()[1] * 0.85, r"$d_{\rm safe}$=%s" % str(D_SELF),
          fontsize=7, color=C_PIPE, ha="left", va="top", fontweight="bold")

ax_d.set_xlabel(r"$d_{\rm pipe}$ (m)", fontsize=8, labelpad=2)
ax_d.set_ylabel("Waypoints", fontsize=8, labelpad=2)
ax_d.set_title("(d)  Distance distribution — before vs after", fontsize=9, loc="left", pad=4)
ax_d.legend(fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC")

# ── 全局标题 ──
fig.suptitle("CBF Trajectory Safety Certification — Pipeline Avoidance  ($d_{\\rm safe}$=%s m, $\\lambda$=0.3)" % str(D_SELF),
             fontsize=11, fontweight="bold", y=0.98)

for fmt in ["png", "pdf", "svg"]:
    fpath = os.path.join(OUT_DIR, f"fig2_traj_safety.{fmt}")
    fig.savefig(fpath, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved: {fpath}")

plt.close(fig)
print("Done — Figure 2: Trajectory Safety (d_safe=0.5)")
