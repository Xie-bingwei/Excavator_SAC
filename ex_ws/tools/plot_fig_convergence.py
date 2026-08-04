#!/usr/bin/env python3
"""
图 3: 代价收敛曲线 (部署配置: d_safe=0.5)

三面板:
  (a) J_obs + J_smooth + J_total vs 迭代 (d_safe=0.5 为主线)
  (b) J_obs 收敛对比 — d=0.5/1.0/1.5 三组叠加
  (c) 安全化效果 — 不安全路点数 + min(d_pipe) 对比

输出: ../plots/fig_cost_convergence.{png,pdf,svg}
"""

import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT_DIR, exist_ok=True)

hist_05 = np.load(os.path.join(DATA_DIR, "safety_history_d0.5.npy"))
hist_10 = np.load(os.path.join(DATA_DIR, "safety_history_d1.0.npy"))
hist_15 = np.load(os.path.join(DATA_DIR, "safety_history_d1.5.npy"))
orig    = np.load(os.path.join(DATA_DIR, "traj_original.npy"))
safe_05 = np.load(os.path.join(DATA_DIR, "traj_safe_d0.5.npy"))
safe_10 = np.load(os.path.join(DATA_DIR, "traj_safe_d1.0.npy"))
safe_15 = np.load(os.path.join(DATA_DIR, "traj_safe_d1.5.npy"))
pipe    = np.load(os.path.join(DATA_DIR, "pipe_info.npy"))
px, pz = pipe[0], pipe[2]

D_SELF = 0.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.7, "axes.unicode_minus": False,
})

C_OBS    = "#D55E00"
C_SMOOTH = "#0072B2"
C_TOTAL  = "#333333"
C_05     = "#0F4D92"
C_10     = "#42949E"
C_15     = "#7BB5A8"
C_REF    = "#B64342"
C_ORIG   = "#767676"

# ═══════════════════════════════════════════════════════════════
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    3, 1, figsize=(5.2, 7.5),
    gridspec_kw={"hspace": 0.60, "top": 0.94, "bottom": 0.07, "left": 0.16, "right": 0.93}
)

# ── Panel (a): 三线 — d_safe=0.5 (实际部署配置) ──
iters = np.arange(1, len(hist_05) + 1)
ax_a.plot(iters, hist_05[:, 0], color=C_OBS, linewidth=1.1, label=r"$J_{\rm obs}$")
ax_a.plot(iters, hist_05[:, 1], color=C_SMOOTH, linewidth=0.9, label=r"$J_{\rm smooth}$")
ax_a.plot(iters, hist_05[:, 2], color=C_TOTAL, linewidth=1.5, label=r"$J_{\rm total}$")

ax_a.set_yscale("log")
ax_a.set_xlabel("Iteration", fontsize=8, labelpad=2)
ax_a.set_ylabel("Cost (log scale)", fontsize=8, labelpad=2)
ax_a.set_title(r"(a)  Cost descent — $d_{\rm safe}$=%s m (deployed), $\lambda$=0.3" % str(D_SELF),
               fontsize=9, loc="left", pad=4)
ax_a.legend(fontsize=7, framealpha=0.85, edgecolor="#CCCCCC", labelspacing=0.2, borderpad=0.3, ncol=3)
ax_a.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_a.spines["right"].set_visible(False)
ax_a.spines["top"].set_visible(False)

# ── Panel (b): J_obs 对比 — 三组 d_safe ──
iters_10 = np.arange(1, len(hist_10) + 1)
iters_15 = np.arange(1, len(hist_15) + 1)

ax_b.plot(iters, hist_05[:, 0], color=C_05, linewidth=1.3,
          label=r"$d_{\rm safe}$=%s (deployed)" % str(D_SELF))
ax_b.plot(iters_10, hist_10[:, 0], color=C_10, linewidth=0.9,
          label=r"$d_{\rm safe}$=1.0")
ax_b.plot(iters_15, hist_15[:, 0], color=C_15, linewidth=0.8, alpha=0.7,
          label=r"$d_{\rm safe}$=1.5")

ax_b.set_yscale("log")
ax_b.set_xlabel("Iteration", fontsize=8, labelpad=2)
ax_b.set_ylabel(r"$J_{\rm obs}$ (log scale)", fontsize=8, labelpad=2)
ax_b.set_title(r"(b)  $J_{\rm obs}$ convergence — sensitivity to $d_{\rm safe}$",
               fontsize=9, loc="left", pad=4)
ax_b.legend(fontsize=7, framealpha=0.85, edgecolor="#CCCCCC", labelspacing=0.2, borderpad=0.3)
ax_b.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_b.spines["right"].set_visible(False)
ax_b.spines["top"].set_visible(False)

# ── Panel (c): 安全化效果柱状图 (以 d_safe=0.5 为阈值) ──
configs = ["Original", f"d=0.5\n(deployed)", "d=1.0", "d=1.5"]
all_wps = [orig, safe_05, safe_10, safe_15]

unsafe_counts = []
min_dists = []
for wps in all_wps:
    d = np.sqrt((wps[:, 0] - px)**2 + (wps[:, 2] - pz)**2)
    unsafe_counts.append(int(np.sum(d < D_SELF)))
    min_dists.append(float(np.min(d)))

x_pos = np.arange(len(configs))
bar_colors = [C_ORIG, C_05, C_10, C_15]

bars = ax_c.bar(x_pos, unsafe_counts, width=0.45, color=bar_colors, alpha=0.75,
                edgecolor="white", linewidth=0.5, zorder=2)
for bar, cnt, col in zip(bars, unsafe_counts, bar_colors):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
              str(cnt), ha="center", fontsize=9, fontweight="bold", color=col)

ax_c2 = ax_c.twinx()
ax_c2.plot(x_pos, min_dists, color=C_REF, linewidth=1.3, marker="D", ms=6,
           mfc="white", mec=C_REF, mew=1.2, zorder=4, label=r"$\min(d_{\rm pipe})$")
ax_c2.axhline(y=D_SELF, color=C_REF, linewidth=0.5, linestyle="--", alpha=0.5)
ax_c2.text(0.2, D_SELF * 1.05, r"$d_{\rm safe}$=%s" % str(D_SELF), fontsize=7, color=C_REF, alpha=0.7)
ax_c2.set_ylabel(r"$\min(d_{\rm pipe})$ (m)", fontsize=8, color=C_REF, labelpad=4)
ax_c2.tick_params(axis="y", labelsize=7.5, colors=C_REF)
ax_c2.set_ylim(0, max(min_dists) * 1.3)

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(configs, fontsize=7)
ax_c.set_ylabel("Unsafe waypoints (count)", fontsize=8, labelpad=2)
ax_c.set_title("(c)  Safety outcome — before vs after  (threshold=%s m)" % str(D_SELF),
               fontsize=9, loc="left", pad=4)
ax_c.set_ylim(0, max(unsafe_counts) * 1.25)
ax_c.grid(True, axis="y", linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7, zorder=0)
ax_c.spines["top"].set_visible(False)

from matplotlib.lines import Line2D
ax_c.legend([Line2D([0], [0], color=C_REF, marker="D", ms=6, mfc="white", mec=C_REF, lw=1.3)],
            [r"$\min(d_{\rm pipe})$"], fontsize=7, loc="upper right",
            framealpha=0.85, edgecolor="#CCCCCC")

fig.suptitle("CBF Trajectory Safety — Cost Convergence & Safety Outcome  ($d_{\\rm safe}$=%s m)" % str(D_SELF),
             fontsize=11, fontweight="bold", y=0.99)

for fmt in ["png", "pdf", "svg"]:
    fpath = os.path.join(OUT_DIR, f"fig3_cost_convergence.{fmt}")
    fig.savefig(fpath, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved: {fpath}")

plt.close(fig)
print("Done — Figure 3: Cost Convergence (d_safe=0.5)")
