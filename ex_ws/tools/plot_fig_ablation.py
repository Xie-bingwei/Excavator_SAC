#!/usr/bin/env python3
"""
图 5: 消融实验 — d_safe 和 λ 灵敏度 (部署: d_safe=0.5, λ=0.3)

两面板:
  (a) d_safe 灵敏度 — 不安全路点数 + min(d_pipe) vs d_safe
  (b) λ 灵敏度 — 粗糙度 + min(d_pipe) vs λ (固定 d_safe=0.5)

输出: ../plots/fig_ablation.{png,pdf,svg}
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import math
import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
OUT_DIR = TOOLS_DIR.parent / "plots"
os.makedirs(OUT_DIR, exist_ok=True)

WAYPOINTS_ORIGINAL = np.array([
    [ 8.389, 0.0,  4.375], [ 8.513, 0.0,  3.965], [ 8.615, 0.0,  3.549],
    [ 8.695, 0.0,  3.129], [ 8.753, 0.0,  2.705], [ 8.789, 0.0,  2.278],
    [ 8.802, 0.0,  1.850], [ 8.793, 0.0,  1.422], [ 8.761, 0.0,  0.995],
    [ 8.707, 0.0,  0.570], [ 8.631, 0.0,  0.148], [ 8.532, 0.0, -0.269],
    [ 8.557, 0.0, -0.173], [ 8.515, 0.0, -0.333], [ 8.470, 0.0, -0.490],
    [ 8.423, 0.0, -0.647], [ 8.372, 0.0, -0.802], [ 8.318, 0.0, -0.955],
    [ 8.262, 0.0, -1.106], [ 8.203, 0.0, -1.256], [ 8.165, 0.0, -1.347],
    [ 8.166, 0.0, -1.345], [ 8.166, 0.0, -1.345], [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345], [ 8.166, 0.0, -1.345], [ 8.166, 0.0, -1.345],
    [ 8.166, 0.0, -1.345], [ 8.166, 0.0, -1.345], [ 8.166, 0.0, -1.345],
    [ 8.127, 0.0, -1.380], [ 8.059, 0.0, -1.435], [ 7.988, 0.0, -1.485],
    [ 7.914, 0.0, -1.531], [ 7.837, 0.0, -1.573], [ 7.758, 0.0, -1.610],
    [ 7.676, 0.0, -1.642], [ 7.593, 0.0, -1.670], [ 7.509, 0.0, -1.692],
    [ 7.423, 0.0, -1.709], [ 7.336, 0.0, -1.721], [ 7.249, 0.0, -1.728],
    [ 7.162, 0.0, -1.729], [ 7.075, 0.0, -1.725], [ 6.988, 0.0, -1.717],
    [ 6.901, 0.0, -1.702], [ 6.816, 0.0, -1.683], [ 6.732, 0.0, -1.659],
    [ 6.650, 0.0, -1.629], [ 6.569, 0.0, -1.595], [ 6.491, 0.0, -1.556],
    [ 6.415, 0.0, -1.513], [ 6.342, 0.0, -1.465], [ 6.380, 0.0, -0.989],
    [ 6.288, 0.0, -0.596], [ 6.168, 0.0, -0.211], [ 6.021, 0.0,  0.166],
    [ 5.847, 0.0,  0.533], [ 5.647, 0.0,  0.889], [ 5.395, 0.0,  1.232],
    [ 5.141, 0.0,  1.542], [ 4.919, 0.0,  1.883], [ 4.799, 0.0,  2.266],
    [ 4.790, 0.0,  2.328],
])

PIPE = np.array([7.5, 0.0, -1.2])
LAM_DEFAULT = 0.3
D_DEPLOYED = 0.5
MAX_ITER = 200


def certify_trajectory(waypoints, pipe_pos, d_safe, lam, max_iter=MAX_ITER):
    safe = waypoints.copy()
    n = safe.shape[0]
    history = []
    for _iter in range(max_iter):
        moved = False
        for i in range(n):
            dx = safe[i, 0] - pipe_pos[0]
            dz = safe[i, 2] - pipe_pos[2]
            d = math.sqrt(dx * dx + dz * dz)
            if d < d_safe and d > 1e-6:
                grad = -2.0 * (d_safe - d) * np.array([dx / d, 0.0, abs(dz) / d])
                alpha = 0.25 / d_safe
                safe[i] = safe[i] - alpha * grad
                if safe[i, 2] < waypoints[i, 2]:
                    safe[i, 2] = waypoints[i, 2] + 0.5 * (safe[i, 2] - waypoints[i, 2])
                moved = True
        for i in range(1, n - 1):
            avg = (safe[i - 1] + safe[i + 1]) * 0.5
            grad_s = 2.0 * (safe[i] - avg)
            safe[i] = safe[i] - (0.25 / d_safe) * lam * grad_s
        J_obs = 0.0
        J_smooth = 0.0
        for i in range(n):
            dx = safe[i, 0] - pipe_pos[0]
            dz = safe[i, 2] - pipe_pos[2]
            d = math.sqrt(dx * dx + dz * dz)
            if d < d_safe:
                J_obs += (d_safe - d) ** 2
            if 1 <= i <= n - 2:
                dev = safe[i] - (safe[i - 1] + safe[i + 1]) * 0.5
                J_smooth += float(np.dot(dev, dev))
        J_total = J_obs + lam * J_smooth
        history.append((J_obs, J_smooth, J_total))
        if not moved or J_obs < 1e-4:
            break
    return safe, history


def evaluate(wps, d_safe):
    d = np.sqrt((wps[:, 0] - PIPE[0])**2 + (wps[:, 2] - PIPE[2])**2)
    n_unsafe = int(np.sum(d < d_safe))
    min_d = float(np.min(d))
    seg_len = np.sqrt(np.sum(np.diff(wps, axis=0)**2, axis=1))
    roughness = float(np.std(seg_len))
    return n_unsafe, min_d, roughness


# ═══════════════════════════════════════════════════════════════
# 扫描
# ═══════════════════════════════════════════════════════════════
print("Running ablation sweep…")

# (a) d_safe 扫描
d_safe_vals = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8]
d_results = []
for ds in d_safe_vals:
    safe, hist = certify_trajectory(WAYPOINTS_ORIGINAL, PIPE, ds, LAM_DEFAULT)
    n_un, min_d, rough = evaluate(safe, ds)
    d_results.append({"ds": ds, "n_unsafe": n_un, "min_d": min_d,
                       "roughness": rough, "n_iter": len(hist),
                       "J_final": hist[-1][2] if hist else 0})
    print(f"  d_safe={ds}: unsafe={n_un}, min_d={min_d:.3f}, rough={rough:.3f}, iters={len(hist)}")

# (b) λ 扫描 (固定 d_safe=0.5)
lam_vals = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
lam_results = []
for lv in lam_vals:
    safe, hist = certify_trajectory(WAYPOINTS_ORIGINAL, PIPE, D_DEPLOYED, lv)
    n_un, min_d, rough = evaluate(safe, D_DEPLOYED)
    lam_results.append({"lam": lv, "n_unsafe": n_un, "min_d": min_d,
                         "roughness": rough, "n_iter": len(hist),
                         "J_smooth_final": hist[-1][1] if hist else 0})
    print(f"  λ={lv}: unsafe={n_un}, min_d={min_d:.3f}, rough={rough:.3f}, J_smooth={lam_results[-1]['J_smooth_final']:.4f}")

# ═══════════════════════════════════════════════════════════════
# 样式
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.7, "axes.unicode_minus": False,
})

C_BLUE  = "#0F4D92"
C_RED   = "#B64342"
C_GREEN = "#42949E"
C_GREY  = "#767676"

orig_d = np.sqrt((WAYPOINTS_ORIGINAL[:, 0] - PIPE[0])**2 + (WAYPOINTS_ORIGINAL[:, 2] - PIPE[2])**2)
orig_unsafe = int(np.sum(orig_d < D_DEPLOYED))
orig_min_d = float(np.min(orig_d))

# ═══════════════════════════════════════════════════════════════
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(7.5, 3.5),
    gridspec_kw={"top": 0.82, "bottom": 0.20, "left": 0.10, "right": 0.95, "wspace": 0.45}
)

# ── Panel (a): d_safe 灵敏度 ──
ds_arr   = np.array([r["ds"] for r in d_results])
un_arr   = np.array([r["n_unsafe"] for r in d_results])
mind_arr = np.array([r["min_d"] for r in d_results])

ax_a1 = ax_a
ax_a2 = ax_a1.twinx()

# 高亮当前部署的 d_safe=0.5
colors_a = [C_RED if ds == D_DEPLOYED else "#E8A0A0" for ds in ds_arr]
bars = ax_a1.bar(ds_arr - 0.05, un_arr, width=0.10, color=colors_a, alpha=0.75,
                 edgecolor="white", linewidth=0.4)
for b, v, ds in zip(bars, un_arr, ds_arr):
    fw = "bold" if ds == D_DEPLOYED else "normal"
    ax_a1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, str(v),
               ha="center", fontsize=7, color=C_RED if ds == D_DEPLOYED else C_GREY,
               fontweight=fw)

ax_a2.plot(ds_arr, mind_arr, color=C_BLUE, linewidth=1.3, marker="D", ms=5.5,
           mfc="white", mec=C_BLUE, mew=1.0, label=r"$\min(d_{\rm pipe})$")
ax_a2.plot(ds_arr, ds_arr, color=C_GREY, linewidth=0.5, ls="--", alpha=0.5, label=r"$y=x$ ideal")
ax_a2.scatter([D_DEPLOYED], [mind_arr[ds_arr == D_DEPLOYED][0]], s=40, color=C_BLUE,
              zorder=5)

ax_a1.set_xlabel(r"$d_{\rm safe}$ (m)", fontsize=8, labelpad=2)
ax_a1.set_ylabel("Unsafe waypoints (count)", fontsize=8, color=C_RED, labelpad=2)
ax_a2.set_ylabel(r"$\min(d_{\rm pipe})$ (m)", fontsize=8, color=C_BLUE, labelpad=4)
ax_a1.tick_params(axis="y", colors=C_RED)
ax_a2.tick_params(axis="y", colors=C_BLUE)
ax_a1.set_title("(a)  Sensitivity to $d_{\\rm safe}$  ($\\lambda$=0.3)", fontsize=9, loc="left", pad=4)
ax_a1.legend(handles=[
    mpatches.Patch(color=C_RED, alpha=0.75, label="Unsafe wp count"),
    Line2D([0], [0], color=C_BLUE, marker="D", ms=5.5, mfc="white", mec=C_BLUE, lw=1.3, label=r"$\min(d_{\rm pipe})$"),
    Line2D([0], [0], color=C_GREY, lw=0.5, ls="--", label=r"$y=x$ ideal"),
], fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC", loc="upper left", labelspacing=0.2, borderpad=0.3)
ax_a1.grid(True, axis="y", linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_a1.spines["top"].set_visible(False)

# ── Panel (b): λ 灵敏度 (固定 d_safe=0.5) ──
lam_arr    = np.array([r["lam"] for r in lam_results])
lam_unsafe = np.array([r["n_unsafe"] for r in lam_results])
lam_rough  = np.array([r["roughness"] for r in lam_results])
lam_min_d  = np.array([r["min_d"] for r in lam_results])

ax_b1 = ax_b
ax_b2 = ax_b1.twinx()

ax_b1.plot(lam_arr, lam_rough, color=C_GREEN, linewidth=1.3, marker="s", ms=5.5,
           mfc="white", mec=C_GREEN, mew=1.0, label="Roughness")
ax_b2.plot(lam_arr, lam_min_d, color=C_BLUE, linewidth=1.3, marker="D", ms=5.5,
           mfc="white", mec=C_BLUE, mew=1.0, label=r"$\min(d_{\rm pipe})$")

# 标注不安全路点数
for lv, nu, r in zip(lam_arr, lam_unsafe, lam_rough):
    ax_b1.annotate(str(nu), (lv, r + 0.012), fontsize=7, color=C_RED, ha="center", fontweight="bold")
ax_b1.text(0.5, 0.08, "Numbers = unsafe wp count", transform=ax_b1.transAxes,
           fontsize=6.5, color=C_RED, ha="center", alpha=0.8)

# 高亮 λ=0.3 (部署值)
ax_b1.axvline(x=LAM_DEFAULT, color=C_GREY, linewidth=0.5, linestyle="--", alpha=0.4)

ax_b1.set_xlabel(r"$\lambda$ (smooth weight)", fontsize=8, labelpad=2)
ax_b1.set_ylabel(r"Roughness $\sigma$ (m)", fontsize=8, color=C_GREEN, labelpad=2)
ax_b2.set_ylabel(r"$\min(d_{\rm pipe})$ (m)", fontsize=8, color=C_BLUE, labelpad=4)
ax_b1.tick_params(axis="y", colors=C_GREEN)
ax_b2.tick_params(axis="y", colors=C_BLUE)
ax_b1.set_title("(b)  Sensitivity to $\\lambda$  ($d_{\\rm safe}$=%s m)" % str(D_DEPLOYED),
                fontsize=9, loc="left", pad=4)
ax_b1.legend(handles=[
    Line2D([0], [0], color=C_GREEN, marker="s", ms=5.5, mfc="white", mec=C_GREEN, lw=1.3, label="Roughness"),
    Line2D([0], [0], color=C_BLUE, marker="D", ms=5.5, mfc="white", mec=C_BLUE, lw=1.3, label=r"$\min(d_{\rm pipe})$"),
], fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC", loc="upper left", labelspacing=0.2, borderpad=0.3)
ax_b1.grid(True, axis="y", linestyle=":", color="#E0E0E0", linewidth=0.4, alpha=0.7)
ax_b1.spines["top"].set_visible(False)

fig.suptitle("Ablation Study — CBF Trajectory Safety Parameters  ($d_{\\rm safe}$=%s m deployed)" % str(D_DEPLOYED),
             fontsize=11, fontweight="bold", y=0.96)
fig.text(0.5, 0.04, "Baseline (demo trajectory): %d unsafe wp, min=%0.2fm" % (orig_unsafe, orig_min_d),
         fontsize=7, color=C_GREY, ha="center")

for fmt in ["png", "pdf", "svg"]:
    fpath = OUT_DIR / f"fig5_ablation.{fmt}"
    fig.savefig(str(fpath), dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved: {fpath}")

plt.close(fig)
print("Done — Figure 5: Ablation (d_safe=0.5)")
