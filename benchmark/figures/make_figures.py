"""
Figures for the thesis (method from ipm_khat.py, removed mu_i = 0).

  fig_mu_z_trayectorias  : mu_i and z_i per IPM iteration (HS21)
  fig_kappa_iteraciones  : kappa(K) vs kappa(K-hat) over 20 extra iterations
  fig_K_vs_Khat          : entries of K and K-hat at convergence (HS21)

Run:  .venv/bin/python benchmark/figures/make_figures.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = "/Users/fattybaking/Documents/thesis_int_point"
OUT = os.path.join(ROOT, "benchmark", "figures")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import ipm_khat as ik
from IPM_functions import create_result_dataframes, update_result_dataframes, update_active_set_mask

# palette (reference categorical slots 1-2 + neutrals)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


# ───────────────────────── 1. mu_i and z_i per iteration ─────────────────────────
def ipm_history(mat):
    """Same IPM as ik.ipm_with_K, but keeps mu, z and the per-iteration criteria mask."""
    P = ik.load(mat); Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    x, lamda, mu, z = np.ones(n), np.zeros(m), np.ones(p), np.ones(p)
    sigma = 0.5; tau = sigma * (mu @ z) / p; allidx = np.arange(p)
    (mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df, ash, sv_df) = create_result_dataframes(n, m, p)
    mu_pct, z_pct = np.zeros(p), np.zeros(p); no_sv = np.full(n + m + p, np.nan)
    MU, Z = [mu.copy()], [z.copy()]
    k = 0
    while k < 100:
        ik._quiet(update_result_dataframes, k, mu, z, tau, mu_pct, z_pct, 0.0, 0.0, n, m, p,
                  mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df, sv_df, no_sv)
        if ik.kkt_norm(P, x, lamda, mu, z) < 1e-7:
            break
        x, lamda, mu, z, sigma, tau, _, mu_pct, z_pct, _ = ik.newton_step(P, x, lamda, mu, z, sigma, tau, allidx, k)
        k += 1
        ash = ik._quiet(update_active_set_mask, mu, z, Q, k, tau, ash, mu_df, mu_pct, z_pct)
        MU.append(mu.copy()); Z.append(z.copy())
    S = ik.ipm_with_K(mat)
    flags = ash.astype(float).reindex(range(k + 1)).fillna(0).values  # rows: iteration, cols: i
    return np.array(MU), np.array(Z), flags, set(S["removed"].tolist())


def fig_mu_z():
    """mu_i and z_i of the restrictions that get removed (HS21)."""
    MU, Z, flags, E = ipm_history("mm_HS21.mat")
    it = np.arange(MU.shape[0])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i in sorted(E):
        ax.plot(it, Z[:, i], color=GRAY, ls="--", lw=1.2)
        ax.plot(it, MU[:, i], color=BLUE, lw=1.2)
        hit = flags[: len(it), i] > 0
        ax.plot(it[hit], MU[hit, i], "o", ms=4.5, color=BLUE, mec=SURF, mew=0.8)
    ax.set_yscale("log"); ax.set_xlabel("Iteración $k$"); ax.set_ylim(top=2e2)
    ax.set_xticks(range(0, len(it), 2))
    ax.plot([], [], color=BLUE, lw=1.2, label=r"$\mu_i$")
    ax.plot([], [], color=GRAY, lw=1.2, ls="--", label=r"$z_i$")
    ax.plot([], [], "o", color=BLUE, ms=4.5, label="cumple los 5 criterios")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "fig_mu_z_trayectorias")


def fig_mu_z_dos_casos(mat="mm_QPCBLEND.mat", missed_idx=5):
    """One restriction that gets removed vs one with mu_i -> 0 that the method does not mark."""
    MU, Z, flags, E = ipm_history(mat)
    it = np.arange(MU.shape[0])
    removed_idx = sorted(E)[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for ax, i, title in ((axes[0], removed_idx, "Se elimina"), (axes[1], missed_idx, "No se marca")):
        ax.plot(it, Z[:, i], color=GRAY, ls="--", lw=1.2)
        ax.plot(it, MU[:, i], color=BLUE, lw=1.2)
        hit = flags[: len(it), i] > 0
        ax.plot(it[hit], MU[hit, i], "o", ms=4.5, color=BLUE, mec=SURF, mew=0.8)
        ax.set_yscale("log"); ax.set_xlabel("Iteración $k$"); ax.set_title(f"{title} (restricción {i + 1})", color=INK)
        ax.set_xticks(range(0, len(it), 4))
    axes[0].plot([], [], color=BLUE, lw=1.2, label=r"$\mu_i$")
    axes[0].plot([], [], color=GRAY, lw=1.2, ls="--", label=r"$z_i$")
    axes[0].plot([], [], "o", color=BLUE, ms=4.5, label="cumple los 5 criterios")
    axes[0].legend(loc="lower left", fontsize=8.5)
    save(fig, "fig_mu_z_dos_casos")


# ───────────────────── 4. kappa over 20 extra iterations ─────────────────────
def fig_kappa():
    probs = [("mm_HS21.mat", "HS21"), ("lp_afiro.mat", "AFIRO"),
             ("mm_DUAL4.mat", "DUAL4"), ("mm_QPCBLEND.mat", "QPCBLEND")]
    N = 5
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    for ax, (mat, name) in zip(axes, probs):
        S = ik.ipm_with_K(mat)
        for use_khat, col, ls, lab in ((False, ORANGE, "--", r"$K$"), (True, BLUE, "-", r"$\widehat{K}$")):
            R = ik.extra_iterations(S, use_khat, extra_iters=N, kappa_at=range(1, N + 1))
            ax.plot(range(1, N + 1), [t["kappa"] for t in R["extra"]], color=col, ls=ls, lw=2, label=lab)
        ax.axhspan(1 / np.finfo(float).eps, 1e30, color=GRID, alpha=0.6, lw=0)
        ax.set_yscale("log"); ax.set_ylim(1, 1e22); ax.set_xlim(1, N)
        ax.set_xticks(range(1, N + 1)); ax.set_title(name, color=INK)
        ax.set_xlabel("Iteración adicional")
    axes[0].set_ylabel(r"$\kappa_1$")
    axes[0].legend(loc="center right", fontsize=9)
    axes[3].text(1.1, 1e19, r"$\kappa > 1/\varepsilon_{\mathrm{máq}}$", fontsize=9, color=INK2, va="bottom")
    save(fig, "fig_kappa_iteraciones")


# ───────────────────────── 3. K vs K-hat ─────────────────────────
def fig_K():
    S = ik.ipm_with_K("mm_HS21.mat")
    P = S["P"]; Q, c, A, b, F, d = P
    n, m, p = S["n"], S["m"], S["p"]; E = S["removed"].tolist()
    x, l, mu, z = S["x"], S["lamda"], S["mu"], S["z"]
    rx, rl, _ = ik.residuals(P, x, l, mu, z)
    K, _ = ik.build_K(P, x, mu, z, S["tau"], rx, rl, np.arange(p))
    keep = np.setdiff1d(np.arange(p), E)
    Kh, _ = ik.build_K(P, x, np.where(np.isin(np.arange(p), E), 0.0, mu), z, S["tau"], rx, rl, keep)
    labels_K = [f"$x_{j+1}$" for j in range(n)] + [f"$\\lambda_{j+1}$" for j in range(m)] + [f"$\\mu_{i+1}$" for i in range(p)]
    labels_Kh = [f"$x_{j+1}$" for j in range(n)] + [f"$\\lambda_{j+1}$" for j in range(m)] + [f"$\\mu_{i+1}$" for i in keep]
    rem = {n + m + i for i in E}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), gridspec_kw=dict(width_ratios=[K.shape[0], Kh.shape[0] + 1.5]))
    for ax, M, labs, title, shade in ((axes[0], K, labels_K, r"$K$", rem), (axes[1], Kh, labels_Kh, r"$\widehat{K}$", set())):
        r = M.shape[0]
        ax.set_xlim(-0.5, r - 0.5); ax.set_ylim(r - 0.5, -0.5); ax.set_aspect("equal")
        for a in range(r):
            for bb in range(r):
                if a in shade or bb in shade:
                    ax.add_patch(plt.Rectangle((bb - 0.5, a - 0.5), 1, 1, fc=ORANGE, alpha=0.15, lw=0))
                v = M[a, bb]
                txt = "0" if v == 0 else (f"{v:.0e}".replace("e-0", "e-") if abs(v) < 1e-3 else f"{v:.2g}")
                ax.text(bb, a, txt, ha="center", va="center", fontsize=7.5, color=INK2 if v == 0 else INK)
        ax.set_xticks(range(r)); ax.set_yticks(range(r))
        ax.set_xticklabels(labs, fontsize=8.5); ax.set_yticklabels(labs, fontsize=8.5)
        ax.set_xticks(np.arange(-0.5, r, 1), minor=True); ax.set_yticks(np.arange(-0.5, r, 1), minor=True)
        ax.grid(False); ax.grid(which="minor", color=GRID, lw=0.8); ax.tick_params(which="both", length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, color=INK)
    for j in rem:
        axes[0].get_xticklabels()[j].set_color(ORANGE)
        axes[0].get_yticklabels()[j].set_color(ORANGE)
    axes[0].text(0.5, -0.12, "Sombreadas: filas y columnas que se eliminan", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=8.5, color=INK2)
    save(fig, "fig_K_vs_Khat")


if __name__ == "__main__":
    fig_mu_z(); print("1 ok", flush=True)
    fig_K(); print("3 ok", flush=True)
    fig_kappa(); print("4 ok", flush=True)
