"""
Figures for the thesis (method from ipm_khat.py, removed mu_i = 0).

  fig_mu_z_trayectorias  : mu_i and z_i per IPM iteration (HS21)
  fig_kappa_iteraciones  : kappa(K) vs kappa(K-hat) over 20 extra iterations
  fig_K_vs_Khat          : entries of K and K-hat at convergence (HS21)
  fig_mu_z_tres_casos    : mu_i and z_i for three QPCBLEND restrictions (removed / not marked / active); LaTeX font
  fig_kappa_mmhs / fig_kappa_netlib : kappa(K) vs kappa(K-hat) at extra iterations 1 and 5 (dumbbells)
  fig_eigenvalues        : |eigenvalues| of K and K-hat at convergence (HS21, DUAL2, QPCBLEND)
  fig_kappa_reduccion_*  : orders of magnitude kappa drops (K -> K-hat), iterations 1 and 5
  fig_ratio_separation   : Clarabel z*/mu* of the 869 restrictions, sorted (rebuild of ratio_separation.pdf)
  fig_hist_cocientes     : z_i/mu_i at convergence for the restrictions with mu_i -> 0 per Clarabel

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
BLUE, ORANGE, GRAY, GREEN = "#2a78d6", "#eb6834", "#8a8984", "#1f9a5e"
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


# LaTeX look for figures that go in the thesis (12pt, letter, 1in margins -> \textwidth = 6.5in).
# Computer Modern via matplotlib's own fonts (no LaTeX install needed; BasicTeX lacks type1cm for usetex).
TEXTWIDTH_IN = 6.5
LATEX_RC = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
}


def save_latex(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220)
    plt.close(fig)


def fig_mu_z_tres_casos(mat="mm_QPCBLEND.mat", missed_idx=5, active_idx=45):
    """mu_i and z_i per IPM iteration for three QPCBLEND restrictions:
    (a) mu_i -> 0 and the method removes it (first removed index),
    (b) mu_i -> 0 per Clarabel but the method does not mark it (index 5; mu_i ~ z_i ~ 1e-4 at the end),
    (c) active: z_i -> 0 per both Clarabel and our IPM (index 45; z/mu ~ 1e-10 in both).
    Panel descriptions go in the LaTeX caption; the figure only carries (a), (b), (c)."""
    MU, Z, flags, E = ipm_history(mat)
    it = np.arange(MU.shape[0]); last = it[-1]
    cases = [(r"(a) $\mu_i \to 0$, se elimina", sorted(E)[0]),
             (r"(b) $\mu_i \to 0$, no se marca", missed_idx),
             (r"(c) activa, $z_i \to 0$", active_idx)]
    with plt.rc_context({**LATEX_RC, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9,
                                 "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                                 "xtick.major.size": 2.5, "ytick.major.size": 2.5}):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.25), sharey=True)
        for ax, (tag, i) in zip(axes, cases):
            ax.plot(it, Z[:, i], color=ORANGE, ls="--", lw=0.8)
            ax.plot(it, MU[:, i], color=BLUE, lw=0.8)
            hit = flags[: len(it), i] > 0
            ax.plot(it[hit], MU[hit, i], "o", ms=2.8, color=BLUE, mec=SURF, mew=0.5, zorder=3)
            # direct labels at the end of each line
            close = abs(np.log10(MU[-1, i] / Z[-1, i])) < 0.8      # labels would overlap
            dy_mu, dy_z = ((-6, 6) if MU[-1, i] < Z[-1, i] else (6, -6)) if close else (0, 0)
            ax.annotate(r"$\mu_i$", (last, MU[-1, i]), xytext=(4, dy_mu), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
            ax.annotate(r"$z_i$", (last, Z[-1, i]), xytext=(4, dy_z), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
            ax.text(0.5, -0.42, tag, transform=ax.transAxes, fontsize=9, color=INK, ha="center", va="top")
            ax.set_yscale("log"); ax.set_xlabel(r"Iteraci$\acute{\mathrm{o}}$n del IPM")
            ax.set_xticks(range(0, len(it), 5)); ax.set_xlim(0, last + 0.5)
            ax.set_ylim(1e-10, 1e2); ax.set_yticks([1e-9, 1e-6, 1e-3, 1e0])
            ax.grid(which="minor", visible=False); ax.grid(axis="x", visible=False); ax.grid(axis="y", lw=0.4)
            ax.tick_params(which="minor", length=0)
        fig.tight_layout(w_pad=2.2)
        save_latex(fig, "fig_mu_z_tres_casos")


def fig_hist_cocientes():
    """z_i/mu_i at our IPM's convergence, for the restrictions Clarabel classifies as mu_i -> 0."""
    import json
    D = json.load(open(os.path.join(ROOT, "benchmark", "experiments", "ratio_ipm_vs_clarabel.json")))
    R = [r for d in D for r in d["rows"] if r["cls_cl"] == "mu0"]
    y = np.array([r["r_ipm"] for r in R]); rem = np.array([r["removed"] for r in R])
    bins = np.logspace(-1, 13, 29)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.hist([y[rem], y[~rem]], bins=bins, stacked=True, color=[BLUE, GRAY], edgecolor=SURF, linewidth=0.8,
            label=[f"se elimina ({rem.sum()})", f"no se marca ({(~rem).sum()})"])
    ax.axvline(100, color=INK2, lw=0.8, ls=":")
    ax.text(130, ax.get_ylim()[1] * 0.45, r"$z_i/\mu_i = 100$", fontsize=8.5, color=INK2, va="top")
    ax.set_xscale("log"); ax.grid(axis="x", visible=False)
    ax.set_xlabel(r"$z_i/\mu_i$ al converger el IPM"); ax.set_ylabel("Restricciones")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.2, 1.0))
    save(fig, "fig_hist_cocientes")


def _dumbbell(ax, names, kK, kKh, title):
    y = np.arange(len(names))[::-1]
    for yy, a, b in zip(y, kK, kKh):
        if b is not None:
            if abs(np.log10(a / b)) > 0.3:   # arrow from K to K-hat (skip when they coincide)
                ax.annotate("", xy=(b, yy), xytext=(a, yy), zorder=1,
                            arrowprops=dict(arrowstyle="-|>", color="#b4b3ae", lw=1.0,
                                            mutation_scale=7, shrinkA=3.5, shrinkB=3.5))
            ax.plot(b, yy, "o", ms=4, color=GREEN, zorder=3)
        ax.plot(a, yy, "o", ms=4, color=BLUE, zorder=2)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_title(title, color=INK); ax.grid(axis="y", visible=False); ax.grid(axis="x", lw=0.4)
    ax.grid(which="minor", visible=False); ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="y", length=0)


def fig_kappa_dumbbell(which=("mmhs", "netlib")):
    """kappa(K) vs kappa(K-hat) at extra iterations 1 and 5: MM/HS and NETLIB (one figure each)."""
    import json
    small = dict(LATEX_RC, **{"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9,
                              "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5})
    # MM/HS
    rows, nrows = [], []
    mm = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat",
          "mm_DUAL2.mat", "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat"]
    for mat in (mm if "mmhs" in which else []):
        S = ik.ipm_with_K(mat)
        RK = ik.extra_iterations(S, False); RH = ik.extra_iterations(S, True)
        rows.append((mat[3:-4], RK["extra"][0]["kappa"], RH["extra"][0]["kappa"], RK["kappa_final"], RH["kappa_final"]))
    # NETLIB (from the official run)
    N = json.load(open(os.path.join(ROOT, "benchmark", "experiments", "netlib_official.json"))) if "netlib" in which else []
    nrows = [(r["name"].replace("_", " "), r["K"]["kappa1"], r["Kh"]["kappa1"] if r["n_removed"] else None,
              r["K"]["kappa_final"], r["Kh"]["kappa_final"] if r["n_removed"] else None)
             for r in N if "KLEEMINTY" not in r["name"].upper()]
    for name, R, h in (("fig_kappa_mmhs", rows, 2.4), ("fig_kappa_netlib", nrows, 4.0)):
        if not R: continue
        names = [r[0] for r in R]
        with plt.rc_context(small):
            fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, h), sharey=True, sharex=True)
            _dumbbell(axes[0], names, [r[1] for r in R], [r[2] for r in R], "Al converger")
            _dumbbell(axes[1], names, [r[3] for r in R], [r[4] for r in R], "Tras 5 iteraciones adicionales")
            for ax in axes: ax.set_xlabel(r"$\kappa_1$", fontsize=9)
            handles = [plt.Line2D([], [], marker="o", ls="none", ms=4, color=BLUE, label=r"$K$"),
                       plt.Line2D([], [], marker="o", ls="none", ms=4, color=GREEN, label=r"$\widehat{K}$")]
            fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05), fontsize=9, frameon=False)
            fig.tight_layout(w_pad=1.5)
            save_latex(fig, name)


def fig_eigenvalues(probs=(("mm_HS21.mat", "HS21"), ("mm_DUAL2.mat", "DUAL2"), ("mm_QPCBLEND.mat", "QPCBLEND"))):
    """|eigenvalues| of K at convergence, sorted. Each eigenvalue is coloured by whether its eigenvector lives on the
    mu-coordinates of the removed restrictions (weight > 1/2): those are the ones K-hat removes. The rest coincide
    with the eigenvalues of K-hat (checked: relative change <= 1e-5)."""
    small = dict(LATEX_RC, **{"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                              "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                              "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    with plt.rc_context(small):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.2), sharey=True)
        for ax, (mat, name) in zip(axes, probs):
            S = ik.ipm_with_K(mat); P = S["P"]; n, m, p = S["n"], S["m"], S["p"]
            x, l, mu, z = S["x"], S["lamda"], S["mu"], S["z"]
            rx, rl, _ = ik.residuals(P, x, l, mu, z); E = S["removed"]
            K, _ = ik.build_K(P, x, mu, z, S["tau"], rx, rl, np.arange(p))
            w, V = np.linalg.eigh(K)
            removed = (V[n + m + E, :] ** 2).sum(axis=0) > 0.5
            order = np.argsort(-np.abs(w)); w, removed = np.abs(w[order]), removed[order]
            idx = np.arange(1, len(w) + 1)
            ms = 3.2 if len(w) < 20 else 1.8
            ax.plot(idx[removed], w[removed], "o", ms=ms, mfc=SURF, mec=GRAY, mew=0.7, ls="none")
            ax.plot(idx[~removed], w[~removed], "o", ms=ms, color=BLUE, mew=0, ls="none")
            ax.set_yscale("log"); ax.set_title(name, color=INK); ax.set_xlabel(r"$i$")
            ax.grid(which="minor", visible=False); ax.grid(axis="x", visible=False); ax.grid(axis="y", lw=0.4)
            ax.tick_params(which="minor", length=0)
        axes[0].set_ylabel(r"$|\lambda_i(K)|$")
        handles = [plt.Line2D([], [], marker="o", ls="none", ms=4, mfc=SURF, mec=GRAY, mew=1, label="se eliminan"),
                   plt.Line2D([], [], marker="o", ls="none", ms=4, color=BLUE, label=r"quedan en $\widehat{K}$")]
        fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=9, frameon=False)
        fig.tight_layout(w_pad=1.5)
        save_latex(fig, "fig_eigenvalues")


def fig_eigenvalues_strip(probs=(("mm_HS21.mat", "HS21"), ("mm_DUAL2.mat", "DUAL2"), ("mm_QPCBLEND.mat", "QPCBLEND"))):
    """Strip plot: |eigenvalues| of K (blue) and K-hat (green) at convergence, one row each per problem, log axis.
    The horizontal span of each row is kappa_2 (max|lambda| / min|lambda|)."""
    small = dict(LATEX_RC, **{"xtick.labelsize": 8, "ytick.labelsize": 9, "axes.labelsize": 9,
                              "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5})
    rows = []
    for mat, name in probs:
        S = ik.ipm_with_K(mat); P = S["P"]; p = S["p"]
        x, l, mu, z, E = S["x"], S["lamda"], S["mu"], S["z"], S["removed"]
        rx, rl, _ = ik.residuals(P, x, l, mu, z)
        K, _ = ik.build_K(P, x, mu, z, S["tau"], rx, rl, np.arange(p))
        muh = mu.copy(); muh[E] = 0.0; keep = np.setdiff1d(np.arange(p), E)
        rx, rl, _ = ik.residuals(P, x, l, muh, z)
        Kh, _ = ik.build_K(P, x, muh, z, S["tau"], rx, rl, keep)
        rows.append((name, np.abs(np.linalg.eigvalsh(K)), np.abs(np.linalg.eigvalsh(Kh))))
    with plt.rc_context(small):
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.6))
        yt, yl = [], []
        y = 0.0
        for name, wK, wH in rows:
            for w, lab, col in ((wK, r"$K$", BLUE), (wH, r"$\widehat{K}$", GREEN)):
                ax.hlines(y, w.min(), w.max(), color=col, lw=0.6, alpha=0.6)
                ax.plot(w, np.full_like(w, y), "|", ms=7, mew=0.7, color=col)
                yt.append(y); yl.append(f"{name}  {lab}")
                y -= 0.6
            y -= 0.5
        ax.set_xscale("log")
        ax.set_yticks(yt); ax.set_yticklabels(yl)
        ax.set_xlabel(r"$|\lambda|$")
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], color=c, marker="o", ms=5, lw=0) for c in (BLUE, GREEN)]
        ax.legend(hd, [r"$K$", r"$\widehat{K}$"], loc="lower left", bbox_to_anchor=(0, 1.0),
                  ncol=2, frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", which="major", lw=0.4)
        ax.grid(which="minor", visible=False); ax.tick_params(which="minor", length=0)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        fig.tight_layout()
        save_latex(fig, "fig_eigenvalues_strip")

def fig_kappa_reduccion():
    """Orders of magnitude kappa drops, log10(kappa(K)/kappa(K-hat)), at extra iterations 1 and 5 (MM/HS + NETLIB)."""
    import json
    small = dict(LATEX_RC, **{"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                              "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5})
    mm = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat",
          "mm_DUAL2.mat", "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat"]
    rows = []
    for mat in mm:
        S = ik.ipm_with_K(mat)
        RK = ik.extra_iterations(S, False); RH = ik.extra_iterations(S, True)
        rows.append((mat[3:-4], np.log10(RK["extra"][0]["kappa"] / RH["extra"][0]["kappa"]),
                     np.log10(RK["extra"][4]["kappa"] / RH["extra"][4]["kappa"])))
    N = json.load(open(os.path.join(ROOT, "benchmark", "experiments", "netlib_official.json")))
    nrows = [(r["name"].replace("_", " "),
              np.log10(r["K"]["kappa1"] / r["Kh"]["kappa1"]) if r["n_removed"] else 0.0,
              np.log10(r["K"]["kappa5"] / r["Kh"]["kappa5"]) if r["n_removed"] else 0.0) for r in N]
    for name, R, h in (("fig_kappa_reduccion_mmhs", rows, 2.3), ("fig_kappa_reduccion_netlib", nrows, 4.0)):
        names = [r[0] for r in R]; y = np.arange(len(R))[::-1]
        with plt.rc_context(small):
            fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, h), sharey=True, sharex=True)
            for ax, col, title in ((axes[0], 1, "Iteraci$\\acute{\\mathrm{o}}$n 1"), (axes[1], 2, "Iteraci$\\acute{\\mathrm{o}}$n 5")):
                v = np.array([r[col] for r in R])
                ax.barh(y, v, height=0.6, color=[BLUE if t >= 0 else GRAY for t in v])
                ax.axvline(0, color=INK2, lw=0.6)
                ax.set_yticks(y); ax.set_yticklabels(names); ax.set_title(title, color=INK)
                ax.set_xlabel(r"$\log_{10}\big(\kappa(K)/\kappa(\widehat{K})\big)$ (\'ordenes que baja $\kappa$)" if False else
                              r"Órdenes de magnitud que baja $\kappa$".replace("Ó", r"$\acute{\mathrm{O}}$"))
                ax.grid(axis="y", visible=False); ax.grid(axis="x", lw=0.4)
                ax.tick_params(axis="y", length=0)
            fig.tight_layout(w_pad=1.5)
            save_latex(fig, name)


def fig_ratio_separation():
    """Clarabel's z*/mu* for the 869 restrictions of the 9 MM/HS problems, sorted (replaces the old ratio_separation.pdf,
    whose script was lost). Included at 0.85\\textwidth, so it is drawn at that width."""
    import json
    D = json.load(open(os.path.join(ROOT, "benchmark", "experiments", "ratio_ipm_vs_clarabel.json")))
    r = np.sort(np.clip([row["r_cl"] for d in D for row in d["rows"]], 1e-14, None))
    amb = int(((r >= 1e-2) & (r <= 1e2)).sum())
    small = dict(LATEX_RC, **{"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9,
                              "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                              "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    with plt.rc_context(small):
        fig, ax = plt.subplots(figsize=(0.85 * TEXTWIDTH_IN, 2.8))
        ax.axhspan(1e-2, 1e2, color=GRID, alpha=0.7, lw=0)
        for v in (1e2, 1e-2):
            ax.axhline(v, color=INK2, lw=0.6, ls="--")
        ax.plot(np.arange(len(r)), r, "o", ms=1.6, color=BLUE, mew=0)
        ax.set_yscale("log"); ax.set_xlim(-10, len(r) + 10)
        ax.set_xlabel(r"Restricci$\acute{\mathrm{o}}$n (ordenadas por cociente)")
        ax.set_ylabel(r"$z_i^*/\mu_i^*$")
        ax.text(len(r) * 0.62, 1e2 * 2.5, r"$z_i^*/\mu_i^* = 100$", fontsize=8, color=INK2)
        ax.text(len(r) * 0.62, 1e-2 / 30, r"$z_i^*/\mu_i^* = \frac{1}{100}$", fontsize=8, color=INK2)
        ax.grid(which="minor", visible=False); ax.grid(axis="x", visible=False); ax.grid(axis="y", lw=0.4)
        ax.tick_params(which="minor", length=0)
        fig.tight_layout()
        save_latex(fig, "fig_ratio_separation")
    print("restricciones:", len(r), " en la zona ambigua:", amb)


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
