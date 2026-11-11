"""
ipm_khat.py
===========
Official version of the method:

  1. Primal-dual IPM with the full system K until ||KKT||_inf < tol.
     During the IPM the five criteria mark indices with mu_i -> 0
     (update_active_set_mask), and an index is confirmed if it was marked
     in at least 2 of the last 4 iterations (active_set_diagnostics).
  2. The confirmed indices are removed: mu_i := 0 from then on, and their
     rows/columns are dropped from K, giving K-hat.
  3. EXTRA_ITERS (= 5) additional iterations with K-hat.

Differences with run_ipm_refine.py (kept for reference):
  - removed mu_i are set to 0 (before, the old value stayed in mu and kept
    entering r_x and the KKT norm);
  - tau is computed only with the kept indices;
  - exactly 5 extra iterations (before: max_refine=20);
  - NETLIB problems use Q = 2I (min x'x + c'x), as in the thesis.

Use `solve(mat, use_khat=False)` to do the same extra iterations with K,
for comparison tables.

Run:  .venv/bin/python ipm_khat.py
"""
import io
import builtins
import contextlib
import numpy as np
import pandas as pd
import scipy.linalg

from IPM_functions import (load_lp_problem, create_result_dataframes,
                           update_result_dataframes, paso_intpoint,
                           solve_catch_error, update_active_set_mask,
                           active_set_diagnostics)

# IPM_functions calls display() (from Jupyter); outside a notebook, make it a no-op.
if not hasattr(builtins, "display"):
    builtins.display = lambda *a, **k: None

EXTRA_ITERS = 5
WINDOW, MIN_HITS = 4, 2


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def load(mat_file):
    """Returns (Q, c, A, b, F, d). For NETLIB (lp_*): Q = 2I."""
    Q, c, A, b, F, d, _ = _quiet(load_lp_problem, mat_file)
    if mat_file.startswith("lp_"):
        Q = 2.0 * np.eye(Q.shape[0])
    return Q, c, A, b, F, d


def residuals(P, x, lamda, mu, z):
    Q, c, A, b, F, d = P
    r_x = Q @ x - A.T @ lamda - F.T @ mu + c
    r_lamda = -A @ x + b
    r_mu = -F @ x + d + z
    return r_x, r_lamda, r_mu


def kkt_norm(P, x, lamda, mu, z):
    r_x, r_lamda, r_mu = residuals(P, x, lamda, mu, z)
    return np.linalg.norm(np.concatenate((r_x, r_lamda, r_mu, mu * z)), np.inf)


def build_K(P, x, mu, z, tau, r_x, r_lamda, keep):
    """K restricted to the indices in `keep` (keep = all -> K, otherwise K-hat).
    The mu-block uses the scaled variable D_mu^{-1} delta_mu, as in the thesis."""
    Q, c, A, b, F, d = P
    n, m = Q.shape[0], A.shape[0]
    Fk, muk, zk, dk = F[keep], mu[keep], z[keep], d[keep]
    pk = len(keep)
    D_mu = np.diag(muk)
    K = np.vstack((
        np.hstack((Q, -A.T, -Fk.T @ D_mu)),
        np.hstack((-A, np.zeros((m, m + pk)))),
        np.hstack((-D_mu @ Fk, np.zeros((pk, m)), -np.diag(muk * zk))),
    ))
    L = np.concatenate((r_x, r_lamda, D_mu @ (dk - Fk @ x) + tau))
    return K, L


def newton_step(P, x, lamda, mu, z, sigma, tau, keep, k):
    """One IPM iteration with K (keep = all) or K-hat. mu outside `keep` stays at 0."""
    Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    r_x, r_lamda, _ = residuals(P, x, lamda, mu, z)
    K, L = build_K(P, x, mu, z, tau, r_x, r_lamda, keep)
    dv = _quiet(solve_catch_error, K, -L, k) if len(keep) == p else scipy.linalg.solve(K, -L)

    dx, dlamda = dv[:n], dv[n:n + m]
    dmu = np.zeros(p)
    dmu[keep] = mu[keep] * dv[n + m:]
    dz = F @ dx + F @ x - d - z

    alpha = min(paso_intpoint(mu[keep], dmu[keep]), paso_intpoint(z, dz))
    if alpha >= 0.9:
        sigma = max(1e-4, sigma / 4)
    if alpha <= 0.75 and sigma < 0.5:
        sigma = min(0.5, sigma * 4)

    mu_pct, z_pct = np.zeros(p), alpha * dz / z
    mu_pct[keep] = alpha * dmu[keep] / mu[keep]

    x = x + alpha * dx
    lamda = lamda + alpha * dlamda
    mu = mu + alpha * dmu
    z = z + alpha * dz
    tau = sigma * (mu[keep] @ z[keep]) / len(keep) if len(keep) else 0.0
    return x, lamda, mu, z, sigma, tau, alpha, mu_pct, z_pct, K


KAPPA_AT = (1, 5)   # extra iterations at which kappa (1-norm) is computed


def ipm_with_K(mat_file, tol=None, kmax=100):
    """Steps 1-2: IPM with K until convergence and the confirmed mu_i -> 0 set."""
    P = load(mat_file)
    Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    if tol is None:
        tol = 1e-6 if mat_file.startswith("lp_") else 1e-7
    f = lambda x: 0.5 * x @ Q @ x + c @ x

    x, lamda, mu, z = np.ones(n), np.zeros(m), np.ones(p), np.ones(p)
    sigma = 0.5
    tau = sigma * (mu @ z) / p
    allidx = np.arange(p)

    (mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df,
     ash, sv_df) = create_result_dataframes(n, m, p)
    mu_pct, z_pct = np.zeros(p), np.zeros(p)
    no_sv = np.full(n + m + p, np.nan)

    k = 0
    while k < kmax:
        _quiet(update_result_dataframes, k, mu, z, tau, mu_pct, z_pct, f(x), np.max(mu * z),
               n, m, p, mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df, sv_df, no_sv)
        if kkt_norm(P, x, lamda, mu, z) < tol:
            break
        x, lamda, mu, z, sigma, tau, _, mu_pct, z_pct, _ = newton_step(
            P, x, lamda, mu, z, sigma, tau, allidx, k)
        k += 1
        ash = _quiet(update_active_set_mask, mu, z, Q, k, tau, ash, mu_df, mu_pct, z_pct)

    stable, _ = _quiet(active_set_diagnostics, ash, pd.DataFrame(columns=ash.columns),
                       p, WINDOW, MIN_HITS)
    E = np.array(sorted(stable), dtype=int)
    return dict(mat=mat_file, P=P, n=n, m=m, p=p, k=k, x=x, lamda=lamda, mu=mu, z=z,
                sigma=sigma, tau=tau, f_conv=f(x), kkt_conv=kkt_norm(P, x, lamda, mu, z),
                removed=E, mu_removed_max=float(mu[E].max()) if len(E) else 0.0)


def extra_iterations(S, use_khat=True, extra_iters=EXTRA_ITERS, kappa_at=KAPPA_AT):
    """Step 3: extra iterations with K-hat (removed mu_i := 0) or, for comparison, with K."""
    P = S["P"]
    Q, c = P[0], P[1]
    p, E = S["p"], S["removed"]
    f = lambda x: 0.5 * x @ Q @ x + c @ x
    x, lamda, mu, z = S["x"].copy(), S["lamda"].copy(), S["mu"].copy(), S["z"].copy()
    sigma, allidx = S["sigma"], np.arange(p)
    if use_khat and len(E) > 0:
        mu[E] = 0.0
        keep = np.setdiff1d(allidx, E)
    else:
        keep = allidx
    tau = sigma * (mu[keep] @ z[keep]) / len(keep) if len(keep) else 0.0

    traj = []
    for i in range(extra_iters):
        x, lamda, mu, z, sigma, tau, alpha, _, _, K = newton_step(
            P, x, lamda, mu, z, sigma, tau, keep, S["k"] + i)
        traj.append(dict(kkt=kkt_norm(P, x, lamda, mu, z), f=f(x), alpha=alpha,
                         kappa=float(np.linalg.cond(K, 1)) if (i + 1) in kappa_at else None))
    return dict(x=x, lamda=lamda, mu=mu, z=z, extra=traj, f=f(x),
                kkt=kkt_norm(P, x, lamda, mu, z), dim=S["n"] + S["m"] + len(keep))


def solve(mat_file, tol=None, kmax=100, extra_iters=EXTRA_ITERS, use_khat=True):
    """Whole method for one problem (steps 1-3)."""
    S = ipm_with_K(mat_file, tol, kmax)
    R = extra_iterations(S, use_khat, extra_iters)
    out = {k: v for k, v in S.items() if k not in ("P", "x", "lamda", "mu", "z")}
    out.update(R, removed=S["removed"].tolist(), dim_K=S["n"] + S["m"] + S["p"], dim_Khat=R["dim"])
    return out


if __name__ == "__main__":
    PROBLEMS = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat",
                "mm_DUAL1.mat", "mm_DUAL2.mat", "mm_DUAL3.mat", "mm_DUAL4.mat",
                "mm_QPCBLEND.mat"]
    C0 = {"mm_HS21.mat": -100.0, "mm_HS35.mat": 9.0, "mm_HS53.mat": 6.0}
    print(f"{'problem':16s} {'k':>3} {'elim':>9} {'KKT conv':>9} {'KKT K+5':>9} {'KKT Kh+5':>9}"
          f" {'f* (K-hat)':>18}")
    for mat in PROBLEMS:
        RK = solve(mat, use_khat=False)
        RH = solve(mat)
        c0 = C0.get(mat, 0.0)
        print(f"{mat:16s} {RH['k']:>3} {len(RH['removed']):>4}/{RH['p']:<4} {RH['kkt_conv']:9.1e}"
              f" {RK['kkt']:9.1e} {RH['kkt']:9.1e} {RH['f'] + c0:18.10f}")
