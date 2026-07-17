"""
validate_benchmark.py
=====================
Reproducible validation of the interior-point method + mu->0 elimination
heuristic against an INDEPENDENT solver (Clarabel, cross-checked with OSQP),
using the Maros-Meszaros QP set and the NETLIB-derived ill-conditioned problems.

Produces three tables:
  1. DETECTION      -- does the heuristic find the true mu->0 (inactive) set?
  2. CONDITIONING   -- does eliminating those rows improve kappa(K -> K1)?
  3. STRICT COMPL.  -- is the strict-complementarity assumption satisfied?

Run:   .venv/bin/python validate_benchmark.py
The IPM is driven here with the REAL IPM_functions detection code, so the
detection logic is identical to the notebook.
"""
import os, io, contextlib
import numpy as np
import scipy.linalg
import scipy.sparse as sp

import IPM_functions
from IPM_functions import (load_lp_problem, paso_intpoint, solve_catch_error,
                           create_result_dataframes, update_result_dataframes,
                           update_active_set_mask, active_set_diagnostics,
                           remove_rows_cols_K)
import pandas as pd
from qpsolvers import Problem, solve_problem

EPS = np.finfo(float).eps
SLACK_TOL = 1e-6
STAB_WINDOW = 2            # active_set_diagnostics window (the fixed default)

# problem set: (mat_file, label)
CORE = [("mm_TAME.mat","TAME"), ("mm_HS21.mat","HS21"), ("mm_HS35.mat","HS35"),
        ("mm_HS53.mat","HS53"), ("mm_DUAL4.mat","DUAL4"), ("mm_DUAL1.mat","DUAL1"),
        ("mm_DUAL2.mat","DUAL2"), ("mm_DUAL3.mat","DUAL3"), ("mm_QPCBLEND.mat","QPCBLEND")]
CONDITIONING = [("lp_kb2.mat","kb2"), ("lp_blend.mat","blend"),
                ("lp_fit1d.mat","fit1d"), ("lp_fit1p.mat","fit1p")]
PROBLEMS = CORE + CONDITIONING


def _quiet(fn, *a, **k):
    """Call fn suppressing its prints (the detector/diagnostics are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def run_ipm(mat_file, tol=1e-7, kmax=200):
    """Drive the thesis IPM (matches the notebook loop) with the real detector."""
    Q, c, A, b, F, d, H = _quiet(load_lp_problem, mat_file)
    AT, FT = A.T, F.T
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    x = np.ones(n); lamda = np.zeros(m); mu = np.ones(p); z = np.ones(p)
    sigma = 0.5; tau = sigma * (mu @ z) / p
    mu_pct = np.zeros(p); z_pct = np.zeros(p)
    obj = 0.5 * x @ Q @ x + c @ x; maxc = np.max(mu * z)

    (mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df,
     ash, sv_df) = create_result_dataframes(n, m, p)
    regression_df = pd.DataFrame(columns=ash.columns)
    dummy_sv = np.full(n + m + p, np.nan)   # skip per-iter SVD (not needed here)

    k = 0
    K = L = r_x = r_lamda = D_mu = D_z = None
    while k < kmax:
        D_z = np.diag(z); D_mu = np.diag(mu)
        r_x = Q @ x - AT @ lamda - FT @ mu + c
        r_lamda = -A @ x + b
        r_mu = -F @ x + d + z
        KKT = np.linalg.norm(np.concatenate((r_x, r_lamda, r_mu, mu * z)), np.inf)
        K = np.vstack((np.hstack((Q, -AT, -FT @ D_mu)),
                       np.hstack((-A, np.zeros((m, m + p)))),
                       np.hstack((-D_mu @ F, np.zeros((p, m)), -np.diag(mu * z)))))
        update_result_dataframes(k, mu, z, tau, mu_pct, z_pct, obj, maxc, n, m, p,
                                 mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
                                 obj_df, maxc_df, sv_df, dummy_sv)
        if KKT < tol:
            break
        L = np.concatenate((r_x, r_lamda, D_mu @ (d - F @ x) + tau))
        dv = _quiet(solve_catch_error, K, -L, k)
        dx, dl, Ddmu = dv[:n], dv[n:n+m], dv[n+m:]
        dmu = D_mu @ Ddmu
        dz = F @ dx + F @ x - d - z
        alpha = min(paso_intpoint(mu, dmu), paso_intpoint(z, dz))
        if alpha >= 0.9: sigma = max(1e-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5: sigma = min(0.5, sigma * 4)
        mu_pct = alpha * dmu / mu; z_pct = alpha * dz / z
        x += alpha * dx; mu += alpha * dmu; lamda += alpha * dl; z += alpha * dz
        tau = sigma * (mu @ z) / p; k += 1
        obj = 0.5 * x @ Q @ x + c @ x; maxc = np.max(mu * z)
        ash = _quiet(update_active_set_mask, mu, z, Q, k, tau, ash, mu_df, mu_pct, z_pct)

    return dict(mat=mat_file, Q=Q, c=c, A=A, b=b, F=F, d=d, AT=AT, FT=FT,
                x=x, lamda=lamda, mu=mu, z=z, n=n, m=m, p=p, k=k,
                K=K, r_x=r_x, r_lamda=r_lamda, D_mu=D_mu, D_z=D_z, tau=tau,
                ash=ash, regression_df=regression_df)


def oracle(R):
    """Independent solve: returns x*, dual mu*, and OSQP cross-check distance."""
    Q, c, A, b, F, d = R["Q"], R["c"], R["A"], R["b"], R["F"], R["d"]
    kw = dict(P=sp.csc_matrix(Q), q=c, G=sp.csc_matrix(-F), h=-d)
    if A.shape[0] > 0:
        kw.update(A=sp.csc_matrix(A), b=b)
    sol = solve_problem(Problem(**kw), solver="clarabel")
    x_cl, mu_cl = sol.x, np.maximum(np.asarray(sol.z).ravel(), 0.0)
    # OSQP cross-check via solve_qp (returns primal only) -- a second, independent opinion
    import qpsolvers
    try:
        x_os = qpsolvers.solve_qp(P=kw["P"], q=kw["q"], G=kw["G"], h=kw["h"],
                                  A=kw.get("A"), b=kw.get("b"), solver="osqp")
        agree = np.linalg.norm(x_cl - x_os, np.inf) if x_os is not None else np.nan
    except Exception:
        agree = np.nan
    return x_cl, mu_cl, agree


def kappa(M):
    return np.linalg.cond(M, 1)


def main():
    rows = []
    for mat, label in PROBLEMS:
        R = run_ipm(mat)
        x_cl, mu_cl, agree = oracle(R)
        F, d, p = R["F"], R["d"], R["p"]
        slack = np.maximum(F @ x_cl - d, 0.0)
        truth_inactive = slack > SLACK_TOL                       # oracle mu->0
        # heuristic's stable eliminated set
        stable, _ = _quiet(active_set_diagnostics, R["ash"], R["regression_df"], p, STAB_WINDOW)
        heur = np.zeros(p, bool); heur[stable] = True
        tp = int((heur & truth_inactive).sum()); fp = int((heur & ~truth_inactive).sum())
        sol_err = np.linalg.norm(R["x"] - x_cl, np.inf)
        # conditioning
        K = R["K"]
        K1, _, _ = remove_rows_cols_K(R["Q"], R["AT"], R["FT"], R["D_mu"], R["A"], R["F"],
                                      R["D_z"], R["mu"], R["z"], R["x"], R["lamda"],
                                      R["c"], R["b"], R["d"], R["tau"], stable,
                                      R["r_x"], R["r_lamda"]) if len(stable) < p else (K, None, None)
        with contextlib.redirect_stdout(io.StringIO()):
            kK = kappa(K); kK1 = kappa(K1)
        sK = np.linalg.svd(K, compute_uv=False)
        unreliable = sK[-1] < sK[0] * EPS * K.shape[0]           # sigma_min below noise floor
        # strict complementarity
        mu_small = mu_cl <= SLACK_TOL
        n_deg = int((mu_small & ~truth_inactive).sum())          # slack~0 AND mu~0
        margin = float(np.min(np.maximum(slack, mu_cl))) if p else np.nan
        rows.append(dict(label=label, n=R["n"], p=p, k=R["k"], sol_err=sol_err,
                         n_in=int(truth_inactive.sum()), tp=tp, fp=fp, agree=agree,
                         pct=100*len(stable)/p, kK=kK, kK1=kK1, unrel=unreliable,
                         n_act=int((~truth_inactive).sum()), n_deg=n_deg, margin=margin))

    # ---- TABLE 1: DETECTION ----
    print("\n================= TABLE 1: DETECTION (heuristic vs Clarabel) =================")
    print(f"{'problem':9s} {'n':>5} {'p':>5} {'it':>3} {'|x_IPM-x_orac|':>14} "
          f"{'true_mu0':>8} {'detect':>6} {'false+':>6} {'orac.agree':>10}")
    for r in rows:
        print(f"{r['label']:9s} {r['n']:>5} {r['p']:>5} {r['k']:>3} {r['sol_err']:>14.1e} "
              f"{r['n_in']:>8} {r['tp']:>6} {r['fp']:>6} {r['agree']:>10.1e}")

    # ---- TABLE 2: CONDITIONING ----
    print("\n================= TABLE 2: CONDITIONING PAYOFF (kappa K -> K1) ===============")
    print(f"{'problem':9s} {'%elim':>6} {'kappa(K)':>11} {'kappa(K1)':>11} {'improve':>9}  note")
    for r in rows:
        note = "kappa unreliable (sigma_min below precision)" if r['unrel'] else ""
        imp = r['kK']/r['kK1'] if r['kK1'] else np.nan
        print(f"{r['label']:9s} {r['pct']:>5.0f}% {r['kK']:>11.2e} {r['kK1']:>11.2e} "
              f"{imp:>9.1e}  {note}")

    # ---- TABLE 3: STRICT COMPLEMENTARITY ----
    print("\n================= TABLE 3: STRICT COMPLEMENTARITY ===========================")
    print(f"{'problem':9s} {'active':>7} {'inactive':>9} {'DEGEN':>6} {'SC margin':>11}  holds?")
    for r in rows:
        print(f"{r['label']:9s} {r['n_act']:>7} {r['n_in']:>9} {r['n_deg']:>6} "
              f"{r['margin']:>11.2e}  {'YES' if r['n_deg']==0 else 'NO'}")
    print("\ndetect/false+ vs Clarabel's true mu->0 set; orac.agree = Clarabel-vs-OSQP distance")
    print("(small => trustworthy key). kappa 'unreliable' = double precision can't resolve it.")


if __name__ == "__main__":
    main()
