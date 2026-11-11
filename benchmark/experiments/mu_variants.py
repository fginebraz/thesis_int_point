import sys, os, io, json, contextlib, builtins
ROOT = "/Users/fattybaking/Documents/thesis_int_point"
sys.path.insert(0, ROOT); os.chdir(ROOT)
builtins.display = print
import numpy as np, pandas as pd, scipy.linalg
import validate_benchmark as vb
from IPM_functions import (load_lp_problem, paso_intpoint, solve_catch_error, create_result_dataframes,
                           update_result_dataframes, update_active_set_mask, active_set_diagnostics,
                           remove_rows_cols_K)

def q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)

def load(mat):
    Q, c, A, b, F, d, H = q(load_lp_problem, mat)
    if mat.startswith("lp_"):
        Q = 2.0 * np.eye(Q.shape[0])
    return Q, c, A, b, F, d

def resid(P, x, l, mu, z):
    Q, c, A, b, F, d = P
    return (Q @ x - A.T @ l - F.T @ mu + c, -A @ x + b, -F @ x + d + z)

def kkt(P, x, l, mu, z):
    rx, rl, rm = resid(P, x, l, mu, z)
    return np.linalg.norm(np.concatenate((rx, rl, rm, mu * z)), np.inf)

def phase1(P, tol):
    Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    x = np.ones(n); l = np.zeros(m); mu = np.ones(p); z = np.ones(p)
    sigma = 0.5; tau = sigma * (mu @ z) / p
    mu_pct = np.zeros(p); z_pct = np.zeros(p)
    (mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df, ash, sv_df) = create_result_dataframes(n, m, p)
    dummy = np.full(n + m + p, np.nan)
    k = 0
    while k < 100:
        rx, rl, rm = resid(P, x, l, mu, z)
        q(update_result_dataframes, k, mu, z, tau, mu_pct, z_pct, 0.0, 0.0, n, m, p,
          mu_df, z_df, tau_df, mu_pct_df, z_pct_df, obj_df, maxc_df, sv_df, dummy)
        if kkt(P, x, l, mu, z) < tol:
            break
        D_mu = np.diag(mu)
        K = np.vstack((np.hstack((Q, -A.T, -F.T @ D_mu)), np.hstack((-A, np.zeros((m, m + p)))),
                       np.hstack((-D_mu @ F, np.zeros((p, m)), -np.diag(mu * z)))))
        L = np.concatenate((rx, rl, D_mu @ (d - F @ x) + tau))
        dv = q(solve_catch_error, K, -L, k)
        dx, dl, dmu = dv[:n], dv[n:n + m], D_mu @ dv[n + m:]
        dz = F @ dx + F @ x - d - z
        alpha = min(paso_intpoint(mu, dmu), paso_intpoint(z, dz))
        if alpha >= 0.9: sigma = max(1e-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5: sigma = min(0.5, sigma * 4)
        mu_pct = alpha * dmu / mu; z_pct = alpha * dz / z
        x = x + alpha * dx; mu = mu + alpha * dmu; l = l + alpha * dl; z = z + alpha * dz
        tau = sigma * (mu @ z) / p; k += 1
        ash = q(update_active_set_mask, mu, z, Q, k, tau, ash, mu_df, mu_pct, z_pct)
    stable, _ = q(active_set_diagnostics, ash, pd.DataFrame(columns=ash.columns), p, vb.STAB_WINDOW, vb.STAB_MIN_HITS)
    return dict(x=x, l=l, mu=mu, z=z, sigma=sigma, tau=tau, k=k, stable=list(stable))

def phase2(P, S, variant, iters=5):
    Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    x, l, mu, z = S["x"].copy(), S["l"].copy(), S["mu"].copy(), S["z"].copy()
    sigma, tau = S["sigma"], S["tau"]
    elim = [] if variant == "K" else S["stable"]
    keep = [i for i in range(p) if i not in set(elim)]
    E = np.array(elim, dtype=int)
    if variant == "zero" and len(E):
        mu[E] = 0.0
    for _ in range(iters):
        rx, rl, rm = resid(P, x, l, mu, z)
        K1, D_mu1, L1 = q(remove_rows_cols_K, Q, A.T, F.T, np.diag(mu), A, F, np.diag(z), mu, z, x, l,
                          c, b, d, tau, elim, rx, rl)
        dv = scipy.linalg.solve(K1, -L1)
        dx, dl = dv[:n], dv[n:n + m]
        dmu = np.zeros(p); dmu[keep] = D_mu1 @ dv[n + m:]
        dz = F @ dx + F @ x - d - z
        alpha = min(paso_intpoint(mu[keep], dmu[keep]), paso_intpoint(z, dz))
        if alpha >= 0.9: sigma = max(1e-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5: sigma = min(0.5, sigma * 4)
        x = x + alpha * dx; mu = mu + alpha * dmu; l = l + alpha * dl; z = z + alpha * dz
        tau = sigma * (mu[keep] @ z[keep]) / max(len(keep), 1) if variant != "K" else sigma * (mu @ z) / p
        if len(E):
            if variant == "central": mu[E] = tau / z[E]
            elif variant == "shrink": mu[E] = sigma * mu[E]
    f = 0.5 * x @ Q @ x + c @ x
    return kkt(P, x, l, mu, z), f

def clarabel_f(P):
    Q, c, A, b, F, d = P
    R = dict(Q=Q, c=c, A=A, b=b, F=F, d=d)
    x_cl, _, _ = q(vb.oracle, R)
    return 0.5 * x_cl @ Q @ x_cl + c @ x_cl

if __name__ == "__main__":
    PROBS = [m for m, _ in vb.CORE] + ["lp_afiro.mat", "lp_kb2.mat", "lp_blend.mat", "lp_recipe.mat",
                                       "lp_share2b.mat", "lp_stocfor1.mat", "lp_grow7.mat", "lp_kleeminty_m5.mat"]
    VARIANTS = ["K", "freeze", "zero", "central", "shrink"]
    OUT = "/private/tmp/claude-501/-Users-fattybaking-Documents-thesis-int-point/73c1e319-c8c3-47dc-9ebb-c1f96042d36a/scratchpad/mu_variants.json"
    res = []
    for mat in PROBS:
        P = load(mat)
        tol = 1e-6 if mat.startswith("lp_") else 1e-7
        S = phase1(P, tol)
        fcl = clarabel_f(P)
        row = dict(mat=mat, k=S["k"], p=P[4].shape[0], n_elim=len(S["stable"]), kkt0=kkt(P, S["x"], S["l"], S["mu"], S["z"]))
        for v in VARIANTS:
            try:
                kk, f = phase2(P, S, v)
                row[v] = dict(kkt=float(kk), ferr=float(abs(f - fcl)), frel=float(abs(f - fcl) / max(1.0, abs(fcl))))
            except Exception as e:
                row[v] = dict(error=repr(e)[:120])
        res.append(row)
        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps(row), flush=True)
    print("DONE", flush=True)
