# Curiosity experiment: how many MINRES iterations does it take to solve the Newton system with K vs K-hat?
# (Montiel solves his reduced system with MINRES; here we just count iterations on our K and K-hat, no preconditioner.)
# Two points per problem: at IPM convergence, and after the 5 extra iterations.
import sys, json, time
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point")
import numpy as np
import scipy.sparse.linalg as spla
import ipm_khat as ik

PROBS = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat", "mm_DUAL2.mat",
         "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat",
         "lp_afiro.mat", "lp_kb2.mat", "lp_blend.mat", "lp_recipe.mat", "lp_share2b.mat", "lp_grow7.mat",
         "lp_scsd1.mat", "lp_kleeminty_m5.mat"]
RTOL = 1e-10
OUT = "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/minres_K_vs_Khat.json"


def run_minres(M, rhs):
    it = [0]
    def cb(_):
        it[0] += 1
    maxit = 20 * M.shape[0]
    try:
        x, info = spla.minres(M, rhs, rtol=RTOL, maxiter=maxit, callback=cb)
    except TypeError:                       # older scipy
        x, info = spla.minres(M, rhs, tol=RTOL, maxiter=maxit, callback=cb)
    relres = float(np.linalg.norm(M @ x - rhs) / np.linalg.norm(rhs))
    return dict(iters=it[0], converged=(info == 0), relres=relres, maxit=maxit)


def systems(S, use_khat, final):
    """(matrix, rhs) at convergence, or at the point after the 5 extra iterations."""
    P, p, E = S["P"], S["p"], S["removed"]
    if final:
        R = ik.extra_iterations(S, use_khat, kappa_at=())
        x, l, mu, z = R["x"], R["lamda"], R["mu"], R["z"]
    else:
        x, l, mu, z = S["x"], S["lamda"], S["mu"].copy(), S["z"]
        if use_khat and len(E):
            mu[E] = 0.0
    keep = np.setdiff1d(np.arange(p), E) if (use_khat and len(E)) else np.arange(p)
    rx, rl, _ = ik.residuals(P, x, l, mu, z)
    K, L = ik.build_K(P, x, mu, z, S["tau"], rx, rl, keep)
    return K, -L


res = []
for mat in PROBS:
    t0 = time.time()
    S = ik.ipm_with_K(mat)
    row = dict(mat=mat, n_removed=int(len(S["removed"])), p=S["p"])
    for when in ("conv", "final"):
        for tag, uk in (("K", False), ("Kh", True)):
            M, rhs = systems(S, uk, when == "final")
            r = run_minres(M, rhs); r["dim"] = M.shape[0]; r["kappa"] = float(np.linalg.cond(M, 1))
            row[f"{tag}_{when}"] = r
    row["secs"] = round(time.time() - t0, 1)
    res.append(row)
    json.dump(res, open(OUT, "w"), indent=1)
    f = lambda r: f"{r['iters']:5d}{'' if r['converged'] else '*'} (dim {r['dim']}, k {r['kappa']:.0e})"
    print(f"{mat:20s} conv: K {f(row['K_conv'])}  Kh {f(row['Kh_conv'])} | final: K {f(row['K_final'])}  Kh {f(row['Kh_final'])}", flush=True)
print("DONE  (* = no convergió en maxiter)")
