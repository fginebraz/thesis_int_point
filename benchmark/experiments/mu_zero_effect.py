# Experiment (does not modify the thesis code): what changes when removed mu_i are set to 0.
import sys, json
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments")
import numpy as np, scipy.linalg
import mu_variants as mv

PUB = {"mm_TAME.mat": 0.0, "mm_HS21.mat": -99.96, "mm_HS35.mat": 1/9, "mm_HS53.mat": 176/43,
       "mm_DUAL1.mat": 0.0350129660, "mm_DUAL2.mat": 0.0337336760, "mm_DUAL3.mat": 0.1357558400,
       "mm_DUAL4.mat": 0.7460908400, "mm_QPCBLEND.mat": -0.0078425409}
C0 = {"mm_HS21.mat": -100.0, "mm_HS35.mat": 9.0, "mm_HS53.mat": 6.0}
PROBS = list(PUB) + ["lp_afiro.mat", "lp_kb2.mat", "lp_blend.mat", "lp_recipe.mat", "lp_share2b.mat", "lp_grow7.mat"]

def run(P, S, variant, iters=5):
    Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    x, l, mu, z = S["x"].copy(), S["l"].copy(), S["mu"].copy(), S["z"].copy()
    sigma, tau = S["sigma"], S["tau"]
    elim = [] if variant == "K" else S["stable"]
    keep = [i for i in range(p) if i not in set(elim)]; E = np.array(elim, dtype=int)
    info = dict(mu_removed_max=float(mu[E].max()) if len(E) else 0.0,
                rx_contrib=float(np.linalg.norm(F[E].T @ mu[E], np.inf)) if len(E) else 0.0)
    if variant == "zero" and len(E): mu[E] = 0.0
    traj, kap = [], []
    for _ in range(iters):
        rx, rl, rm = mv.resid(P, x, l, mu, z)
        K1, D_mu1, L1 = mv.q(mv.remove_rows_cols_K, Q, A.T, F.T, np.diag(mu), A, F, np.diag(z), mu, z, x, l,
                             c, b, d, tau, elim, rx, rl)
        if K1.shape[0] <= 1500: kap.append(float(np.linalg.cond(K1, 1)))
        dv = scipy.linalg.solve(K1, -L1)
        dx, dl = dv[:n], dv[n:n + m]
        dmu = np.zeros(p); dmu[keep] = D_mu1 @ dv[n + m:]
        dz = F @ dx + F @ x - d - z
        alpha = min(mv.paso_intpoint(mu[keep], dmu[keep]), mv.paso_intpoint(z, dz))
        if alpha >= 0.9: sigma = max(1e-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5: sigma = min(0.5, sigma * 4)
        x = x + alpha * dx; mu = mu + alpha * dmu; l = l + alpha * dl; z = z + alpha * dz
        tau = sigma * (mu[keep] @ z[keep]) / max(len(keep), 1)
        traj.append(float(mv.kkt(P, x, l, mu, z)))
    info.update(kkt=traj, f=float(0.5 * x @ Q @ x + c @ x), kappa_max=max(kap) if kap else None, x=x)
    return info

out = []
for mat in PROBS:
    P = mv.load(mat); tol = 1e-6 if mat.startswith("lp_") else 1e-7
    S = mv.phase1(P, tol)
    ref = (PUB[mat] - C0.get(mat, 0.0)) if mat in PUB else mv.clarabel_f(P)
    R = {v: run(P, S, v) for v in ["K", "freeze", "zero"]}
    row = dict(mat=mat, k=S["k"], p=P[4].shape[0], n_elim=len(S["stable"]),
               kkt_conv=float(mv.kkt(P, S["x"], S["l"], S["mu"], S["z"])),
               ref=ref + C0.get(mat, 0.0), ref_is="publicado" if mat in PUB else "Clarabel",
               mu_removed_max=R["freeze"]["mu_removed_max"], rx_contrib=R["freeze"]["rx_contrib"],
               x_diff_freeze_zero=float(np.linalg.norm(R["freeze"]["x"] - R["zero"]["x"], np.inf)))
    for v in R:
        row[v] = dict(kkt=R[v]["kkt"], f=R[v]["f"] + C0.get(mat, 0.0), ferr=abs(R[v]["f"] - ref), kappa_max=R[v]["kappa_max"])
    out.append(row); print(mat, "done", flush=True)
json.dump(out, open("/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/mu_zero_effect.json", "w"), indent=1)
