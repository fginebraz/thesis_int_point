# Experiment: K vs K-hat with removed mu_i = 0, for 20 extra iterations. Tracks each KKT component.
import sys, json
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments")
import numpy as np, scipy.linalg
import mu_variants as mv
from mu_zero_effect_defs import PUB, C0, PROBS

def comps(P, x, l, mu, z):
    rx, rl, rm = mv.resid(P, x, l, mu, z)
    n = lambda v: float(np.linalg.norm(v, np.inf))
    return dict(rx=n(rx), rl=n(rl), rm=n(rm), comp=n(mu * z))

def run(P, S, variant, iters):
    Q, c, A, b, F, d = P
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]
    x, l, mu, z = S["x"].copy(), S["l"].copy(), S["mu"].copy(), S["z"].copy()
    sigma, tau = S["sigma"], S["tau"]
    elim = [] if variant == "K" else S["stable"]
    keep = [i for i in range(p) if i not in set(elim)]; E = np.array(elim, dtype=int)
    if variant == "zero" and len(E): mu[E] = 0.0
    tr = []
    for _ in range(iters):
        rx, rl, rm = mv.resid(P, x, l, mu, z)
        K1, D_mu1, L1 = mv.q(mv.remove_rows_cols_K, Q, A.T, F.T, np.diag(mu), A, F, np.diag(z), mu, z, x, l,
                             c, b, d, tau, elim, rx, rl)
        kap = float(np.linalg.cond(K1, 1)) if K1.shape[0] <= 1500 else None
        try:
            dv = scipy.linalg.solve(K1, -L1)
        except Exception as e:
            tr.append(dict(error=repr(e)[:80])); break
        dx, dl = dv[:n], dv[n:n + m]
        dmu = np.zeros(p); dmu[keep] = D_mu1 @ dv[n + m:]
        dz = F @ dx + F @ x - d - z
        alpha = min(mv.paso_intpoint(mu[keep], dmu[keep]), mv.paso_intpoint(z, dz))
        if alpha >= 0.9: sigma = max(1e-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5: sigma = min(0.5, sigma * 4)
        x = x + alpha * dx; mu = mu + alpha * dmu; l = l + alpha * dl; z = z + alpha * dz
        tau = sigma * (mu[keep] @ z[keep]) / max(len(keep), 1)
        r = comps(P, x, l, mu, z); r.update(kappa=kap, alpha=float(alpha), f=float(0.5 * x @ Q @ x + c @ x),
                                            nan=bool(np.isnan(x).any()))
        tr.append(r)
    return tr

out = []
for mat in PROBS:
    P = mv.load(mat); tol = 1e-6 if mat.startswith("lp_") else 1e-7
    S = mv.phase1(P, tol)
    ref = (PUB[mat] - C0.get(mat, 0.0)) if mat in PUB else mv.clarabel_f(P)
    row = dict(mat=mat, ref=ref, conv=comps(P, S["x"], S["l"], S["mu"], S["z"]),
               K=run(P, S, "K", 20), zero=run(P, S, "zero", 20))
    out.append(row); print(mat, "done", flush=True)
    json.dump(out, open("mu_zero_long.json", "w"), indent=1)
