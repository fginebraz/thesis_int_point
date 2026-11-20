"""Per-iteration behaviour of the five detection criteria (thesis numbering 1..5, see criterios_ablacion.py).

For every problem: run the IPM once, record pass/fail of the five criteria for every restriction at every iteration,
and save it (criterios_por_iteracion/<problem>.npz: C[k, i, c] bool, mu0[i], z0[i]) so later analyses need no rerun.

Analysis here ("only culprit"): over all iterations, count the (restriction, iteration) pairs in which exactly ONE
criterion fails, i.e. that criterion alone blocks the marking at that iteration; split by reference class
(mu_i -> 0 vs active). Also reported: pairs where all five pass.
Output: criterios_por_iteracion.json / .txt
"""
import os, sys, json, time
import numpy as np
ROOT = "/Users/fattybaking/Documents/thesis_int_point"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import ipm_khat as ik
import IPM_functions as ipf
import validate_benchmark as vb

OUT = os.path.join(ROOT, "benchmark", "experiments", "criterios_por_iteracion")
os.makedirs(OUT, exist_ok=True)
MM = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat", "mm_DUAL2.mat",
      "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat"]
NL = [f"lp_{x}.mat" for x in ["afiro", "blend", "fit1d", "grow7", "grow15", "grow22", "kb2", "recipe",
                              "scagr7", "scsd1", "scsd6", "sctap1", "share2b", "stair", "stocfor1", "fit1p"]]
REF_MM = {d["mat"]: d for d in json.load(open(os.path.join(ROOT, "benchmark/experiments/ratio_ipm_vs_clarabel.json")))}

REC = []
_orig = ipf.update_active_set_mask


def recorder(mu, z, Q, k, tau, ash, mudf, mu_pct, z_pct, **kw):
    prev = mudf.loc[k - 1].values
    REC.append(np.stack([z_pct > -0.03, mu / (prev + 1e-300) < 0.8, mu * z <= 1e-5, mu < prev, mu < z], axis=1))
    return _orig(mu, z, Q, k, tau, ash, mudf, mu_pct, z_pct, **kw)


ik.update_active_set_mask = recorder

res, lines = [], []
for mat in MM + NL:
    t0 = time.time(); REC.clear()
    S = ik.ipm_with_K(mat)
    if mat.startswith("mm_"):
        cls = np.array([r["cls_cl"] for r in sorted(REF_MM[mat]["rows"], key=lambda r: r["i"])])
        mu0, z0 = cls == "mu0", cls == "z0"
    else:
        Q, c, A, b, F, d = S["P"]
        x_cl, _, _ = ik._quiet(vb.oracle, dict(Q=Q, c=c, A=A, b=b, F=F, d=d))
        mu0 = (F @ x_cl - d) > vb.SLACK_TOL; z0 = ~mu0
    C = np.array(REC)                                   # (iters, p, 5)
    np.savez_compressed(os.path.join(OUT, mat[:-4] + ".npz"), C=C, mu0=mu0, z0=z0, removed=S["removed"])
    nfail = (~C).sum(axis=2)
    row = dict(mat=mat, iters=int(C.shape[0]), p=int(C.shape[1]))
    for grp, g in (("mu0", mu0), ("activas", z0)):
        pairs = int(C.shape[0] * g.sum())
        only = {str(c + 1): int(((nfail == 1) & ~C[:, :, c])[:, g].sum()) for c in range(5)}
        row[grp] = dict(pairs=pairs, all_pass=int((nfail == 0)[:, g].sum()), only=only)
    res.append(row); json.dump(res, open(OUT + ".json", "w"), indent=1)
    s = (f"{mat:18s} it={row['iters']:3d} | mu0: pares={row['mu0']['pairs']:6d} pasan5={row['mu0']['all_pass']:6d} "
         f"solo={row['mu0']['only']} | activas: pares={row['activas']['pairs']:6d} pasan5={row['activas']['all_pass']:5d} "
         f"solo={row['activas']['only']}  ({time.time()-t0:.0f}s)")
    lines.append(s); print(s, flush=True)
open(OUT + ".txt", "w").write("\n".join(lines) + "\nDONE\n"); print("DONE")
