# Ratios z_i/mu_i at our IPM's convergence vs Clarabel's z*/mu* (MM/HS problems).
import sys, json
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point")
import numpy as np
import ipm_khat as ik
import validate_benchmark as vb

PROBS = [m for m, _ in vb.CORE]
OUT = "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/ratio_ipm_vs_clarabel.json"


def cls(r):
    return "mu0" if r > 100 else ("z0" if r < 1 / 100 else "amb")


res = []
for mat in PROBS:
    S = ik.ipm_with_K(mat)
    Q, c, A, b, F, d = S["P"]
    x_cl, mu_cl, _ = ik._quiet(vb.oracle, dict(Q=Q, c=c, A=A, b=b, F=F, d=d))
    z_cl = F @ x_cl - d
    r_cl = np.maximum(z_cl, 0) / np.maximum(mu_cl, 1e-300)
    r_ipm = S["z"] / S["mu"]
    E = set(S["removed"].tolist())
    rows = []
    for i in range(S["p"]):
        rows.append(dict(i=i, r_ipm=float(r_ipm[i]), r_cl=float(r_cl[i]), cls_ipm=cls(r_ipm[i]), cls_cl=cls(r_cl[i]),
                         removed=i in E, mu=float(S["mu"][i]), z=float(S["z"][i])))
    res.append(dict(mat=mat, k=S["k"], rows=rows))
    json.dump(res, open(OUT, "w"), indent=1)

    cl_mu0 = [r for r in rows if r["cls_cl"] == "mu0"]
    cl_z0 = [r for r in rows if r["cls_cl"] == "z0"]
    agree = sum(r["cls_ipm"] == r["cls_cl"] for r in rows)
    print(f"\n{mat}  p={S['p']}  agree {agree}/{S['p']}  "
          f"Clarabel mu0={len(cl_mu0)} z0={len(cl_z0)} amb={S['p']-len(cl_mu0)-len(cl_z0)}  removed={len(E)}")
    if cl_mu0:
        print(f"  Clarabel mu0 -> IPM ratio min {min(r['r_ipm'] for r in cl_mu0):.1e} / median {np.median([r['r_ipm'] for r in cl_mu0]):.1e}")
    if cl_z0:
        print(f"  Clarabel z0  -> IPM ratio max {max(r['r_ipm'] for r in cl_z0):.1e}")
    for r in rows:
        if r["cls_ipm"] != r["cls_cl"] or (r["cls_cl"] == "mu0" and not r["removed"]):
            print(f"    i={r['i']:3d} IPM {r['r_ipm']:.1e} ({r['cls_ipm']})  Clarabel {r['r_cl']:.1e} ({r['cls_cl']})  "
                  f"removed={r['removed']}  mu={r['mu']:.1e} z={r['z']:.1e}")
print("\nDONE")
