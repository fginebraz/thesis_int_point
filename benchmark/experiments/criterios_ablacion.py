"""Ablation of the five detection criteria (thesis numbering):
    1  z_i deja de disminuir      dz/z > -0.03
    2  Tapia                      mu_k / mu_{k-1} < 0.8
    3  complementariedad          mu_i z_i <= 1e-5
    4  mu_i disminuye             mu_k < mu_{k-1}
    5  mu_i < z_i
The IPM trajectory does not depend on the criteria (the heuristic acts after convergence), so we run the IPM once per
problem, record the five pass/fail arrays at every iteration, and evaluate every configuration offline with the
official window (marked in >= 2 of the last 4 iterations).

Reference (mu_i -> 0): MM/HS = Clarabel ratio classification (ratio_ipm_vs_clarabel.json, cls_cl == 'mu0'),
NETLIB = Clarabel slack > 1e-6 (as in the NETLIB pipeline). Error = marked but active (z -> 0) per the reference.

Also: for every mu_i -> 0 restriction NOT marked by the official method, which criteria fail at the last iteration
('ventana' if none fail, i.e. it passes all five only in the last iteration).
Output: criterios_ablacion.json / .txt
"""
import os, sys, json, time, itertools
import numpy as np
ROOT = "/Users/fattybaking/Documents/thesis_int_point"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import ipm_khat as ik
import IPM_functions as ipf
import validate_benchmark as vb

OUT = os.path.join(ROOT, "benchmark", "experiments", "criterios_ablacion")
MM = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat", "mm_DUAL2.mat",
      "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat"]
NL = [f"lp_{x}.mat" for x in ["afiro", "blend", "fit1d", "grow7", "grow15", "grow22", "kb2", "recipe",
                              "scagr7", "scsd1", "scsd6", "sctap1", "share2b", "stair", "stocfor1", "fit1p"]]
CONFIGS = {  # criteria that are switched OFF
    "todos": (), "sin 1": (1,), "sin 2": (2,), "sin 3": (3,), "sin 4": (4,), "sin 5": (5,),
    "sin 1 ni 5 (Tapia sin apoyo)": (1, 5), "sin 2 ni 5 (z sin apoyo)": (2, 5),
    "sin 1, 2 ni 5 (solo 3 y 4)": (1, 2, 5),
}
WINDOW, MIN_HITS = 4, 2
REF_MM = {d["mat"]: d for d in json.load(open(os.path.join(ROOT, "benchmark/experiments/ratio_ipm_vs_clarabel.json")))}

REC = []          # per iteration: dict(k, c1..c5 bool arrays, mu0 bool array)
_orig = ipf.update_active_set_mask


def recorder(mu, z, Q, k, tau, ash, mudf, mu_pct, z_pct, **kw):
    prev = mudf.loc[k - 1].values
    REC.append(dict(k=k, c1=z_pct > -0.03, c2=mu / (prev + 1e-300) < 0.8, c3=mu * z <= 1e-5,
                    c4=mu < prev, c5=mu < z, zero=mu == 0))
    return _orig(mu, z, Q, k, tau, ash, mudf, mu_pct, z_pct, **kw)


ik.update_active_set_mask = recorder


def marked(off):
    last = REC[-WINDOW:]
    hits = 0
    for r in last:
        ok = r["zero"].copy()
        allc = np.ones_like(r["c1"])
        for c in range(1, 6):
            if c not in off:
                allc &= r[f"c{c}"]
        hits = hits + (ok | allc).astype(int)
    return hits >= MIN_HITS


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
    off_mark = marked(())
    assert set(np.where(off_mark)[0]) == set(S["removed"].tolist()), "replay != official"
    row = dict(mat=mat, mu0=int(mu0.sum()), cfg={}, culprit={})
    for name, off in CONFIGS.items():
        M = marked(off)
        row["cfg"][name] = dict(marked=int(M.sum()), ok=int((M & mu0).sum()), err=int((M & z0).sum()),
                                amb=int((M & ~mu0 & ~z0).sum()))
    last = REC[-1]
    for i in np.where(mu0 & ~off_mark)[0]:
        fails = [c for c in range(1, 6) if not last[f"c{c}"][i]]
        key = "ventana" if not fails else "+".join(map(str, fails))
        row["culprit"][key] = row["culprit"].get(key, 0) + 1
    res.append(row); json.dump(res, open(OUT + ".json", "w"), indent=1)
    s = (f"{mat:18s} mu0={row['mu0']:4d} | " +
         " | ".join(f"{k}: {v['ok']}/{v['err']}e" for k, v in row["cfg"].items()) +
         f" | no marcadas: {row['culprit']}  ({time.time()-t0:.0f}s)")
    lines.append(s); print(s, flush=True)
open(OUT + ".txt", "w").write("formato por config: correctas/errores\n" + "\n".join(lines) + "\nDONE\n")
print("DONE")
