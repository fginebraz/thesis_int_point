# How risky is it to drop the stability window? Compare the official window (2 of last 4) with
# (1 of last 1) = "passes all five criteria at the last iteration" and (1 of last 4).
# Errors = removed restrictions that are NOT inactive per Clarabel. Also KKT after 5 extra iterations with K-hat.
import sys, json, time
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point")
import numpy as np
import ipm_khat as ik
import validate_benchmark as vb

MM = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat", "mm_DUAL2.mat",
      "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat"]
NL = [f"lp_{x}.mat" for x in ["afiro", "blend", "fit1d", "fit1p", "grow7", "grow15", "grow22", "kb2", "recipe",
                              "scagr7", "scsd1", "scsd6", "sctap1", "sctap2", "sctap3", "share2b", "stair", "stocfor1"]]
CONFIGS = {"2 de 4 (oficial)": (4, 2), "1 de 1 (sin ventana)": (1, 1), "1 de 4": (4, 1)}
OUT = "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/ventana"
res, lines = [], []
for mat in MM + NL:
    t0 = time.time(); row = dict(mat=mat, cfg={})
    for name, (w, h) in CONFIGS.items():
        ik.WINDOW, ik.MIN_HITS = w, h
        S = ik.ipm_with_K(mat)
        if "inactive" not in row:
            Q, c, A, b, F, d = S["P"]
            x_cl, _, _ = ik._quiet(vb.oracle, dict(Q=Q, c=c, A=A, b=b, F=F, d=d))
            inact = (F @ x_cl - d) > vb.SLACK_TOL
            row["inactive"] = int(inact.sum())
        E = S["removed"]
        err = int((~inact[E]).sum()) if len(E) else 0
        kkt = float(ik.extra_iterations(S, True, kappa_at=())["kkt"]) if len(E) else None
        row["cfg"][name] = dict(removed=int(len(E)), err=err, kkt_khat=kkt)
    ik.WINDOW, ik.MIN_HITS = 4, 2
    res.append(row); json.dump(res, open(OUT + ".json", "w"), indent=1)
    s = f"{mat:20s} mu->0={row['inactive']:5d} | " + " | ".join(
        f"{k}: {v['removed']:5d} elim, {v['err']} err, kkt {v['kkt_khat'] if v['kkt_khat'] is None else format(v['kkt_khat'], '.0e')}"
        for k, v in row["cfg"].items()) + f"  ({time.time()-t0:.0f}s)"
    lines.append(s); print(s, flush=True)
open(OUT + ".txt", "w").write("\n".join(lines) + "\nDONE\n"); print("DONE")
