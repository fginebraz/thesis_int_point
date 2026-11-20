# What happens if the heuristic wrongly removes restrictions that are ACTIVE (z_i -> 0, mu_i > 0)?
# For each MM/HS problem: take the method's removed set E and add "false positives" from the
# restrictions Clarabel classifies as active (cls_cl == "z0"), then run the 5 extra iterations with K-hat.
# Scenarios: 1 FP with the largest mu_i (most important), 1 FP with the smallest mu_i, 10% of the active, all active.
# Reference: the same 5 extra iterations with K (no removal).
import sys, json
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point")
import numpy as np
import ipm_khat as ik

D = {d["mat"]: d for d in json.load(open(
    "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/ratio_ipm_vs_clarabel.json"))}
OUT = "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/falsos_positivos"
PROBS = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat", "mm_DUAL1.mat", "mm_DUAL2.mat",
         "mm_DUAL3.mat", "mm_DUAL4.mat", "mm_QPCBLEND.mat"]


def run(S, E):
    S2 = dict(S); S2["removed"] = np.array(sorted(E), dtype=int)
    R = ik.extra_iterations(S2, True, kappa_at=())
    Q, c, A, b, F, d = S["P"]
    viol = float(np.min(F @ R["x"] - d))                         # < 0 means some restriction is violated
    return dict(f=float(R["f"]), kkt=float(R["kkt"]), min_slack=viol,
                min_alpha=float(min(t["alpha"] for t in R["extra"])))


res, lines = [], []
for mat in PROBS:
    S = ik.ipm_with_K(mat)
    E = set(int(i) for i in S["removed"])
    act = [r["i"] for r in D[mat]["rows"] if r["cls_cl"] == "z0"]
    ref = ik.extra_iterations(S, False, kappa_at=())
    row = dict(mat=mat, n_removed=len(E), n_active=len(act), f_ref=float(ref["f"]), kkt_ref=float(ref["kkt"]),
               base=run(S, E), scen={})
    if act:
        mu = S["mu"]
        by_mu = sorted(act, key=lambda i: mu[i])
        scen = {"1 (mu max)": [by_mu[-1]], "1 (mu min)": [by_mu[0]],
                "10%": by_mu[::-1][:max(1, round(0.1 * len(act)))], "todas": act}
        for name, fp in scen.items():
            r = run(S, E | set(fp)); r["n_fp"] = len(fp)
            r["rel_err_f"] = abs(r["f"] - row["f_ref"]) / max(1.0, abs(row["f_ref"]))
            row["scen"][name] = r
    res.append(row)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    lines.append(f"{mat:16s} activas={len(act):3d} | ref: f={row['f_ref']:.8g} kkt={row['kkt_ref']:.1e} | "
                 f"sin FP: kkt={row['base']['kkt']:.1e}")
    for name, r in row["scen"].items():
        lines.append(f"    FP {name:11s} (n={r['n_fp']:3d}): err_rel_f={r['rel_err_f']:.1e} kkt={r['kkt']:.1e} "
                     f"min(Fx-d)={r['min_slack']:+.1e} min_alpha={r['min_alpha']:.1e}")
    print("\n".join(lines[-1 - len(row['scen']):]), flush=True)
open(OUT + ".txt", "w").write("\n".join(lines) + "\nDONE\n")
print("DONE")
