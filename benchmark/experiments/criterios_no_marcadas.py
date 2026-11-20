# For the restrictions with mu_i -> 0 (per Clarabel) that the method does NOT mark (MM/HS problems):
# which of the five criteria fail in the last 4 IPM iterations, and by how much.
# Thesis numbering: 1 dz/z > -0.03 | 2 Tapia mu_k/mu_{k-1} < 0.8 | 3 mu*z <= 1e-5 | 4 mu decreases | 5 mu < z
import sys, json
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point")
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point/benchmark/figures")
import numpy as np
import make_figures as mf
import validate_benchmark as vb

OUT = "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/criterios_no_marcadas.json"
D = {d["mat"]: d for d in json.load(open(
    "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/ratio_ipm_vs_clarabel.json"))}


def criteria(MU, Z, k):
    """Values and pass/fail of the five criteria at iteration k (k >= 1)."""
    mu, z, pmu, pz = MU[k], Z[k], MU[k - 1], Z[k - 1]
    v = dict(c1=(z - pz) / pz, c2=mu / pmu, c3=mu * z, c4=mu - pmu, c5=mu / z)
    ok = dict(c1=v["c1"] > -0.03, c2=v["c2"] < 0.8, c3=v["c3"] <= 1e-5, c4=v["c4"] < 0, c5=v["c5"] < 1)
    return v, ok


res, mismatch = [], 0
for mat, _ in vb.CORE:
    MU, Z, flags, E = mf.ipm_history(mat)
    K = MU.shape[0] - 1
    # sanity: recomputed criteria reproduce the code's flags
    for k in range(1, K + 1):
        _, ok = criteria(MU, Z, k)
        allok = ok["c1"] & ok["c2"] & ok["c3"] & ok["c4"] & ok["c5"]
        mismatch += int(np.sum(allok.astype(int) != (flags[k] > 0)))
    missed = [r["i"] for r in D[mat]["rows"] if r["cls_cl"] == "mu0" and not r["removed"]]
    for i in missed:
        last = []
        for k in range(max(1, K - 3), K + 1):
            v, ok = criteria(MU, Z, k)
            last.append(dict(k=k, **{f"v{c}": float(v[c][i]) for c in v}, **{f"ok{c}": bool(ok[c][i]) for c in ok}))
        res.append(dict(mat=mat, i=i, hits=int(sum(all(l[f"okc{j}"] for j in range(1, 6)) for l in last)), last=last))

json.dump(dict(mismatch=mismatch, rows=res), open(OUT, "w"), indent=1)
print("flags mismatch (should be 0):", mismatch)
print("missed:", len(res))
names = {1: "1 dz/z>-0.03", 2: "2 Tapia<0.8", 3: "3 mu*z<=1e-5", 4: "4 mu baja", 5: "5 mu<z"}
for c in range(1, 6):
    n_fail_last = sum(not r["last"][-1][f"okc{c}"] for r in res)
    n_fail_any = sum(any(not l[f"okc{c}"] for l in r["last"]) for r in res)
    print(f"  {names[c]:14s}: falla en la última iteración {n_fail_last:2d}/{len(res)}, en alguna de las últimas 4 {n_fail_any:2d}/{len(res)}")
for r in res:
    L = r["last"][-1]
    fails = [str(c) for c in range(1, 6) if not L[f"okc{c}"]]
    print(f"  {r['mat']:16s} i={r['i']:3d} hits={r['hits']}  falla(última)={','.join(fails) or '-'}  "
          f"dz/z={L['vc1']:+.3f} tapia={L['vc2']:.2f} muz={L['vc3']:.1e} mu/z={L['vc5']:.2f}")
