# NETLIB with the official method (ipm_khat.py): K vs K-hat (mu_i = 0), 5 extra iterations.
import sys, json, time
sys.path.insert(0, "/Users/fattybaking/Documents/thesis_int_point")
import numpy as np
import ipm_khat as ik
import validate_benchmark as vb

PROBS = ["afiro", "blend", "fit1d", "fit1p", "grow7", "grow15", "grow22", "kb2", "recipe", "scagr7",
         "scsd1", "scsd6", "sctap1", "sctap2", "sctap3", "share2b", "stair", "stocfor1",
         "kleeminty_m3", "kleeminty_m5"]
OUT = "/Users/fattybaking/Documents/thesis_int_point/benchmark/experiments/netlib_official.json"
res = []
for name in PROBS:
    t0 = time.time()
    S = ik.ipm_with_K(f"lp_{name}.mat")
    Q, c, A, b, F, d = S["P"]
    x_cl, mu_cl, _ = ik._quiet(vb.oracle, dict(Q=Q, c=c, A=A, b=b, F=F, d=d))
    f_cl = float(0.5 * x_cl @ Q @ x_cl + c @ x_cl)
    inactive = (F @ x_cl - d) > vb.SLACK_TOL
    E = S["removed"]
    RK = ik.extra_iterations(S, use_khat=False)
    RH = ik.extra_iterations(S, use_khat=True) if len(E) else RK
    row = dict(name=name.upper(), n=S["n"], m=S["m"], p=S["p"], k=S["k"],
               n_removed=int(len(E)), n_inactive=int(inactive.sum()), n_err=int((~inactive[E]).sum()) if len(E) else 0,
               mu_removed_max=S["mu_removed_max"], kkt_conv=float(S["kkt_conv"]), f_clarabel=f_cl,
               x_err_K=float(np.linalg.norm(RK["x"] - x_cl, np.inf)), x_err_Kh=float(np.linalg.norm(RH["x"] - x_cl, np.inf)))
    for tag, R in (("K", RK), ("Kh", RH)):
        row[tag] = dict(kkt=float(R["kkt"]), f=float(R["f"]), ferr_rel=float(abs(R["f"] - f_cl) / max(1, abs(f_cl))),
                        kappa1=R["extra"][0]["kappa"], kappa5=R["extra"][4]["kappa"],
                        kkt_traj=[t["kkt"] for t in R["extra"]], alpha=[t["alpha"] for t in R["extra"]])
    row["secs"] = round(time.time() - t0, 1)
    res.append(row)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{row['name']:13s} {row['n_removed']}/{row['n_inactive']} err {row['n_err']}  "
          f"kK {row['K']['kappa1']:.1e}->{row['K']['kappa5']:.1e}  kKh {row['Kh']['kappa1']:.1e}->{row['Kh']['kappa5']:.1e}  "
          f"KKT K {row['K']['kkt']:.1e} Kh {row['Kh']['kkt']:.1e}  ferr {row['K']['ferr_rel']:.1e}/{row['Kh']['ferr_rel']:.1e}  {row['secs']}s", flush=True)
print("DONE", flush=True)
