"""
compare.py — compare the IPM's mu->0 detection against the Clarabel answer key and print the result tables.
"""
import builtins
import numpy as np
import pandas as pd
import scipy.linalg

from IPM_functions import (load_lp_problem, create_result_dataframes,
                           update_result_dataframes, paso_intpoint,
                           solve_catch_error, update_active_set_mask,
                           active_set_diagnostics, remove_rows_cols_K)
from run_ipm_elim import run_ipm_elim

# display() exists in notebooks but not in scripts, so we need this
if not hasattr(builtins, "display"):
    builtins.display = print


# =====================================================================
# IPM MAIN LOOP
# =====================================================================
def run_ipm(mat_file, tol=1e-7, kmax=100, use_z_flatness=True, use_tapia=True):
    """Run the thesis IPM and return the detected mu->0 set plus the iterate."""
    Q, c, A, b, F, d, H = load_lp_problem(mat_file)
    AT = A.T
    FT = F.T

    k = 0
    n = Q.shape[0]
    m = A.shape[0]
    p = F.shape[0]

    # Initial values
    x = np.ones(n)
    lamda = np.zeros(m)
    mu = np.ones(p)
    z = np.ones(p)

    sigma = 0.5
    M_c = np.dot(mu, z) / p
    tau = sigma * M_c

    sigmas = [sigma]
    alphas = []

    obj_function_value = 0.5 * x @ Q @ x + c @ x
    max_complementarity_value = np.max(mu * z)
    mu_percentage_change = np.zeros(p)
    z_percentage_change = np.zeros(p)

    (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
     obj_function_df, max_complementarity_df, active_set_history,
     singular_values_df) = create_result_dataframes(n, m, p)

    regression_df = pd.DataFrame(columns=active_set_history.columns)

    while k < kmax:
        D_z = np.diag(z)
        D_mu = np.diag(mu)

        r_x = Q @ x - AT @ lamda - FT @ mu + c
        r_lamda = - A @ x + b
        r_mu = -F @ x + d + z
        r_z = np.multiply(mu, z)

        F_x = np.concatenate((r_x, r_lamda, r_mu, r_z), 0)
        KKT_norm = np.linalg.norm(F_x, np.inf)

        row1 = np.hstack((Q, -AT, -FT @ D_mu))
        row2 = np.hstack((-A, np.zeros((m, m + p))))
        row3 = np.hstack((-D_mu @ F, np.zeros((p, m)), -np.diag(mu * z)))
        K = np.vstack((row1, row2, row3))
        singular_values = np.linalg.svd(K, compute_uv=False)

        (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
         obj_function_df, max_complementarity_df, singular_values_df) = update_result_dataframes(
            k, mu, z, tau, mu_percentage_change, z_percentage_change,
            obj_function_value, max_complementarity_value, n, m, p,
            mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
            obj_function_df, max_complementarity_df, singular_values_df, singular_values)

        if KKT_norm < tol:
            print("CNPO norm is below the tolerance, stopping iterations.")
            break

        L = np.concatenate((r_x, r_lamda, D_mu @ (d - F @ x) + tau))

        delta_vector = solve_catch_error(K, -L, k)
        delta_x = delta_vector[:n]
        delta_lamda = delta_vector[n:n + m]
        Ddelta_mu = delta_vector[n + m:]
        delta_mu = D_mu @ Ddelta_mu
        delta_z = F @ delta_x + F @ x - d - z

        alpha_mu = paso_intpoint(mu, delta_mu)
        alpha_z = paso_intpoint(z, delta_z)
        alpha = min(alpha_mu, alpha_z)
        alphas.append(alpha)

        if alpha >= 0.9:
            sigma = max(10**-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5:
            sigma = min(0.5, sigma * 4)

        mu_percentage_change = alpha * delta_mu / mu
        z_percentage_change = alpha * delta_z / z

        x += alpha * delta_x
        mu += alpha * delta_mu
        lamda += alpha * delta_lamda
        z += alpha * delta_z

        M_c = np.dot(mu, z) / p
        tau = sigma * M_c
        k += 1
        sigmas.append(sigma)

        obj_function_value = 0.5 * x @ Q @ x + c @ x
        max_complementarity_value = np.max(mu * z)

        active_set_history = update_active_set_mask(
            mu, z, Q, k, tau, active_set_history, mu_df,
            mu_percentage_change, z_percentage_change,
            use_z_flatness=use_z_flatness, use_tapia=use_tapia)

    print("Numero de iteraciones hasta el criterio de paro:", k)

    # The detected mu->0 set (indices the heuristic would eliminate)
    stable_active_indices, regressed_indices = active_set_diagnostics(
        active_set_history, regression_df, p)

    return dict(mat=mat_file, Q=Q, c=c, A=A, b=b, F=F, d=d, x=x, mu=mu, z=z,
                lamda=lamda, tau=tau, n=n, m=m, p=p, k=k,
                detected=stable_active_indices, mu_df=mu_df, z_df=z_df,
                # final-iterate KKT pieces (for conditioning comparison)
                K=K, AT=AT, FT=FT, D_mu=D_mu, D_z=D_z, r_x=r_x, r_lamda=r_lamda)


# =====================================================================
# PART 2: score the IPM's detected set against the independent answer key
# =====================================================================
import numpy as np
import mm_qp_problems as key

# IPM runs the converted mm_*.mat; keep a list of the ones we have.
PROBLEMS = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat",
            "mm_DUAL1.mat", "mm_DUAL2.mat", "mm_DUAL3.mat", "mm_DUAL4.mat",
            "mm_QPCBLEND.mat"]


def score(mat_file):
    """Run everything for one problem and return a dict for all tables."""
    # ── Step 1: run our IPM (full K every iteration, no elimination) ──
    R = run_ipm(mat_file)
    p = R["p"]

    # ── Step 2: solve the same problem with Clarabel (independent answer key) ──
    x_star = key.solve(R["Q"], R["c"], R["A"], R["b"], R["F"], R["d"])
    truth_inactive = key.mu_to_zero(R["F"], R["d"], x_star)   # boolean mask, length p

    # ── Step 3: compare which mu_i we flagged as ->0 vs Clarabel's answer ──
    # Build a boolean array: True at index i if our heuristic said mu_i -> 0
    detected = np.zeros(p, dtype=bool)
    detected[R["detected"]] = True

    n_true = int(truth_inactive.sum())                        # how many mu_i actually -> 0
    true_positives = int((detected & truth_inactive).sum())   # we said ->0 AND it really is
    false_positives = int((detected & ~truth_inactive).sum()) # we said ->0 BUT it's active (mistake)
    recall = 100.0 * true_positives / n_true if n_true else float("nan")  # % of real ->0 we caught
    sol_err = np.linalg.norm(R["x"] - x_star, np.inf)         # how far our x from x*

    # ── Step 4: conditioning — kappa(K) vs kappa(K-hat) ──
    stable = R["detected"]
    K = R["K"]
    kK = np.linalg.cond(K, 1)
    L = np.concatenate((R["r_x"], R["r_lamda"],
                        R["D_mu"] @ (R["d"] - R["F"] @ R["x"]) + R["tau"]))

    if len(stable) > 0:
        K1, _, L1 = remove_rows_cols_K(
            R["Q"], R["AT"], R["FT"], R["D_mu"], R["A"], R["F"], R["D_z"],
            R["mu"], R["z"], R["x"], R["lamda"], R["c"], R["b"], R["d"],
            R["tau"], stable, R["r_x"], R["r_lamda"])
        kK1 = np.linalg.cond(K1, 1)
    else:
        K1, L1 = K, L
        kK1 = kK

    # check if kappa is reliable (sigma_min too small = numerical noise)
    sK = np.linalg.svd(K, compute_uv=False)
    unreliable = sK[-1] < sK[0] * 1e-15 * K.shape[0]

    # ── Step 5: run IPM with elimination in the loop ──
    E = run_ipm_elim(mat_file)
    sol_err_elim = np.linalg.norm(E["x"] - x_star, np.inf)

    # ── Step 6a: run again WITHOUT z-stability (cond3 always True) ──
    R_no_zf = run_ipm(mat_file, use_z_flatness=False)
    detected_no_zf = np.zeros(p, dtype=bool)
    detected_no_zf[R_no_zf["detected"]] = True
    tp_no_zf = int((detected_no_zf & truth_inactive).sum())
    fp_no_zf = int((detected_no_zf & ~truth_inactive).sum())
    recall_no_zf = 100.0 * tp_no_zf / n_true if n_true else float("nan")
    fp_indices_no_zf = np.where(detected_no_zf & ~truth_inactive)[0].tolist()

    # ── Step 6b: run again WITHOUT Tapia (cond5 always True) ──
    R_no_tp = run_ipm(mat_file, use_tapia=False)
    detected_no_tp = np.zeros(p, dtype=bool)
    detected_no_tp[R_no_tp["detected"]] = True
    tp_no_tp = int((detected_no_tp & truth_inactive).sum()) 
    fp_no_tp = int((detected_no_tp & ~truth_inactive).sum())
    recall_no_tp = 100.0 * tp_no_tp / n_true if n_true else float("nan")
    fp_indices_no_tp = np.where(detected_no_tp & ~truth_inactive)[0].tolist()

    # ── Objective values (add back the constant c0 that .mat drops) ──
    clarabel_obj = 0.5 * x_star @ R["Q"] @ x_star + R["c"] @ x_star + key.C0.get(
        mat_file.replace("mm_", "").replace(".mat", ""), 0.0)
    published = key.PUBLISHED.get(mat_file.replace("mm_", "").replace(".mat", ""), None)
    ipm_obj = 0.5 * R["x"] @ R["Q"] @ R["x"] + R["c"] @ R["x"] + key.C0.get(
        mat_file.replace("mm_", "").replace(".mat", ""), 0.0)

    return dict(label=mat_file.replace("mm_", "").replace(".mat", ""),
                n=R["n"], m=R["m"], p=p, k=R["k"],
                published=published, clarabel_obj=clarabel_obj, ipm_obj=ipm_obj,
                n_true=n_true, n_active=int((~truth_inactive).sum()),
                tp=true_positives, fp=false_positives, recall=recall, sol_err=sol_err,
                pct=100.0 * len(stable) / p, kK=kK, kK1=kK1, unreliable=unreliable,
                sol_err_elim=sol_err_elim, k_elim=E["k"],
                elim_at=E["elim_started_at"],
                tp_no_zf=tp_no_zf, fp_no_zf=fp_no_zf,
                recall_no_zf=recall_no_zf, fp_indices_no_zf=fp_indices_no_zf,
                tp_no_tp=tp_no_tp, fp_no_tp=fp_no_tp,
                recall_no_tp=recall_no_tp, fp_indices_no_tp=fp_indices_no_tp)


# =====================================================================
# PRINT ALL TABLES — run score() on every problem, then print results
# =====================================================================
rows = [score(m) for m in PROBLEMS]

# ---- TABLE 1 ----
print("\nTABLE 1: BENCHMARK PROBLEMS")
df1 = pd.DataFrame([{
    "problem": r["label"], "n": r["n"], "m": r["m"], "p": r["p"],
    "published f*": r["published"],
    "source": "Hock-Schittkowski 1981" if r["label"].startswith("HS") else "Maros-Meszaros 1999"
} for r in rows]).set_index("problem")
print(df1)

# ---- TABLE 2 ----
print("\nTABLE 2: CLARABEL ANSWER KEY")
df2 = pd.DataFrame([{
    "problem": r["label"],
    "Clarabel f*": r["clarabel_obj"], "published": r["published"],
    "|diff|": abs(r["clarabel_obj"] - (r["published"] or 0.0)),
    "mu->0": r["n_true"], "active": r["n_active"]
} for r in rows]).set_index("problem")
print(df2)

# ---- TABLE 3 ----
print("\nTABLE 3: IPM DETECTION vs CLARABEL")
df3 = pd.DataFrame([{
    "problem": r["label"], "iter": r["k"],
    "true_mu0": r["n_true"], "detected": r["tp"],
    "sensib": f"{r['recall']:.0f}%", "FP": r["fp"],
    "|x-x*|": r["sol_err"]
} for r in rows]).set_index("problem")
print(df3)

# ---- TABLE 4 ----
print("\nTABLE 4: CONDITIONING")
df4 = pd.DataFrame([{
    "problem": r["label"],
    "%elim": f"{r['pct']:.0f}%",
    "kappa(K)": r["kK"], "kappa(Kh)": r["kK1"],
    "improve": r["kK"] / r["kK1"] if r["kK1"] else float("nan"),
    "note": "unreliable" if r["unreliable"] else ""
} for r in rows]).set_index("problem")
print(df4)

# ---- TABLE 5 ----
print("\nTABLE 5: TAPIA (with vs without)")
df5 = pd.DataFrame([{
    "problem": r["label"],
    "sensib (with)": f"{r['recall']:.0f}%", "FP (with)": r["fp"],
    "sensib (without)": f"{r['recall_no_tp']:.0f}%", "FP (without)": r["fp_no_tp"],
} for r in rows]).set_index("problem")
print(df5)
