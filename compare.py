"""
compare.py
==========
Compare the IPM's mu->0 detection heuristic against the independent answer key
in mm_qp_problems.py.

The IPM main loop below is COPIED VERBATIM from intpoint_tryout.ipynb.  The
heuristic itself (update_active_set_mask and active_set_diagnostics) is IMPORTED
from IPM_functions.py -- the single source of truth -- so changing the detection
criteria there updates both the notebook and this comparison at once.  Everything
else (loader, dataframes, step size, linear solve) is imported unchanged too.

Part 1: load a problem, run the IPM, print the detected mu->0 set.
Part 2: score the detected set against the mm_qp_problems.py answer key.
"""
import builtins
import numpy as np
import pandas as pd
import scipy.linalg

from IPM_functions import (load_lp_problem, create_result_dataframes,
                           update_result_dataframes, paso_intpoint,
                           solve_catch_error, update_active_set_mask,
                           active_set_diagnostics)

# active_set_diagnostics calls display() (a notebook builtin) in its regression
# branch; expose a plain fallback in builtins so IPM_functions resolves it when
# imported into a script.
if not hasattr(builtins, "display"):
    builtins.display = print


# =====================================================================
# IPM MAIN LOOP (copied verbatim from intpoint_tryout.ipynb, cell 929eddf9)
# =====================================================================
def run_ipm(mat_file, tol=1e-7, kmax=100):
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
            mu_percentage_change, z_percentage_change)

    print("Numero de iteraciones hasta el criterio de paro:", k)

    # The detected mu->0 set (indices the heuristic would eliminate)
    stable_active_indices, regressed_indices = active_set_diagnostics(
        active_set_history, regression_df, p)

    return dict(mat=mat_file, Q=Q, c=c, A=A, b=b, F=F, d=d, x=x, mu=mu, z=z,
                n=n, m=m, p=p, k=k, detected=stable_active_indices,
                mu_df=mu_df, z_df=z_df)   # per-iteration histories (for indicators)


# =====================================================================
# PART 2: score the IPM's detected set against the independent answer key
# =====================================================================
import contextlib, io
import numpy as np
import mm_qp_problems as key   # the answer-key module (Clarabel)

# IPM runs the converted mm_*.mat; keep a list of the ones we have.
PROBLEMS = ["mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat",
            "mm_DUAL1.mat", "mm_DUAL2.mat", "mm_DUAL3.mat", "mm_DUAL4.mat",
            "mm_QPCBLEND.mat"]


def score(mat_file):
    """Run the IPM, get the true mu->0 set from Clarabel on the SAME matrices,
    and compare index-by-index."""
    with contextlib.redirect_stdout(io.StringIO()):      # hush the chatty loop
        R = run_ipm(mat_file)
    p = R["p"]

    # answer key: solve the IPM's own problem, then inactive = positive slack
    x_star = key.solve(R["Q"], R["c"], R["A"], R["b"], R["F"], R["d"])
    truth_inactive = key.mu_to_zero(R["F"], R["d"], x_star)   # boolean mask, length p

    detected = np.zeros(p, dtype=bool)
    detected[R["detected"]] = True

    n_true = int(truth_inactive.sum())
    tp = int((detected & truth_inactive).sum())          # correctly flagged mu->0
    fp = int((detected & ~truth_inactive).sum())         # WRONGLY flagged (active!)
    recall = 100.0 * tp / n_true if n_true else float("nan")
    sol_err = np.linalg.norm(R["x"] - x_star, np.inf)    # did the IPM reach x*?
    return dict(label=mat_file.replace("mm_", "").replace(".mat", ""),
                p=p, k=R["k"], n_true=n_true, tp=tp, fp=fp,
                recall=recall, sol_err=sol_err)


if __name__ == "__main__":
    rows = [score(m) for m in PROBLEMS]
    print(f"\n{'problem':10s} {'p':>4} {'it':>3} {'true_mu0':>8} {'detected':>8} "
          f"{'recall':>7} {'false+':>6} {'|x-x*|':>9}")
    print("-" * 62)
    for r in rows:
        print(f"{r['label']:10s} {r['p']:>4} {r['k']:>3} {r['n_true']:>8} "
              f"{r['tp']:>8} {r['recall']:>6.0f}% {r['fp']:>6} {r['sol_err']:>9.1e}")
    print("\nrecall = detected mu->0 / true mu->0 ;  false+ = active constraints "
          "wrongly flagged (want 0)")
