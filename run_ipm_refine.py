"""
run_ipm_refine.py
=================
Run the thesis IPM to convergence with full K, then do 5 extra iterations
with K̂ (eliminated rows/cols for μ→0 indices) to recover precision.

Based on the main loop in intpoint_tryout.ipynb (cell 929eddf9) and the
post-convergence step (cell 9f83d8b3).

Run:  .venv/bin/python run_ipm_refine.py
"""
import numpy as np
import pandas as pd
import scipy.linalg

from IPM_functions import (load_lp_problem, create_result_dataframes,
                           update_result_dataframes, paso_intpoint,
                           solve_catch_error, update_active_set_mask,
                           active_set_diagnostics, remove_rows_cols_K)

PROBLEMS = [
    "mm_TAME.mat", "mm_HS21.mat", "mm_HS35.mat", "mm_HS53.mat",
    "mm_DUAL1.mat", "mm_DUAL2.mat", "mm_DUAL3.mat", "mm_DUAL4.mat",
    "mm_QPCBLEND.mat",
]

# Objective constant that the .mat format drops
C0 = {"HS21": -100.0, "HS35": 9.0, "HS53": 6.0}


def run_ipm(mat_file, tol=1e-7, kmax=100, max_refine=20, stall_patience=2, stall_ratio=1.5):
    """Run the IPM to convergence, then do `refine_iters` extra iterations with K̂."""
    Q, c, A, b, F, d, H = load_lp_problem(mat_file)
    AT, FT = A.T, F.T
    n, m, p = Q.shape[0], A.shape[0], F.shape[0]

    x = np.ones(n)
    lamda = np.zeros(m)
    mu = np.ones(p)
    z = np.ones(p)

    sigma = 0.5
    M_c = np.dot(mu, z) / p
    tau = sigma * M_c

    mu_percentage_change = np.zeros(p)
    z_percentage_change = np.zeros(p)

    (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
     obj_function_df, max_complementarity_df,
     active_set_history, singular_values_df) = create_result_dataframes(n, m, p)
    regression_df = pd.DataFrame(columns=active_set_history.columns)

    obj_function_value = 0.5 * x @ Q @ x + c @ x
    max_complementarity_value = np.max(mu * z)

    # ── Phase 1: full K to convergence ──
    k = 0
    while k < kmax:
        D_z = np.diag(z)
        D_mu = np.diag(mu)

        r_x = Q @ x - AT @ lamda - FT @ mu + c
        r_lamda = -A @ x + b
        r_mu = -F @ x + d + z
        r_z = np.multiply(mu, z)

        F_x = np.concatenate((r_x, r_lamda, r_mu, r_z))
        KKT_norm = np.linalg.norm(F_x, np.inf)

        row1 = np.hstack((Q, -AT, -FT @ D_mu))
        row2 = np.hstack((-A, np.zeros((m, m + p))))
        row3 = np.hstack((-D_mu @ F, np.zeros((p, m)), -np.diag(mu * z)))
        K = np.vstack((row1, row2, row3))
        singular_values = np.linalg.svd(K, compute_uv=False)

        (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
         obj_function_df, max_complementarity_df,
         singular_values_df) = update_result_dataframes(
            k, mu, z, tau, mu_percentage_change, z_percentage_change,
            obj_function_value, max_complementarity_value, n, m, p,
            mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
            obj_function_df, max_complementarity_df,
            singular_values_df, singular_values)

        if KKT_norm < tol:
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

        if alpha >= 0.9:
            sigma = max(1e-4, sigma / 4)
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

        obj_function_value = 0.5 * x @ Q @ x + c @ x
        max_complementarity_value = np.max(mu * z)

        active_set_history = update_active_set_mask(
            mu, z, Q, k, tau, active_set_history, mu_df,
            mu_percentage_change, z_percentage_change)

    k_converged = k
    obj_K = 0.5 * x @ Q @ x + c @ x
    kkt_K = KKT_norm

    # ── Detect μ→0 indices ──
    stable_active_indices, _ = active_set_diagnostics(
        active_set_history, regression_df, p)

    if len(stable_active_indices) == 0 or len(stable_active_indices) == p:
        return dict(
            label=mat_file, k=k_converged,
            obj_K=obj_K, kkt_K=kkt_K,
            n_elim=len(stable_active_indices), p=p,
            refine_kkt=[], refine_obj=[], n_refine=0,
            obj_final=obj_K, kkt_final=kkt_K)

    # ── Phase 2: refine with K̂ (tolerance-driven) ──
    active_indices = [i for i in range(p) if i not in stable_active_indices]
    refine_kkt = []
    refine_obj = []
    prev_kkt_norm = kkt_K
    stall_count = 0

    for r in range(max_refine):
        D_z = np.diag(z)
        D_mu = np.diag(mu)

        r_x = Q @ x - AT @ lamda - FT @ mu + c
        r_lamda = -A @ x + b
        r_mu = -F @ x + d + z
        r_z = np.multiply(mu, z)

        F_x = np.concatenate((r_x, r_lamda, r_mu, r_z))
        KKT_norm = np.linalg.norm(F_x, np.inf)
        refine_kkt.append(KKT_norm)
        refine_obj.append(0.5 * x @ Q @ x + c @ x)

        # Tolerance-driven stopping (commented out; using fixed iterations for now)
        # if prev_kkt_norm / KKT_norm < stall_ratio if KKT_norm > 0 else False:
        #     stall_count += 1
        # else:
        #     stall_count = 0
        # if stall_count >= stall_patience:
        #     break
        # prev_kkt_norm = KKT_norm

        K1, D_mu1, L1 = remove_rows_cols_K(
            Q, AT, FT, D_mu, A, F, D_z, mu, z, x, lamda,
            c, b, d, tau, stable_active_indices, r_x, r_lamda)

        delta_vector = scipy.linalg.solve(K1, -L1)
        delta_x = delta_vector[:n]
        delta_lamda = delta_vector[n:n + m]
        Ddelta_mu = delta_vector[n + m:]

        delta_mu_reduced = D_mu1 @ Ddelta_mu
        delta_mu = np.zeros_like(mu)
        for i, idx in enumerate(active_indices):
            delta_mu[idx] = delta_mu_reduced[i]

        delta_z = F @ delta_x + F @ x - d - z

        alpha_mu = paso_intpoint(mu, delta_mu)
        alpha_z = paso_intpoint(z, delta_z)
        alpha = min(alpha_mu, alpha_z)

        if alpha >= 0.9:
            sigma = max(1e-4, sigma / 4)
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

    obj_final = 0.5 * x @ Q @ x + c @ x
    r_x = Q @ x - AT @ lamda - FT @ mu + c
    r_lamda = -A @ x + b
    r_mu = -F @ x + d + z
    r_z = np.multiply(mu, z)
    kkt_final = np.linalg.norm(np.concatenate((r_x, r_lamda, r_mu, r_z)), np.inf)

    n_refine = len(refine_kkt)

    return dict(
        label=mat_file, k=k_converged,
        obj_K=obj_K, kkt_K=kkt_K,
        n_elim=len(stable_active_indices), p=p,
        refine_kkt=refine_kkt, refine_obj=refine_obj,
        n_refine=n_refine,
        obj_final=obj_final, kkt_final=kkt_final)


if __name__ == "__main__":
    print(f"{'problem':10s} {'k':>3} {'elim':>6} {'ref':>4} "
          f"{'KKT(K)':>10} {'KKT(K̂)':>10} "
          f"{'f*(K)':>16} {'f*(K̂)':>16}")
    print("-" * 90)

    for mat in PROBLEMS:
        name = mat.replace("mm_", "").replace(".mat", "")
        c0 = C0.get(name, 0.0)
        R = run_ipm(mat)

        print(f"{name:10s} {R['k']:>3} {R['n_elim']:>3}/{R['p']:<3} {R['n_refine']:>4}"
              f" {R['kkt_K']:>10.2e} {R['kkt_final']:>10.2e}"
              f" {R['obj_K'] + c0:>16.10f} {R['obj_final'] + c0:>16.10f}")

        if R['refine_kkt']:
            prev = R['kkt_K']
            for i, (kkt, obj) in enumerate(zip(R['refine_kkt'], R['refine_obj'])):
                ratio = prev / kkt if kkt > 0 else float('inf')
                print(f"{'':10s}       ref {i+1}: KKT={kkt:.2e}  f*={obj + c0:.12f}  ({ratio:.1f}x)")
                prev = kkt
