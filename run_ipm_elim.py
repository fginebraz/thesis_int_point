"""
run_ipm_elim.py — IPM with elimination inside the loop (not just after convergence).
"""
import builtins
import numpy as np
import scipy.linalg

from IPM_functions import (load_lp_problem, create_result_dataframes,
                           update_result_dataframes, paso_intpoint,
                           solve_catch_error, update_active_set_mask,
                           active_set_diagnostics, remove_rows_cols_K)

if not hasattr(builtins, "display"):
    builtins.display = print

import pandas as pd


def run_ipm_elim(mat_file, tol=1e-7, kmax=100):
    """IPM main loop (cell 929eddf9) with elimination step (cell 9f83d8b3)
    inserted at the solve point once the heuristic fires."""

    # ── Load problem (identical to cell 929eddf9) ──
    Q, c, A, b, F, d, H = load_lp_problem(mat_file)
    AT = A.T
    FT = F.T

    k = 0
    n = Q.shape[0]
    m = A.shape[0]
    p = F.shape[0]

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
    elim_started_at = None

    while k < kmax:

        # ── Diagonal matrices (cell 929eddf9) ──
        D_z = np.diag(z)
        D_mu = np.diag(mu)

        r_x = Q @ x - AT @ lamda - FT @ mu + c
        r_lamda = - A @ x + b
        r_mu = -F @ x + d + z
        r_z = np.multiply(mu, z)

        F_x = np.concatenate((r_x, r_lamda, r_mu, r_z), 0)
        KKT_norm = np.linalg.norm(F_x, np.inf)

        # ── Build K (cell 929eddf9) ──
        row1 = np.hstack((Q, -AT, -FT @ D_mu))
        row2 = np.hstack((-A, np.zeros((m, m + p))))
        row3 = np.hstack((-D_mu @ F, np.zeros((p, m)), -np.diag(mu * z)))
        K = np.vstack((row1, row2, row3))

        singular_values = np.linalg.svd(K, compute_uv=False)

        (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
         obj_function_df, max_complementarity_df, singular_values_df) = update_result_dataframes(
            k, mu, z, tau,
            mu_percentage_change, z_percentage_change,
            obj_function_value, max_complementarity_value, n, m, p,
            mu_df, z_df, tau_df,
            mu_pct_df, z_pct_df,
            obj_function_df, max_complementarity_df, singular_values_df, singular_values)

        if KKT_norm < tol:
            print("CNPO norm is below the tolerance, stopping iterations.")
            break

        # ── Check if heuristic has detected stable indices ──
        stable_active_indices, regressed_indices = active_set_diagnostics(
            active_set_history, regression_df, p)

        if len(stable_active_indices) > 0 and len(active_set_history) >= 2:
            # ══════════════════════════════════════════════════════════════
            # ELIMINATION PATH (from cell 9f83d8b3)
            # ══════════════════════════════════════════════════════════════
            if elim_started_at is None:
                elim_started_at = k

            K1, D_mu1, L1 = remove_rows_cols_K(
                Q, AT, FT, D_mu, A, F, D_z, mu, z, x, lamda, c, b, d, tau,
                stable_active_indices, r_x, r_lamda)

            delta_vector = solve_catch_error(K1, -L1, k)

            delta_x     = delta_vector[:n]
            delta_lamda = delta_vector[n:n + m]
            Ddelta_mu   = delta_vector[n + m:]

            # Reconstruct delta_mu (cell 9f83d8b3 logic)
            delta_mu_reduced = D_mu1 @ Ddelta_mu
            delta_mu = np.zeros_like(mu)
            active_indices = [i for i in range(p) if i not in stable_active_indices]
            for i, idx in enumerate(active_indices):
                delta_mu[idx] = delta_mu_reduced[i]

            # Reconstruct delta_z (cell 9f83d8b3)
            delta_z = F @ delta_x - (F @ x - d - z)

        else:
            # ══════════════════════════════════════════════════════════════
            # FULL K PATH (cell 929eddf9)
            # ══════════════════════════════════════════════════════════════
            L = np.concatenate((
                r_x,
                r_lamda,
                D_mu @ (d - F @ x) + tau
            ))

            delta_vector = solve_catch_error(K, -L, k)

            delta_x     = delta_vector[:n]
            delta_lamda = delta_vector[n:n + m]
            Ddelta_mu   = delta_vector[n + m:]
            delta_mu = D_mu @ Ddelta_mu

            delta_z = F @ delta_x + F @ x - d - z

        # ── Step size (cell 929eddf9) ──
        alpha_mu = paso_intpoint(mu, delta_mu)
        alpha_z  = paso_intpoint(z, delta_z)
        alpha    = min(alpha_mu, alpha_z)
        alphas.append(alpha)

        if alpha >= 0.9:
            sigma = max(10**-4, sigma / 4)
        if alpha <= 0.75 and sigma < 0.5:
            sigma = min(0.5, sigma * 4)

        # Percentage changes before update (cell 929eddf9)
        mu_percentage_change = alpha * delta_mu / mu
        z_percentage_change  = alpha * delta_z / z

        # Update variables (cell 929eddf9)
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

    return dict(x=x, k=k, elim_started_at=elim_started_at,
                mu=mu, z=z, lamda=lamda, tau=tau)


if __name__ == "__main__":
    import sys
    mat = sys.argv[1] if len(sys.argv) > 1 else "mm_HS35.mat"
    R = run_ipm_elim(mat)
    print(f"Converged in {R['k']} iterations, elimination started at iter {R['elim_started_at']}")
