import numpy as np
import pandas as pd
import copy as copy
import scipy
import scipy.io
import time
import os
from scipy.linalg import solve, LinAlgWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import random

from matplotlib.animation import FuncAnimation, PillowWriter
import openpyxl
import xlsxwriter

def loadProblem(fname, useSparse=False):
    mat = scipy.io.loadmat(fname)

    if useSparse:
        A = mat.get('Problem')[0,0][2].astype(float)
    else:
        A = mat.get('Problem')[0,0][2].astype(float).toarray()

    b = mat.get('Problem')[0,0][3].astype(float)[:,0]
    aux = mat.get('Problem')[0,0][5]
    c = aux[0,0][0].astype(float)[:,0]

    print('Norma infinita de b: ', np.linalg.norm(b, np.inf))

    return {
        'A': A,
        'bE': b,
        'c': c,
        'AE': A  # ← check if this is correct in your structure
    }

def create_result_dataframes(n,m,p):
    mu_df = pd.DataFrame(columns=range(p))
    z_df = pd.DataFrame(columns=range(p))
    tau_df = pd.DataFrame(columns=range(p))
    mu_pct_df = pd.DataFrame(columns=range(p))    
    z_pct_df  = pd.DataFrame(columns=range(p))     
    obj_function_df = pd.DataFrame(columns=['Objective Function Value'])
    max_complementarity_df = pd.DataFrame(columns=['Maximum complementarity value: max_i (mu_i * z_i)'])
    active_set_history = pd.DataFrame(columns=range(p))
    active_set_history.index.name = "Iteration"
    mu_pct_df.index.name = "Iteration"             
    z_pct_df.index.name  = "Iteration"             
    singular_values_df = pd.DataFrame(columns=range(n+m+p))

    return (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
            obj_function_df, max_complementarity_df, active_set_history, singular_values_df)

def update_result_dataframes(k, mu, z, tau,
                             mu_percentage_change, z_percentage_change,   
                             obj_function_value, max_complementarity_value, n,m,p,
                             mu_df, z_df, tau_df,
                             mu_pct_df, z_pct_df,                          
                             obj_function_df, max_complementarity_df, singular_values_df,singular_values):
    mu_df.loc[k] = mu
    z_df.loc[k]  = z
    tau_df.loc[k] = np.full(p, tau)
    mu_pct_df.loc[k] = mu_percentage_change      
    z_pct_df.loc[k]  = z_percentage_change       
    obj_function_df.loc[k] = obj_function_value
    max_complementarity_df.loc[k] = max_complementarity_value
    singular_values_df.loc[k] = singular_values

    return (mu_df, z_df, tau_df, mu_pct_df, z_pct_df,
            obj_function_df, max_complementarity_df, singular_values_df)

def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: lightgreen' if is_max.any() else '' for v in is_max]

def paso_intpoint(mu, delta_mu):
    eps = 0.01
    alphas = np.ones( len(mu) +1 ) # this way the last entry remains one.
    for i in range( len(mu) ):
        if delta_mu[i] < 0:
            alphas[i] = -(1-eps)*mu[i]/delta_mu[i]

    alpha = min(alphas)
    return alpha

def solve_catch_error(K,ld,k):
    with warnings.catch_warnings(record=True) as warneo:
        warnings.simplefilter("always", LinAlgWarning)
        # Solve the linear system
        delta_vector = scipy.linalg.solve(K, ld)
        
        # Check if any warning was triggered
        if any(issubclass(warn.category, LinAlgWarning) for warn in warneo):
            # Print the warning message for all warnings captured
            for warn in warneo:
                print(f"Warning: {warn.message} In iter {k}")
    return delta_vector

def update_active_set_mask1(mu, z, Q, k, tau, active_set_history, mudf, mu_percentage_change, z_percentage_change, epsilon=1e-5, comp_tol=1e-5):
    # Previous μ from mudf
    prev_mu = mudf.loc[k-1].values
    
    # Only consider indices where μ*z <= comp_tol
    mask = mu * z <= comp_tol
    
    # Highlight if μ did not increase significantly and μ*z is small enough
    highlighted_rows = [i for i in range(len(mu)) 
                    if mask[i] and mu[i] < prev_mu[i] and z_percentage_change[i] > -0.03 and z[i] > 0]
    #highlighted_rows = [i for i in range(len(mu)) if mask[i] and mu[i] <= prev_mu[i] + epsilon]
    
    # Build row of 1s and 0s
    highlighted_row = [1 if i in highlighted_rows else 0 for i in range(len(mu))]
    
    # Store in DataFrame
    active_set_history.loc[k] = highlighted_row
    return active_set_history

def update_active_set_mask( mu, z, Q, k, tau, active_set_history, mudf, mu_percentage_change, z_percentage_change,
                           epsilon=1e-5, complementarity_tol=1e-5, tapia_tol=0.8):
    prev_mu = mudf.loc[k-1].values

    # Condition: complementarity is sufficiently small
    mask = mu * z <= complementarity_tol

    # Tapia indicator of mu: successive-iterate ratio mu_i^k / mu_i^{k-1}.
    # Under strict complementarity it -> 0 for inactive constraints (mu -> 0) and
    # -> 1 for active ones; a degenerate index parks near 1/2. cond5 below is the
    # continuous refinement of cond2's binary "mu decreased" test.
    # Ref: El-Bakry, Tapia & Zhang (1994), SIAM Review 36(1), 45-72.
    tapia_mu = mu / (prev_mu + 1e-300)

    highlighted_rows = []

    for i in range(len(mu)):
        cond1 = mask[i]
        cond2 = mu[i] < prev_mu[i]
        cond3 = z_percentage_change[i] > -0.03
        cond4 = z[i] > 0 # we could remove this as it's interior
        cond5 = tapia_mu[i] < tapia_tol   # Tapia indicator says mu is collapsing to 0

        if (cond1 and cond2 and cond3 and cond4 and cond5) or mu[i]==0:
            highlighted_rows.append(i)
        else:
            # Only print if it was active before → regression
            if len(active_set_history) > 1 and active_set_history.iloc[-1, i] == 1:
                failed_conditions = []

                if not cond1:
                    failed_conditions.append(
                        f"Complementarity too large (mu*z = {mu[i]*z[i]:.2e} > {complementarity_tol})"
                    )
                if not cond2:
                    failed_conditions.append(
                        f"μ did not decrease AT ALL (current: {mu[i]:.2e}, previous: {prev_mu[i]:.2e})",
                    )
                    failed_conditions.append(
                        f"z_{i} is {z[i]:.2e}",
                    )
                if not cond3:
                    failed_conditions.append(
                        f"z decreased too fast (Δz/z = {z_percentage_change[i]:.2%}, threshold = -3%)" #STRICT COMPLEMENTARITY
                    )
                if not cond4:
                    failed_conditions.append(
                        f"z is not positive (z = {z[i]:.2e})"
                    )
                if not cond5:
                    failed_conditions.append(
                        f"Tapia indicator too high (mu ratio = {tapia_mu[i]:.3f} >= {tapia_tol}: mu not collapsing to 0)"
                    )
                
                print(f"[Iteration {k}] Index {i} stopped meeting at least one criteria to be considered as zero:")
                print(f"mu_percentage_change_{i} is {mu_percentage_change[i]:.2e}")
                print(f"z_percentage_change_{i} is {z_percentage_change[i]:.2e}")
                for reason in failed_conditions:
                    print(f"   - {reason}")
    
    highlighted_row = [1 if i in highlighted_rows else 0 for i in range(len(mu))]
    
    active_set_history.loc[k] = highlighted_row
    return active_set_history


def active_set_diagnostics(active_set_history,regression_df,p, num_last_iterations=2):
    # Indices consistently flagged over the last `num_last_iterations` iterations
    # (a robustness filter / debounce). Keep this window small: fast-converging
    # problems only spend ~2 iterations near the solution, so a large window
    # (e.g. 5) starves the elimination on them. Validated against an independent
    # solver on the Maros-Meszaros set: shrinking 5 -> 2 raised recall (tiny
    # problems 0% -> 100%, larger ones ~+15pts) with zero false positives.
    # Default 2; pass a larger value for extra caution on noisy problems.
    last_iterations = active_set_history.tail(num_last_iterations)

    # Columns consistently highlighted (1 in all last x iterations)
    stable_active_indices = last_iterations.columns[(last_iterations == 1).all(axis=0)].tolist()
    stable_active_indices = [int(x) for x in stable_active_indices]  # ensure ints
    print(f"Consistently highlighted in last {num_last_iterations} iterations:", stable_active_indices)
    print("How many in percentage of mu's dimension? ", (len(stable_active_indices)/p)*100,"%")
    #print("How many?=", len(stable_active_indices))
    
    # Columns that were highlighted in previous iteration but regressed this iteration (1 → 0)
    regressed_indices = []
    if len(active_set_history) >= 2:
        prev_iter = active_set_history.iloc[-2]
        last_iter = active_set_history.iloc[-1]
        regressed_indices = prev_iter.index[(prev_iter == 1) & (last_iter == 0)].tolist()
        regressed_indices = [int(x) for x in regressed_indices]
        if regressed_indices:
            print(f"WARNING: Iter {active_set_history.index[-1]} regressed columns (1 -> 0):", regressed_indices)

            active_set_history_regressed = active_set_history.tail(10)[regressed_indices]
            display(active_set_history_regressed)
        iter_idx = active_set_history.index[-1]
        if regression_df is not None:
                row = pd.Series(0, index=active_set_history.columns)
                row[regressed_indices] = 1
                regression_df.loc[iter_idx] = row
    
    return stable_active_indices, regressed_indices


def progress_summary_df_clean(problem_results_before_heuristic, metrics_after=None):
    """
    Build a progress summary DataFrame.

    If metrics_after is provided → Before vs After comparison.
    If metrics_after is None → Only show the IPM metrics.
    """

    summary_data = {
        "Metric": [],
        "Before": []
    }

    if metrics_after is not None:
        summary_data["After"] = []
        summary_data["Did it decrease?"] = []

    for metric in problem_results_before_heuristic:
        before = problem_results_before_heuristic[metric]

        summary_data["Metric"].append(metric)
        summary_data["Before"].append(before)

        if metrics_after is not None:
            after = metrics_after[metric]
            summary_data["After"].append(after)
            summary_data["Did it decrease?"].append(after < before)

    summary_df = pd.DataFrame(summary_data)

    return summary_df

def remove_rows_cols_K(Q, AT, FT, D_mu, A, F, D_z, mu, z, x, lamda, c, b, d, tau, stable_active_indices,r_x, r_lamda):
    """
    Builds the full KKT system K and a reduced system called K1 by eliminating
    rows/columns corresponding to highlighted indices.
    
    Returns:
        K      : full KKT system
        K1     : reduced KKT system (square)
        D_mu1     : filtered diagonal of mu for reduced system
        L1    : reduced RHS vector
    """
    # Dimensions
    n = Q.shape[0]
    m = A.shape[0]
    p = D_mu.shape[0]

    # ────────── Build system ──────────¡
    row1 = np.hstack((Q, -AT, -FT @ D_mu))
    row2 = np.hstack((-A, np.zeros((m, m + p))))
    row3 = np.hstack((-D_mu @ F, np.zeros((p, m)), -np.diag(mu * z)))

    K = np.vstack((row1, row2, row3))

    L = np.concatenate((
        r_x ,   # dual residual
        r_lamda,                          # primal residual
        D_mu @ (d - F @ x) + tau               # complementarity row
    ))

    # ────────── Filter mu ──────────
    active_indices = [i for i in range(p) if i not in stable_active_indices]
    mu_filtered = mu[active_indices]
    D_mu1 = np.diag(mu_filtered)

    # ────────── Build reduced system ──────────
    rows_to_remove = [i + (n + m) for i in stable_active_indices]  # only the μ rows
    K1 = np.delete(K, rows_to_remove, axis=0)
    K1 = np.delete(K1, rows_to_remove, axis=1)

    if K1.shape[0] != K1.shape[1]:
        raise ValueError("K1 is not square! Check highlighted indices.")

    L1 = np.delete(L, rows_to_remove, axis=0)

    print(f"Deleted {len(rows_to_remove)} rows/columns. K1 shape: {K1.shape}")

    return K1, D_mu1, L1

def build_reduced_system(Q, AT, FT, D_mu, A, F, D_z, mu, x, lamda, c, b, d, tau, stable_active_indices):
    """
    Builds the full KKT system K and a reduced system called K1 by eliminating
    rows/columns corresponding to highlighted indices.
    
    Returns:
        K      : full KKT system
        K1     : reduced KKT system (square)
        D_mu1     : filtered diagonal of mu for reduced system
        L1    : reduced RHS vector
    """
    # Dimensions
    n = Q.shape[0]
    m = A.shape[0]
    p = D_mu.shape[0]

    # ────────── Build full system ──────────
    r1 = np.hstack((Q, AT, -FT @ D_mu))
    r2 = np.hstack((A, np.zeros((m, m + p))))
    r3 = np.hstack((-D_mu @ F, np.zeros((p, m)), -D_z @ D_mu))

    K = np.vstack((r1, r2, r3))   # full KKT system

    # ────────── Filter mu ──────────
    active_indices = [i for i in range(p) if i not in stable_active_indices]
    mu_filtered = mu[active_indices]
    D_mu1 = np.diag(mu_filtered)

    # ────────── Build reduced system ──────────
    rows_to_remove = [i + (n + m) for i in stable_active_indices]  # only the μ rows
    K1 = np.delete(K, rows_to_remove, axis=0)
    K1 = np.delete(K1, rows_to_remove, axis=1)

    if K1.shape[0] != K1.shape[1]:
        raise ValueError("K1 is not square! Check highlighted indices.")

    # ────────── Build reduced RHS vector ──────────
    L_full = np.concatenate((
        Q @ x + AT @ lamda - FT @ mu + c,   # dual residual
        A @ x - b,                          # primal residual
        D_mu @ (d - F @ x) + tau               # complementarity row
    ))

    L1 = np.delete(L_full, rows_to_remove, axis=0)

    print(f"Deleted {len(rows_to_remove)} rows/columns. K1 shape: {K1.shape}")

    return K, K1, D_mu1, L1

def load_lp_problem(mat_file: str):
    """
    Loads a linear/quadratic problem from a .mat file and initializes
    the problem matrices for the interior-point algorithm.
    
    Returns:
        Q : ndarray    Quadratic matrix (identity by default)
        c : ndarray    Linear term vector
        A : ndarray    Equality constraint matrix
        b : ndarray    Equality constraint RHS
        F : ndarray    Inequality constraint matrix (identity by default)
        d : ndarray    Inequality constraint RHS (zeros)
        H : dict       Raw data loaded from the .mat file
    """
    print(f"Loading problem from: {mat_file}")

    # --- General QP format (e.g. converted Maros-Meszaros): explicit Q,c,A,b,F,d ---
    # A file that already carries these fields is a full QP
    #   min 1/2 x^T Q x + c^T x   s.t.  A x = b,   F x - d >= 0
    # and is loaded verbatim (Q and the constraints are NOT overwritten).
    raw = scipy.io.loadmat(f"mat_files/{mat_file}")
    if all(k in raw for k in ("Q", "c", "A", "b", "F", "d")):
        Q = np.asarray(raw["Q"], dtype=float)
        n = Q.shape[0]
        c = np.asarray(raw["c"], dtype=float).ravel()
        A = np.asarray(raw["A"], dtype=float).reshape(-1, n)   # (m, n); m may be 0
        b = np.asarray(raw["b"], dtype=float).ravel()
        F = np.asarray(raw["F"], dtype=float).reshape(-1, n)   # (p, n)
        d = np.asarray(raw["d"], dtype=float).ravel()
        print(f"Problem loaded (general QP). n={n}, m={A.shape[0]}, p={F.shape[0]}")
        return Q, c, A, b, F, d, raw

    # --- Legacy NETLIB LP -> QP (Q=I, F=I, d=0) ---
    H = loadProblem(f"mat_files/{mat_file}")

    # Quadratic term: identity (can be changed if needed)
    Q = np.eye(H['c'].shape[0])

    # Linear term
    c = H['c'].ravel()  # flatten in case it's a column vector

    # Equality constraints
    A = H['AE']
    b = H['bE'].ravel()  # flatten in case it's a column vector

    # Compute percentage of zeros in b
    num_zeros_b = np.sum(b == 0)
    pct_zeros_b = 100 * num_zeros_b / len(b)
    
    # Inequality constraints (x >= 0 by default)
    F = np.eye(H['c'].shape[0])
    d = np.zeros(H['c'].shape[0])

    print(f"Problem loaded. n={Q.shape[0]}, m={A.shape[0]}, p={F.shape[0]}")
    #print(f"Equality RHS b: {b}")
    print(f"Number of zeros in b: {num_zeros_b} ({pct_zeros_b:.2f}%)")

    return Q, c, A, b, F, d, H

def export_latex_tables(summary_df, red_results_df, mat_file, p):
    """
    Prepares and exports LaTeX tables for summary_df and red_results_df.
    """

    # Copy to avoid modifying originals
    summary_df_latex = summary_df.copy()
    red_results_df_latex = red_results_df.copy()

    # ==========================================================
    # SUMMARY TABLE
    # ==========================================================
    summary_df_latex.columns = [
        "Métrica",
        "Después del IPM",
        "Después del experimento",
        "Mejoró?"
    ]

    latex_metric_names = {
        "overall ||ld||∞": r"$\|F\|_\infty$",
        "primal ||·||∞": r"$\|r_{\text{primal}}\|_\infty$",
        "ineq ||·||∞": r"$\|r_{\text{ineq}}\|_\infty$",
        "max(mu*z)": r"$\max(\mu_i z_i)$",
        "tau": r"$\tau$",
        "cond(G)": r"$\mathrm{cond}(G)$",
        "objective": r"$\frac{1}{2} x^T Q x + c^T x$",
        "complementariedad": r"$\max(\mu_i z_i)$"
    }

    summary_df_latex.iloc[:, 0] = summary_df_latex.iloc[:, 0].map(latex_metric_names)

    # ==========================================================
    # REDUCED RESULTS TABLE
    # ==========================================================

    red_results_df_latex.columns = [
        "fx",
        "max_muz",
        "dimM1",
        "condG",
        "condM1",
        "elim"
    ]

    red_results_df_latex["dimM1"] = red_results_df_latex["dimM1"].astype(int)
    red_results_df_latex["elim"]  = red_results_df_latex["elim"].astype(int)

    # % reducción
    red_results_df_latex["perc_red"] = (
        ((red_results_df_latex["elim"] / 
          (red_results_df_latex["dimM1"] + red_results_df_latex["elim"])) * 100)
        .round(0)
        .astype(int)
        .astype(str) + "\%"
    )

    # % mu cero
    red_results_df_latex["perc_mu_zero"] = (
        ((red_results_df_latex["elim"] / p) * 100)
        .round(0)
        .astype(int)
        .astype(str) + "\%"
    )

    # Format floats
    float_cols = ["fx", "max_muz", "condG", "condM1"]
    red_results_df_latex[float_cols] = \
        red_results_df_latex[float_cols].map(lambda x: f"{x:.2e}")

    # Final LaTeX column names
    red_results_df_latex.columns = [
        r"$f(x)$",
        r"$\max(\mu_i z_i)$",
        r"$\mathrm{dim}(M_1)$",
        r"$\mathrm{Cond}(G)$",
        r"$\mathrm{Cond}(M_1)$",
        r"Filas/Cols elim",
        r"\% Reducción",
        r"\% $\mu$ cero"
    ]

    red_results_df_latex.index = red_results_df.index
    red_results_df_latex.index.name = "Iter"

    # ==========================================================
    # EXPORT FILES
    # ==========================================================

    with open(f"progress_summary/summary_df_latex_{mat_file}.txt", "w") as f:
        f.write(f"% Summary table for LaTeX — Matrices: {mat_file}\n\n")
        f.write(summary_df_latex.to_latex(
            index=False,
            float_format="%.4e",
            escape=False
        ))

    with open(f"progress_summary/red_results_df_latex_{mat_file}.txt", "w") as f:
        f.write(f"% Reduced results table for LaTeX — Matrices: {mat_file}\n\n")
        f.write(red_results_df_latex.to_latex(
            index=True,
            escape=False
        ))

def export_summary_to_latex(summary_before, mat_file, output_dir="progress_summary"):
    """Export summary_before DataFrame to a LaTeX table file."""

    latex_metric_names = {
    "KKT residual (‖r‖∞)": r"$\|r_{\mathrm{KKT}}\|_\infty$",
    "Primal residual (‖r_p‖∞)": r"$\|r_p\|_\infty$",
    "Inequality residual (‖r_d‖∞)": r"$\|r_d\|_\infty$",
    "Complementarity gap (max μᵢ zᵢ)": r"$\max_i \mu_i z_i$",
    "Barrier parameter (τ)": r"$\tau$",
    "Condition number κ(G)": r"$\kappa(G)$",
    "Objective value f(x)": r"$f(x)$"
    }

    summary_df_latex = summary_before.copy()

    summary_df_latex.columns = ["Métrica", "Después del IPM"]

    summary_df_latex.iloc[:, 0] = summary_df_latex.iloc[:, 0].map(latex_metric_names)

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"summary_IPM_latex_{mat_file}.txt")

    with open(file_path, "w") as f:
        f.write(f"% Summary table for LaTeX — Matrices: {mat_file}\n\n")
        f.write(summary_df_latex.to_latex(
            index=False,
            float_format="%.4e",
            escape=False
        ))

    return file_path