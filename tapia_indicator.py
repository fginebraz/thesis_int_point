"""
tapia_indicator.py
==================
SANDBOX (observational only -- changes no detection logic).

Computes the Tapia indicator on one problem to see whether it separates the
inactive constraints (mu -> 0) from the active ones, grouped by the independent
answer key in mm_qp_problems.py.

Tapia indicator of a sequence v_i:   t_i^k = v_i^{k+1} / v_i^k
Under strict complementarity:
    v_i -> 0             =>  t_i^k -> 0
    v_i -> positive const=>  t_i^k -> 1
So per constraint we expect:
    inactive (mu->0):  mu-indicator -> 0 , z-indicator -> 1
    active   (z ->0):  mu-indicator -> 1 , z-indicator -> 0
    degenerate (both->0): both -> ~1/2   (no such problem in this test set)

Ref: El-Bakry, Tapia & Zhang (1994), "A study of indicators for identifying
zero variables in interior-point methods", SIAM Review 36(1), 45-72.

Usage:  .venv/bin/python tapia_indicator.py [mm_QPCBLEND.mat]
"""
import sys, io, contextlib
import numpy as np

from compare import run_ipm
import mm_qp_problems as key

EPS = 1e-300   # guard against 0/0 on fully-converged rows


def tapia_last(df):
    """Final Tapia indicator per column = last row / previous row."""
    v_prev = df.iloc[-2].to_numpy(dtype=float)
    v_last = df.iloc[-1].to_numpy(dtype=float)
    return v_last / (v_prev + EPS)


def group_stats(name, indicator, inactive_mask):
    """Print min/median/max of an indicator, split by inactive vs active."""
    for grp, mask in [("inactive (expect ->0 for mu)", inactive_mask),
                      ("active   (expect ->1 for mu)", ~inactive_mask)]:
        vals = indicator[mask]
        if vals.size:
            print(f"  {name} | {grp:30s}  n={vals.size:4d}  "
                  f"min={vals.min():7.3f}  median={np.median(vals):7.3f}  max={vals.max():7.3f}")


def main(mat_file):
    with contextlib.redirect_stdout(io.StringIO()):      # hush the chatty loop
        R = run_ipm(mat_file)
    p = R["p"]

    # answer key: which constraints are truly inactive (mu -> 0)
    x_star = key.solve(R["Q"], R["c"], R["A"], R["b"], R["F"], R["d"])
    inactive = key.mu_to_zero(R["F"], R["d"], x_star)     # boolean, length p

    mu_ind = tapia_last(R["mu_df"])
    z_ind = tapia_last(R["z_df"])

    print(f"\n=== Tapia indicator on {mat_file}  (p={p}, {R['k']} iters) ===")
    print(f"true inactive (mu->0): {int(inactive.sum())}   active: {int((~inactive).sum())}\n")
    group_stats("mu-ind", mu_ind, inactive)
    group_stats("z-ind ", z_ind, inactive)

    # a couple of concrete example constraints, last few iterations
    print("\nExample trajectories (mu-indicator over last iterations):")
    ex = []
    if inactive.any():
        ex.append((int(np.where(inactive)[0][0]), "inactive"))
    if (~inactive).any():
        ex.append((int(np.where(~inactive)[0][0]), "active"))
    tail = min(6, len(R["mu_df"]) - 1)
    mu_hist = R["mu_df"]
    for idx, kind in ex:
        col = mu_hist[idx].to_numpy(dtype=float)
        ratios = col[1:] / (col[:-1] + EPS)
        print(f"  constraint {idx:4d} ({kind:8s}): " +
              " ".join(f"{r:6.3f}" for r in ratios[-tail:]))


if __name__ == "__main__":
    mat = sys.argv[1] if len(sys.argv) > 1 else "mm_QPCBLEND.mat"
    main(mat)
