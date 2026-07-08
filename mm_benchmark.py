"""
mm_benchmark.py
===============
Independent "answer key" for validating the interior-point heuristic, using the
Maros-Meszaros convex-QP test set (Maros & Meszaro, 1999) -- a standard, citable
benchmark. The point: get the TRUE solution / active set of each problem from a
method that is NOT our own IPM, so it is not corrupted by the ill-conditioning
we are studying.

Pipeline per problem:
  1. load the Maros-Meszaros .mat  (fields Q, c, A, rl, ru, lb, ub) with
         min 1/2 x^T Q x + c^T x    s.t.  rl <= A x <= ru,   lb <= x <= ub
  2. convert to the thesis QP form
         min 1/2 x^T Q x + c^T x    s.t.  A_eq x = b_eq,   F x - d >= 0
  3. solve it with TWO independent solvers (clarabel, osqp) via qpsolvers,
  4. cross-check that they agree, and report the active set
         active constraint  <=>  (F x* - d)_i ~ 0
     The inactive constraints are the mu_i -> 0 ones the heuristic must detect.

Note: the .mat files drop the objective's constant term c0, so the reported
objective differs from the published optimum by that constant; the SOLUTION
VECTOR x* (hence the active set) is unaffected.

Usage:
    python mm_benchmark.py            # run the whole curated PD set, print a table
    python mm_benchmark.py HS21       # run one problem, print full detail
    python mm_benchmark.py --list     # list available problems

Requires: qpsolvers, clarabel, osqp   (pip install qpsolvers clarabel osqp)
"""
import os
import sys
import glob
import numpy as np
import scipy.io
import scipy.sparse as sp

try:
    import qpsolvers
except ImportError:
    sys.exit("Need qpsolvers: pip install qpsolvers clarabel osqp")

MM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_files", "maros_meszaros")
ACTIVE_TOL = 1e-6          # |F x - d| below this  =>  constraint counted active
PD_TOL = 1e-9              # min eigenvalue above this => Q strictly positive definite


def load_mm_to_ipm(name):
    """Load a Maros-Meszaros problem and convert to (Q, c, A_eq, b_eq, F, d).

    Rows with rl==ru become equalities; one-sided/ranged rows and finite
    variable bounds become rows of F with  F x - d >= 0.
    """
    m = scipy.io.loadmat(os.path.join(MM_DIR, f"{name}.mat"))
    Q = m["Q"].toarray().astype(float)
    c = m["c"].astype(float).ravel()
    A = m["A"].toarray().astype(float)
    rl = m["rl"].astype(float).ravel()
    ru = m["ru"].astype(float).ravel()
    lb = m["lb"].astype(float).ravel()
    ub = m["ub"].astype(float).ravel()
    n = Q.shape[0]

    A_eq, b_eq, F, d, labels = [], [], [], [], []
    for i in range(A.shape[0]):
        if np.isfinite(rl[i]) and np.isfinite(ru[i]) and np.isclose(rl[i], ru[i]):
            A_eq.append(A[i]); b_eq.append(rl[i])
        else:
            if np.isfinite(rl[i]):
                F.append(A[i]);  d.append(rl[i]);  labels.append(f"Ax_{i} >= {rl[i]:g}")
            if np.isfinite(ru[i]):
                F.append(-A[i]); d.append(-ru[i]); labels.append(f"Ax_{i} <= {ru[i]:g}")
    for j in range(n):
        if np.isfinite(lb[j]):
            e = np.zeros(n); e[j] = 1.0
            F.append(e);  d.append(lb[j]);  labels.append(f"x_{j} >= {lb[j]:g}")
        if np.isfinite(ub[j]):
            e = np.zeros(n); e[j] = -1.0
            F.append(e); d.append(-ub[j]); labels.append(f"x_{j} <= {ub[j]:g}")

    return dict(
        name=name, n=n,
        Q=Q, c=c,
        A=np.array(A_eq) if A_eq else np.zeros((0, n)),
        b=np.array(b_eq) if b_eq else np.zeros(0),
        F=np.array(F) if F else np.zeros((0, n)),
        d=np.array(d) if d else np.zeros(0),
        labels=labels,
    )


def is_pd(Q):
    try:
        np.linalg.cholesky(Q)
        return True
    except np.linalg.LinAlgError:
        return False


def solve_independent(P, solver):
    """Solve  min 1/2 xQx + cx  s.t. Ax=b, Fx>=d  with an external solver.
    qpsolvers form: min 1/2 xPx+qx s.t. Gx<=h, Ax=b  ->  Fx>=d is -Fx<=-d."""
    kw = dict(P=sp.csc_matrix(P["Q"]), q=P["c"],
              G=sp.csc_matrix(-P["F"]), h=-P["d"])
    if P["A"].shape[0] > 0:
        kw.update(A=sp.csc_matrix(P["A"]), b=P["b"])
    return qpsolvers.solve_qp(**kw, solver=solver)


def active_set(P, x):
    slack = P["F"] @ x - P["d"]
    active = np.where(np.abs(slack) < ACTIVE_TOL)[0]
    return active, slack


def analyze(name, verbose=False):
    P = load_mm_to_ipm(name)
    pd = is_pd(P["Q"])
    xs = {s: None for s in ("clarabel", "osqp")}
    for s in xs:
        if s in qpsolvers.available_solvers:
            try:
                xs[s] = solve_independent(P, s)
            except Exception:
                xs[s] = None
    xc = xs["clarabel"] if xs["clarabel"] is not None else xs["osqp"]
    result = dict(P=P, pd=pd, xs=xs, x=xc)
    if xc is None:
        result.update(ok=False)
        return result
    obj = 0.5 * xc @ P["Q"] @ xc + P["c"] @ xc
    agree = (np.linalg.norm(xs["clarabel"] - xs["osqp"], np.inf)
             if xs["clarabel"] is not None and xs["osqp"] is not None else np.nan)
    active, slack = active_set(P, xc)
    result.update(ok=True, obj=obj, agree=agree,
                  n_active=len(active), n_ineq=P["F"].shape[0], active=active)
    if verbose:
        print(f"\n=== {name} ===")
        print(f"  n={P['n']}  equalities={P['A'].shape[0]}  inequalities(p)={P['F'].shape[0]}  Q>0={pd}")
        print(f"  objective (up to dropped constant c0) = {obj:.6f}")
        print(f"  two-solver agreement  ||x_clarabel - x_osqp||inf = {agree:.2e}")
        print(f"  x* = {np.round(xc, 5)}" if P['n'] <= 12 else f"  ||x*|| = {np.linalg.norm(xc):.4f}")
        print(f"  ACTIVE constraints: {len(active)} of {P['F'].shape[0]}")
        if P["n"] <= 20:
            for i in active:
                print(f"      [{i}] {P['labels'][i]}")
        print(f"  INACTIVE (mu->0 targets for the heuristic): {P['F'].shape[0] - len(active)}")
    return result


def main():
    args = sys.argv[1:]
    if not os.path.isdir(MM_DIR):
        sys.exit(f"No problem directory: {MM_DIR}")
    problems = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(MM_DIR, "*.mat")))

    if args and args[0] == "--list":
        print("available problems:", ", ".join(problems)); return
    if args:
        for name in args:
            analyze(name, verbose=True)
        return

    # table over the whole curated set
    print(f"solvers available: {qpsolvers.available_solvers}\n")
    hdr = f"{'problem':12s} {'n':>5} {'eq':>4} {'ineq':>5} {'Q>0':>4} {'objective':>14} {'agree':>9} {'active/ineq':>12}"
    print(hdr); print("-" * len(hdr))
    for name in problems:
        r = analyze(name)
        if not r["ok"]:
            print(f"{name:12s}  (solve failed)"); continue
        P = r["P"]
        print(f"{name:12s} {P['n']:>5} {P['A'].shape[0]:>4} {P['F'].shape[0]:>5} "
              f"{str(r['pd']):>4} {r['obj']:>14.6f} {r['agree']:>9.1e} "
              f"{r['n_active']:>5}/{r['n_ineq']:<6}")
    print("\nactive = tight constraints; inactive (ineq - active) are the mu->0 rows")
    print("the heuristic aims to detect. objective omits the .mat's dropped constant c0.")


if __name__ == "__main__":
    main()
