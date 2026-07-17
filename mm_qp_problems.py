"""
mm_qp_problems.py
=================
Answer-key module (kept fully separate from the IPM code).
Uses Clarabel to get the TRUE solution of each Maros-Meszaros problem, so we can
later compare our IPM + heuristic against it.

Part 1: load the problems and get their optimal value.
Part 2: get the solution x* and the "answer key" -- which constraints have mu->0.

The .mat files only store the data:
    min 1/2 x^T Q x + c^T x    s.t.   rl <= A x <= ru,   lb <= x <= ub
so we (a) rewrite it in our thesis form and (b) solve it with Clarabel.
"""
import numpy as np
import scipy.io
import scipy.sparse as sp
from qpsolvers import solve_qp

FOLDER = "mat_files/maros_meszaros"
SLACK_TOL = 1e-6          # slack above this => the constraint is inactive (mu -> 0)

# Q positive definite, with equality and inequality constraints.
# The HS* entries are the true QPs (quadratic objective + LINEAR constraints,
# classification QLR) from Hock & Schittkowski (1981) that are also in this set.
PROBLEMS = ["TAME",
            "HS21", "HS35", "HS53", "HS76", "HS118",
            "DUAL1", "DUAL2", "DUAL3", "DUAL4", "QPCBLEND"]

# Reference objective values.  Two provenances:
#
#  (a) HS problems -- ANALYTIC optima from Hock & Schittkowski (1981), "Test
#      Examples for Nonlinear Programming Codes".  These are closed-form (exact
#      fractions), NOT solver output.  Page = the printed page of that book.
#  (b) all others (TAME, DUAL*, QPCBLEND) -- the OPT column of Maros & Meszaros
#      (1999), DOI 10.1080/10556789908805768.  The paper defines OPT as "the
#      solution value obtained by the default settings of the BPMPD solver", so
#      these are themselves solver-computed (BPMPD, interior-point), not analytic.
#      Comparing Clarabel to OPT is a cross-check between two independent solvers.
PUBLISHED = {
    "TAME":      0.0,            #  MM OPT  0.0000000e+00
    "HS21":    -99.96,           #  HS p.44   x*=(2,0),                f*=-99.96
    "HS35":      1/9,            #  HS p.58   x*=(4/3,7/9,4/9),        f*=1/9
    "HS53":    176/43,           #  HS p.76   x*=(-33,11,27,-5,11)/43, f*=176/43
    "HS76":   -103/22,           #  HS p.99   f*=-103/22 = -4.681818...
    "HS118": 13296409/20000,     #  HS p.140  x*=integers, f*=13296409/20000 = 664.82045 (exact)
    "DUAL1":     0.035012966,    #  MM OPT  3.5012966e-02
    "DUAL2":     0.033733676,    #  MM OPT  3.3733676e-02
    "DUAL3":     0.13575584,     #  MM OPT  1.3575584e-01
    "DUAL4":     0.74609084,     #  MM OPT  7.4609084e-01
    "QPCBLEND": -0.0078425409,   #  MM OPT -7.8425409e-03
}
# Objective constant c0 that the .mat format drops (from the .QPS RHS OBJ.FUNC,
# or the constant term of the HS objective).  Clarabel computes 1/2 x'Qx + c'x;
# the published value includes + c0.
C0 = {"HS21": -100.0, "HS35": 9.0, "HS53": 6.0}   # 0 for every other problem


def load(name):
    """Read a Maros-Meszaros problem from its .mat file."""
    d = scipy.io.loadmat(f"{FOLDER}/{name}.mat")
    Q  = d["Q"].toarray()
    c  = d["c"].astype(float).ravel()
    A  = d["A"].toarray()
    rl = d["rl"].astype(float).ravel()   # lower bound of A x
    ru = d["ru"].astype(float).ravel()   # upper bound of A x
    lb = d["lb"].astype(float).ravel()   # lower bound of x
    ub = d["ub"].astype(float).ravel()   # upper bound of x
    return Q, c, A, rl, ru, lb, ub


def to_thesis_form(Q, c, A, rl, ru, lb, ub):
    """Rewrite  rl <= A x <= ru,  lb <= x <= ub  into our thesis form:
           min 1/2 x^T Q x + c^T x    s.t.   A x = b,   F x - d >= 0
    Equalities: rows where rl == ru. Everything else (one-sided rows and the
    variable bounds) becomes a row of F with  F x - d >= 0.
    Returns Q, c, Aeq, beq, F, d  (F, d hold every inequality/bound).
    """
    n = Q.shape[0]
    Aeq, beq, F, d = [], [], [], []
    # linear rows  rl <= A x <= ru
    for i in range(A.shape[0]):
        if np.isclose(rl[i], ru[i]):                       # equality
            Aeq.append(A[i]); beq.append(rl[i])
        else:
            if np.isfinite(rl[i]): F.append( A[i]); d.append( rl[i])   # A x >= rl
            if np.isfinite(ru[i]): F.append(-A[i]); d.append(-ru[i])   # A x <= ru
    # variable bounds  lb <= x <= ub
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        if np.isfinite(lb[j]): F.append( e.copy()); d.append( lb[j])   # x_j >= lb
        if np.isfinite(ub[j]): F.append(-e.copy()); d.append(-ub[j])   # x_j <= ub
    Aeq = np.array(Aeq) if Aeq else np.zeros((0, n))
    beq = np.array(beq) if beq else np.zeros(0)
    return Q, c, Aeq, beq, np.array(F), np.array(d)


def solve(Q, c, Aeq, beq, F, d):
    """Solve  min 1/2 x^T Q x + c^T x  s.t. Aeq x = beq, F x - d >= 0  with Clarabel.
    qpsolvers uses G x <= h, so F x >= d becomes -F x <= -d.  Returns x*."""
    return solve_qp(
        P=sp.csc_matrix(Q), q=c,
        G=sp.csc_matrix(-F), h=-d,
        A=sp.csc_matrix(Aeq) if Aeq.shape[0] else None,
        b=beq if Aeq.shape[0] else None,
        solver="clarabel",
    )


def mu_to_zero(F, d, x):
    """Answer key: boolean mask of the inactive constraints (mu -> 0).
    By complementary slackness, if the slack (F x - d)_i > 0 then mu_i = 0."""
    slack = F @ x - d
    return slack > SLACK_TOL


if __name__ == "__main__":
    print(f"{'problem':10s} {'mu->0':>6} {'active':>7} | "
          f"{'Clarabel+c0':>12} {'published':>12} {'|diff|':>9}  match?")
    print("-" * 68)
    for name in PROBLEMS:
        Q, c, A, rl, ru, lb, ub = load(name)
        Q, c, Aeq, beq, F, d = to_thesis_form(Q, c, A, rl, ru, lb, ub)
        x = solve(Q, c, Aeq, beq, F, d)
        inactive = mu_to_zero(F, d, x)

        clarabel = 0.5 * x @ Q @ x + c @ x + C0.get(name, 0.0)   # add back the dropped constant
        published = PUBLISHED[name]
        diff = abs(clarabel - published)
        ok = "OK" if diff <= 1e-4 * max(1.0, abs(published)) else "CHECK"
        print(f"{name:10s} {int(inactive.sum()):>6} {int((~inactive).sum()):>7} | "
              f"{clarabel:>12.6f} {published:>12.6f} {diff:>9.1e}  {ok}")
