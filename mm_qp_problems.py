"""
mm_qp_problems.py — load Maros-Meszaros problems, solve with Clarabel, get answer key for mu->0.
"""
import numpy as np
import scipy.io
import scipy.sparse as sp
from qpsolvers import Problem, solve_problem

FOLDER = "mat_files/maros_meszaros"
RATIO_TOL = 100           # z/mu above this => clearly inactive (mu -> 0)

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
    """Solve with Clarabel. Returns the full Solution object (x, z, obj, etc).
    qpsolvers uses G x <= h, so F x >= d becomes -F x <= -d."""
    problem = Problem(
        P=sp.csc_matrix(Q), q=c,
        G=sp.csc_matrix(-F), h=-d,
        A=sp.csc_matrix(Aeq) if Aeq.shape[0] else None,
        b=beq if Aeq.shape[0] else None,
    )
    return solve_problem(problem, solver="clarabel")


ANSWER_KEY_FILE = "answer_key.json"


def mu_to_zero(F, d, x, mu_star):
    """Answer key: boolean mask of inactive constraints (mu -> 0).
    Uses z*/mu* ratio: when slack >> dual (ratio > RATIO_TOL), mu clearly went to zero."""
    slack = F @ x - d
    ratio = slack / np.maximum(mu_star, 1e-20)
    return ratio > RATIO_TOL


def generate_answer_key(problems=None):
    """Solve all problems with Clarabel, store everything in a JSON file."""
    import json
    if problems is None:
        problems = PROBLEMS

    key = {}
    for name in problems:
        Q, c, A, rl, ru, lb, ub = load(name)
        Q, c, Aeq, beq, F, d = to_thesis_form(Q, c, A, rl, ru, lb, ub)
        sol = solve(Q, c, Aeq, beq, F, d)

        c0 = C0.get(name, 0.0)
        slack = F @ sol.x - d

        key[name] = {
            "n": Q.shape[0],
            "m": Aeq.shape[0],
            "p": F.shape[0],
            "c0": c0,
            "published": PUBLISHED[name],
            "source": "Hock-Schittkowski 1981" if name.startswith("HS") else "Maros-Meszaros 1999",
            "clarabel_obj": sol.obj + c0,
            "x_star": sol.x.tolist(),
            "mu_star": sol.z.tolist(),
            "y_star": list(sol.y) if sol.y is not None else [],
            "slack": slack.tolist(),
        }
        print(f"  {name}: p={F.shape[0]}")

    with open(ANSWER_KEY_FILE, "w") as f:
        json.dump(key, f, indent=2)
    print(f"Saved to {ANSWER_KEY_FILE}")
    return key


def load_answer_key():
    """Load previously generated answer key from JSON."""
    import json
    with open(ANSWER_KEY_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Generating answer key...\n")
    generate_answer_key()

    print(f"\n{'problem':10s} {'p':>4} | {'sol.obj+c0':>14} {'published':>14} {'|diff|':>9}  "
          f"{'z* sorted (all)':>50}")
    print("-" * 110)
    ak = load_answer_key()
    for name in PROBLEMS:
        r = ak[name]
        slack = np.sort(np.array(r["slack"]))
        diff = abs(r["clarabel_obj"] - r["published"])
        ok = "OK" if diff <= 1e-4 * max(1.0, abs(r["published"])) else "CHECK"
        # show first few smallest and largest slacks
        small = ", ".join(f"{v:.1e}" for v in slack[:3])
        big = ", ".join(f"{v:.1e}" for v in slack[-3:])
        print(f"{name:10s} {r['p']:>4} | "
              f"{r['clarabel_obj']:>14.10f} {r['published']:>14.10f} {diff:>9.1e}  "
              f"[{small}, ..., {big}]  {ok}")
