import numpy as np
import jax

import scipy.optimize as opt
import cvxpy as cp
import gurobipy as gp
from jaxopt import OSQP
from linrax import linprog

from typing import Callable
import time


def timed(f: Callable):
    def f_timed(*args, **kwargs):
        t0 = time.time()
        ret = jax.block_until_ready(f(*args, **kwargs))
        tf = time.time()
        return ret, (tf - t0)

    return f_timed


def run_times(N: int, f: Callable, *args, **kwargs):
    f_timed = timed(f)
    times = []
    for i in range(N):
        ret, dt = f_timed(*args, **kwargs)
        times.append(dt)
    return ret, np.array(times)


# Generate a random non-trivial linear program.
m = 20
n = 15
np.random.seed(1)
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

x_cp = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T @ x_cp), [A @ x_cp <= b])

model = gp.Model("lp")
model.setParam("OutputFlag", 0)
x_gb = model.addMVar(shape=n, name="x", lb=-gp.GRB.INFINITY)
model.setObjective(c @ x_gb, gp.GRB.MINIMIZE)
model.addConstr(A @ x_gb <= b)


def gb_solve():
    model.reset(1)
    model.optimize()
    sol = model.x
    return sol


Q = np.zeros((c.size, c.size))
jaxopt_solver = OSQP()


@jax.jit
def jaxopt_solve(A, b, c):
    return jaxopt_solver.run(params_obj=(Q, c), params_ineq=(A, b))


sp_sol, sp_times = run_times(100, opt.linprog, c=c, A_ub=A, b_ub=b, bounds=(None, None))
print(f"Scipy took {sp_times.mean():.4g} ± {sp_times.std():.4g}s")
cp_sol, cp_times = run_times(100, prob.solve)
print(f"Cvxpy took {cp_times.mean():.4g} ± {cp_times.std():.4g}s")
gb_sol, gb_times = run_times(100, gb_solve)
print(f"Gurobi took {gb_times.mean():.4g} ± {gb_times.std():.4g}s")
jaxopt_solve(A, b, c)  # JIT compile
jaxopt_sol, jaxopt_times = run_times(100, jaxopt_solve, A, b, c)
print(f"Jaxopt took {jaxopt_times.mean():.4g} ± {jaxopt_times.std():.4g}s")
linprog(c=c, A_ub=A, b_ub=b, unbounded=True)  # JIT compile
my_sol, my_times = run_times(100, linprog, c, A_ub=A, b_ub=b, unbounded=True)
print(f"Mine took  {my_times.mean():.4g} ± {my_times.std():.4g}s")
