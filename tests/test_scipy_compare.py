import jax
import jax.numpy as jnp
import scipy.optimize as opt
from typing import Tuple

import pytest

from linrax import linprog, SimplexStep, SimplexSolutionType


def compare(
    my_ans: Tuple[SimplexStep, SimplexSolutionType], sp_ans
) -> Tuple[bool, str]:
    my_sol, my_sol_type = my_ans
    if sp_ans.status == 2:
        if not my_sol_type.feasible:
            return True, "SUCCESS: problem is infeasible"
        else:
            return False, "FAILURE: we did not detect problem as infeasible"
    elif sp_ans.status == 3:
        if not my_sol_type.bounded:
            return True, "SUCCESS: problem is unbounded"
        else:
            return False, "FAILURE: we did not detect problem as unbounded"
    elif sp_ans.status == 0:
        if my_sol_type.success:
            correct = jnp.allclose(my_sol.fun, sp_ans.fun, atol=1e-7)

            if correct:
                return True, f"SUCCESS: x={my_sol.x}"
            else:
                return False, "FAILURE: objective value does not match"
        else:
            if not my_sol_type.feasible:
                return False, "FAILURE: we incorrectly identified problem as infeasible"
            elif not my_sol_type.bounded:
                return False, "FAILURE: we incorrectly identified problem as unbounded"

    return False, "FAILURE: unknown status"


def verify(
    c: jax.Array,
    A_ub: jax.Array = jnp.empty((0, 0)),
    b_ub: jax.Array = jnp.empty((0,)),
    A_eq: jax.Array = jnp.empty((0, 0)),
    b_eq: jax.Array = jnp.empty((0,)),
    unbounded: bool = False,
):
    my_sol = linprog(c, A_ub, b_ub, A_eq, b_eq, unbounded)

    bounds = (None, None) if unbounded else (0, None)
    sp_A_eq = A_eq if A_eq.size > 0 else None
    sp_b_eq = b_eq if b_eq.size > 0 else None
    sp_A_ub = A_ub if A_ub.size > 0 else None
    sp_b_ub = b_ub if b_ub.size > 0 else None
    sp_sol = opt.linprog(c, sp_A_ub, sp_b_ub, sp_A_eq, sp_b_eq, bounds=bounds)

    res = compare(my_sol, sp_sol)
    assert res[0], res[1]


test_cases = []


A = jnp.array(
    [
        [1, -1],
        [3, 2],
        [1, 0],
        [-2, 3],
    ]
)
b = jnp.array([1, 12, 2, 9])
c = jnp.array([-4, -2])
test_cases.append({"c": c, "A_ub": A, "b_ub": b})


A = jnp.array(
    [
        [1],
    ]
)
b = jnp.array([10])
c = jnp.array([1])
test_cases.append({"c": c, "A_ub": A, "b_ub": b, "unbounded": True})

A = jnp.array(
    [
        [-1],
    ]
)
b = jnp.array([-10])
c = jnp.array([-1])
test_cases.append({"c": c, "A_ub": A, "b_ub": b})

A = jnp.array(
    [
        [1, -1],
        [-1, 1],
    ]
)
b = jnp.array([1, -2])
c = jnp.array([-4, -2])
test_cases.append({"c": c, "A_ub": A, "b_ub": b})

A = jnp.array(
    [
        [1.0, 1, -1, -1, 0, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, 0, 0],
        [0, 1, 0, -1, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, 0, -1, 0],
        [0, -1, 0, 1, 0, 0, 0, 1],
    ]
)
b = jnp.array([1.2, 1.1, 0.1, 0.9, 0.1])
c = jnp.array([1.0, 0, -1, 0, 0, 0, 0, 0])
test_cases.append({"c": c, "A_eq": A, "b_eq": b})

# Programs with dependent constraints

A = jnp.array([[1.0, 0], [1, 0]])
b = jnp.array([3.0, 3])
c = jnp.array([-1.0, 0])
test_cases.append({"c": c, "A_ub": A, "b_ub": b})

A_eq = jnp.array([[1.0, 2]])
b_eq = jnp.array([1.0])
A_ub = jnp.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])
b_ub = jnp.array([0.9, 0.1, -0.9, 0.1])
c = jnp.array([0.0, 1])
test_cases.append(
    {"c": c, "A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub, "unbounded": True}
)

A_eq = jnp.array([[0.5, 0.5]])
b_eq = jnp.array([0.4])
A_ub = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -0.0], [-0.0, -1.0]])
b_ub = jnp.array([0.9, 0.1, -0.9, 0.1])
c = -jnp.array([0.0, 1.0])
test_cases.append(
    {"c": c, "A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub, "unbounded": True}
)

# Programs based on aux var refinement
N = 6
aux_vars = jnp.array(
    [
        [jnp.cos(n * jnp.pi / (N + 1)), jnp.sin(n * jnp.pi / (N + 1))]
        for n in range(1, N + 1)
    ]
)
H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
Hs = [jnp.vstack((H, aux_vars[: i + 1])) for i in range(N)]

A_eq = jnp.array([aux_vars[0]])
b_eq = jnp.dot(aux_vars[0], jnp.array([1.1, 0.1])).reshape(-1)
A_ub = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
b_ub = jnp.array([1.1, 0.1, -1.1, -0.1])
c = aux_vars[0]
test_cases.append(
    {"c": c, "A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub, "unbounded": True}
)


@pytest.mark.parametrize("kwargs", test_cases)
def test_solutions(kwargs):
    verify(**kwargs)
