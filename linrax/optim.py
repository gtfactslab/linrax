import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from typing import Tuple
from functools import partial


@register_pytree_node_class
class SimplexStep:
    tableau: jax.Array
    basis: jax.Array
    x: jax.Array

    def __init__(
        self,
        tableau,
        basis,
        x,
    ):
        self.tableau = tableau
        self.basis = basis
        self.x = x

    def tree_flatten(self):
        return (
            (
                self.tableau,
                self.basis,
                self.x,
            ),
            "SimplexStep",
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def fun(self) -> jax.Array:
        return -self.tableau[-1, -1]

    def __repr__(self) -> str:
        return f"SimplexStep(tableau={self.tableau}, basis={self.basis}, x={self.x})"

    def __str__(self) -> str:
        def shapes(tup):
            if hasattr(tup, "shape"):
                return str(tup.shape)
            return (
                "("
                + ", ".join([f"{i}: {item.shape}" for i, item in enumerate(tup)])
                + ")"
            )

        return f"SimplexStep(tableau={shapes(self.tableau)}, basis={shapes(self.basis)}, x={shapes(self.x)})"


@register_pytree_node_class
class SimplexSolutionType:
    feasible: jax.Array
    bounded: jax.Array

    def __init__(self, feasible: jax.Array, bounded: jax.Array):
        self.feasible = feasible
        self.bounded = bounded

    def tree_flatten(self):
        return (
            (
                self.feasible,
                self.bounded,
            ),
            "SimplexSolutionType",
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def success(self) -> jax.Array:
        return jnp.logical_and(self.feasible, self.bounded)

    def __repr__(self) -> str:
        return f"SimplexSolutionType(feasible={self.feasible}, bounded={self.bounded})"


def _fuzzy_argmin(arr: jax.Array, tolerance: float = 1e-5) -> jax.Array:
    min_val = jnp.min(arr)
    within_tolerance = jnp.abs(arr - min_val) <= tolerance
    indices = jnp.arange(arr.shape[0])
    valid_indices = jnp.where(within_tolerance, indices, jnp.inf)
    return jnp.argmin(valid_indices)


@partial(jax.jit, static_argnames=["unbounded"])
def _standard_form(
    obj: jax.Array,
    A_eq: jax.Array,
    b_eq: jax.Array,
    A_ub: jax.Array,
    b_ub: jax.Array,
    unbounded: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    if unbounded:
        # Convert from unbounded vars to non-negative vars
        obj = jnp.concatenate((obj, -obj))
        A_eq = jnp.hstack((A_eq, -A_eq))
        A_ub = jnp.hstack((A_ub, -A_ub))

    # Ensure RHS of equality constraints are positive
    idx = jnp.less(b_eq, 0).repeat(A_eq.shape[1]).reshape(A_eq.shape)
    A_eq = jax.lax.select(idx, -A_eq, A_eq)
    A_eq = A_eq.reshape((A_eq.shape[0], obj.shape[0]))  # hack to make stacking work
    b_eq = jnp.abs(b_eq)

    # Add slack vars to turn inequalities into equalities
    obj = jnp.concatenate((obj, jnp.zeros_like(b_ub)))
    A_eq = jnp.hstack((A_eq, jnp.zeros((A_eq.shape[0], b_ub.shape[0]))))

    idx = jnp.less(b_ub, 0)
    slack_signs = jax.lax.select(
        idx, -jnp.ones_like(b_ub), jnp.ones_like(b_ub)
    )  # track direction of inequalities
    idx = idx.repeat(A_ub.shape[1]).reshape(A_ub.shape)
    A_ub = jax.lax.select(idx, -A_ub, A_ub)  # flip signs to make RHS positive
    b_ub = jnp.abs(b_ub)

    A_ub = jnp.hstack((A_ub, jnp.diag(slack_signs)))
    A_ub = A_ub.reshape((A_ub.shape[0], obj.shape[0]))  # hack to make stacking work

    A = jnp.vstack((A_eq, A_ub))
    b = jnp.concatenate((b_eq, b_ub))

    return (A, b, obj)


def _iteration_needed(
    simplex_data: Tuple[SimplexStep, SimplexSolutionType],
) -> jax.Array:
    step, sol_type = simplex_data

    _iteration_needed = jnp.less(
        step.tableau[-1, :-1], -1e-5
    ).any()  # Stop when optimal solution found
    _iteration_needed = jnp.logical_and(
        _iteration_needed, sol_type.feasible
    )  # Stop if infeasible
    _iteration_needed = jnp.logical_and(
        _iteration_needed, sol_type.bounded
    )  # Stop if unbounded
    return _iteration_needed[0]


def _simplex(
    step: SimplexStep, sol_type: SimplexSolutionType, num_cost_rows: int
) -> Tuple[SimplexStep, SimplexSolutionType]:
    def pivot(
        simplex_data: Tuple[SimplexStep, SimplexSolutionType],
    ) -> Tuple[SimplexStep, SimplexSolutionType]:
        step, sol_type = simplex_data
        tableau = step.tableau

        # Find entering variable (with Bland's rule)
        neg_cost_mul_idx = jnp.where(
            tableau[-1, :-1] < -1e-5,
            size=tableau.shape[1] - 1,
            fill_value=tableau.shape[1] - 1,
        )[0]
        entering_col = jnp.min(neg_cost_mul_idx)

        # Find exiting variable / pivot row
        exiting_rates = tableau[:-num_cost_rows, entering_col]
        div = jnp.divide(tableau[:-num_cost_rows, -1], exiting_rates)
        ratios = jax.lax.select(
            jnp.greater(exiting_rates, 1e-5), div, jnp.inf * jnp.ones_like(div)
        )  # Don't worry about constraints that entering var improves / doesn't affect
        exiting_row = _fuzzy_argmin(ratios)
        sol_type.bounded = jnp.any(exiting_rates > -1e-5).reshape(1)

        # Pivot
        pivot_val = tableau[exiting_row, entering_col]
        tableau = tableau.at[exiting_row].set(
            tableau[exiting_row] / pivot_val
        )  # normalize pivot val to 1
        other_rows = jnp.setdiff1d(
            jnp.arange(tableau.shape[0]),
            exiting_row,
            assume_unique=True,
            size=tableau.shape[0] - 1,
        )
        for i in other_rows:
            tableau = tableau.at[i].set(
                tableau[i] - tableau[i, entering_col] * tableau[exiting_row]
            )

        # Update basis set, BFS
        basis = step.basis.at[exiting_row].set(entering_col)
        x = jnp.zeros_like(step.x)
        x = x.at[basis].set(tableau[:-num_cost_rows, -1])

        return SimplexStep(tableau, basis, x), sol_type

    # NOTE: this looses reverse mode autodiff. We can accomplish all the same calculations
    # with forward mode, which this does not lose, but they might be slightly less efficient.
    step, sol_type = jax.lax.while_loop(_iteration_needed, pivot, (step, sol_type))

    # Uncomment to debug
    # while _iteration_needed((step, sol_type)):
    #     step, sol_type = pivot((step, sol_type))

    return step, sol_type


@partial(jax.jit, static_argnames=["unbounded"])
def linprog(
    obj: jax.Array,
    A_ub: jax.Array = jnp.empty((0, 0)),
    b_ub: jax.Array = jnp.empty((0,)),
    A_eq: jax.Array = jnp.empty((0, 0)),
    b_eq: jax.Array = jnp.empty((0,)),
    unbounded: bool = False,
) -> Tuple[SimplexStep, SimplexSolutionType]:
    """
    Solves a linear program of the form: min c @ x s.t. A_eq @ x = b_eq, A_ub @ x <= b_ub

    Args:
        obj: The coefficients of the linear function to minimize
        A_eq: Equality constraint matrix
        b_eq: Equality constraint vector
        A_ub: Inequality constraint matrix
        b_ub: Inequality constraint vector
        unbounded: If False (default), only considers x >= 0. If True, will consider all x

    Returns:
        The vector x that minimizes c @ x subject to the constraints given.
    """
    A, b, c = _standard_form(obj, A_eq, b_eq, A_ub, b_ub, unbounded)

    # _simplex assumes that the last A.shape[0] variables form a feasible basis for the problem
    # This is not true in general (e.g. for problems with lots of equality constraints)
    # Therefore, we first solve a problem with auxiliary variables to find a feasible basis
    tableau = jnp.hstack((A, jnp.eye(A.shape[0]), b.reshape(-1, 1)))
    c_extended = jnp.concatenate((c, jnp.zeros(A.shape[0] + 1)))
    c_aux = jnp.concatenate((jnp.zeros_like(c), jnp.ones(A.shape[0]), jnp.zeros(1)))
    tableau = jnp.vstack((tableau, c_extended, c_aux))

    # Zero out reduced cost muls of initial basis
    for i in range(A.shape[0]):
        tableau = tableau.at[-1].set(tableau[-1] - tableau[i])

    basis = jnp.arange(A.shape[1], A.shape[1] + A.shape[0])
    x = jnp.concatenate((jnp.zeros_like(c), b))
    aux_start = SimplexStep(tableau, basis, x)
    aux_sol_type = SimplexSolutionType(jnp.array([True]), jnp.array([True]))
    aux_sol, aux_sol_type = _simplex(aux_start, aux_sol_type, num_cost_rows=2)
    x = aux_sol.x[: c.size]

    # Remove auxiliary variables from tableau for real problem
    tableau = aux_sol.tableau[:-1, :]
    tableau = jnp.delete(
        tableau,
        jnp.arange(A.shape[1], A.shape[1] + A.shape[0]),
        axis=1,
        assume_unique_indices=True,
    )

    redundant_cons = jnp.concatenate((aux_sol.basis > A.shape[1], jnp.array([False])))
    redundant_cons = redundant_cons.repeat(tableau.shape[1]).reshape(tableau.shape)
    # TODO: I don't know if this is correct in general. Really, what I want to do is
    # make the column corresponding to the auxiliary variable basic, but I'm not sure
    # of an efficient way to identify this column
    tableau = jnp.where(redundant_cons, -tableau, tableau)

    # NOTE: The tolerance here determines how much violation of constraints we are comfortable with.
    # The default 1e-9 seems to be too small, especially for the problems we generate with aux-var
    # refinement when collapsing along a face. Along auxilliary faces, this selects exactly a corner
    # of the region in real variables. Logically, this should be fine, but numerically it causes the
    # problem to be marked as infeasible, and we get no refinement just when we should get the most.
    feasible = jnp.allclose(c_aux[:-1] @ aux_sol.x, jnp.array([0]), atol=1e-3).reshape(
        1
    )

    # I can increase the above tolerance a lot, but the bigger it gets the more I allow
    # solutions to violate constraints. This directly makes my refinement worse (by considering
    # points outside the feasible region). I am confident that I find the right set of basic
    # variables, but error propagated over all the row operations with large tolerances can
    # make the solution wrong. Really what I want to do is project this final point back
    # onto the feasible subspace.

    # To do this, I build the (n-m) x n matrix encoding the constraint that every non-basic
    # variable is zero. Then, vstack A on top of this matrix, b on top of a vector of 0s,
    # and solve the system Ax = b. Since I can't guarantee the rows of A are independent,
    # I can't invert A, but I might be able to pinv

    real_start = SimplexStep(
        tableau,
        aux_sol.basis,
        aux_sol.x[: c.size],
    )
    real_sol_type = SimplexSolutionType(feasible, jnp.array([True]))
    sol, sol_type = _simplex(real_start, real_sol_type, num_cost_rows=1)

    # Remove synthetic variables from returned result
    if unbounded:
        real_pos, real_neg, _ = jnp.split(sol.x, (len(obj), 2 * len(obj)))
        sol.x = real_pos - real_neg
    else:
        sol.x, _ = jnp.split(sol.x, (len(obj),))

    return sol, sol_type


