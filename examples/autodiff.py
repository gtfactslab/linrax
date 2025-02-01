import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from linrax import linprog


print("Checking autodiff")
A = jnp.array([[1.0, 1.0], [-3.0, -1.0], [1.0, 0.0]])
b = jnp.array([6.0, -3.0, 3.0])
c = jnp.array([-2.0, -1.0])

sol, sol_type = linprog(c, A, b)

g = jax.jacfwd(linprog, has_aux=True, argnums=[0, 1, 2])
jac, jac_type = g(c, A, b)
dxdc, dxdA, dxdb = jac.x
dfdc, dfdA, dfdb = [arr[-1, -1] for arr in jac.tableau]
dbdc, dbdA, dbdb = jac.basis

dAs = jnp.array(
    [
        jnp.zeros_like(A).at[(i, j)].set(1)
        for i in range(A.shape[0])
        for j in range(A.shape[1])
    ]
)
print(f"Checking wrt A:\n{dxdA=},\n{dfdA=},\n{dbdA=}")
sol_fig, val_fig = None, None
sol_axs, val_axs = [], []
for dA in dAs / 10:
    Ap = A + dA
    solp, solp_type = linprog(c, Ap, b)

    linearized = sol.x + jnp.einsum("ijk, jk->i", dxdA, dA)
    if not jnp.allclose(linearized, solp.x):
        basis_changed = jnp.any(sol.basis != solp.basis)
        print(
            f"Linearized solution off by {jnp.linalg.norm(linearized - solp.x)} ({basis_changed=})"
        )

        sol_fig = plt.figure() if sol_fig is None else sol_fig
        n = len(sol_axs) + 1
        ax = sol_fig.add_subplot(1, 3, n, projection="3d")
        sol_axs.append(ax)

        idx = jnp.arange(-5, 6)
        xs = jnp.array([linprog(c, A + i * dA, b)[0].x for i in idx])
        x0s, x1s = xs.T
        lin_xs = jnp.array(
            [sol.x + jnp.einsum("ijk, jk->i", dxdA, i * dA) for i in idx]
        )
        lin_x0s, lin_x1s = lin_xs.T
        ax.scatter(idx, x0s, x1s, label="true")
        ax.scatter(idx, lin_x0s, lin_x1s, label="linearized")

        ax.set_xlabel("i")
        ax.set_ylabel("x0")
        ax.set_zlabel("x1")

    linearized = (
        sol.fun - jnp.einsum("jk, jk", dfdA, dA)
    )  # NOTE: I have to subtract here because arr[-1, -1] is the negative of the objective value
    if not jnp.allclose(linearized, solp.fun):
        basis_changed = jnp.any(sol.basis != solp.basis)
        print(
            f"WARN: linearized objective off by {jnp.linalg.norm(linearized - solp.fun)} ({basis_changed=})"
        )

        val_fig = plt.figure() if val_fig is None else val_fig
        n = len(val_axs) + 1
        ax = val_fig.add_subplot(1, 3, n)
        val_axs.append(ax)

        idx = jnp.arange(-5, 6)
        true_objs = jnp.array([linprog(c, A + i * dA, b)[0].fun for i in idx])
        lin_objs = jnp.array(
            [sol.fun - jnp.einsum("jk, jk", dfdA, i * dA) for i in idx]
        )
        ax.plot(idx, true_objs, label="true")
        ax.plot(idx, lin_objs, label="linearized")

        ax.set_xlabel("i")
        ax.set_ylabel("Objective Value")

    assert jnp.allclose(sol.basis + jnp.einsum("ijk, jk->i", dbdA, dA), solp.basis)

if sol_fig is not None:
    sol_fig.legend()
    sol_fig.suptitle("Optimal point (linearized vs true)")

if val_fig is not None:
    val_fig.legend()
    val_fig.suptitle("Objective value (linearized vs true)")

if sol_fig is not None or val_fig is not None:
    plt.show()

dbs = jnp.eye(3)
print(f"\nChecking wrt b:\n{dxdb=},\n{dfdb=},\n{dbdb=}")
for db in dbs / 10:
    bp = b + db
    solp, solp_type = linprog(c, A, bp)
    assert jnp.allclose(sol.x + dxdb @ db, solp.x)
    assert jnp.allclose(sol.fun - dfdb @ db, solp.fun)
    assert jnp.allclose(sol.basis + dbdb @ db, solp.basis)

dcs = jnp.eye(2)
print(f"\nChecking wrt c:\n{dxdc=},\n{dfdc=},\n{dbdc=}")
for dc in dcs / 10:
    cp = c + dc
    solp, solp_type = linprog(cp, A, b)
    assert jnp.allclose(sol.x + dxdc @ dc, solp.x)
    assert jnp.allclose(sol.fun - dfdc @ dc, solp.fun)
    assert jnp.allclose(sol.basis + dbdc @ dc, solp.basis)

print("\nChecking basis change point")
g = jax.jacfwd(linprog, has_aux=True, argnums=2)
print(f"Original:\nsol x: {sol.x}, fun: {sol.fun}, basis: {sol.basis}")

b -= 2.99999 * dbs[0]
solp, solp_type = linprog(c, A, b)
print(f"Before basis change:\nsolp: {solp.x}, funp: {solp.fun}, basisp: {solp.basis}")

b -= 0.00001 * dbs[0]
solp, solp_type = linprog(c, A, b)
print(f"At basis change:\nsolp: {solp.x}, funp: {solp.fun}, basisp: {solp.basis}")
jac, jac_type = g(c, A, b)
print(f"dxdb: {jac.x}, dfdb: {jac.tableau[-1, -1]}, dbdb: {jac.basis}")

b -= dbs[0]
solp, solp_type = linprog(c, A, b)
print(f"After basis change:\nsolp: {solp.x}, funp: {solp.fun}, basisp: {solp.basis}")
