# linrax

JAX-compatible, simplex method-based linear program solver. As part of the JAX ecosystem, `linrax` supports

- JIT compilation,
- Automatic Differentiability (forward mode only, currently), and
- GPU parallelization.

`linrax` is designed for use as a subroutine in a larger JAX pipeline. Its performance excels on smaller problems ($<50$ input variables), and is fully tracable in any of JAX's main transformations. In particular, `linrax` can solve problems that are specified with linearly dependent constraints, an area where other JAX-based solvers struggle. 

## Usage

The interface of `linrax` is designed to closely mimic that of `scipy.linprog`. 
The public function is 
```python
import jax 
import jax.numpy as jnp
@partial(jax.jit, static_argnames=[
        "unbounded"])
def linprog(
    c: jax.Array,
    A_ub: jax.Array = jnp.empty((0, 0)),
    b_ub: jax.Array = jnp.empty((0,)),
    A_eq: jax.Array = jnp.empty((0, 0)),
    b_eq: jax.Array = jnp.empty((0,)),
    unbounded: bool = False,
) -> Tuple[SimplexStep, 
        SimplexSolutionType]:
    ...
```

The `SimplexSolutionType` contains fields indicating if the problem is `feasible` or `bounded`, and a `success` property to check both simultaneously. 
Assuming the problem has solutions, the `SimplexStep` object describes this solution. In particular, the fields `x` and `fun` retrieve the optimal point and objective value, respectively. 

<!-- ## Citing

If `linrax` is useful or relevant for your academic work, please cite the corresponding paper with this bibtex entry. 
```
asdf
``` -->
