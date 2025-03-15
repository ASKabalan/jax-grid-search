from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
    import jax._src.interpreters.ad

import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu

from jax_grid_search import ProgressBar, optimize


# Define a simple quadratic function: f(x) = (x - 3)^2
def quadratic(x: "jax._src.interpreters.ad.JVPTracer") -> "jax._src.interpreters.ad.JVPTracer":
    return jnp.sum((x - 3.0) ** 2)


def test_optimize_quadratic() -> None:
    # Run the optimizer with non-verbose mode.

    with ProgressBar() as p:
        init_params = jnp.array([0.0])
        solver = optax.lbfgs()
        final_params, final_state = optimize(
            init_params,
            quadratic,
            solver,
            max_iter=50,
            tol=1e-4,
            progress=p,
        )

    # The minimum of (x-3)^2 is at x=3.0.
    np.testing.assert_allclose(final_params, jnp.array([3.0]), atol=1e-2)

    # The objective value should be near 0 at the minimum.
    final_value = quadratic(final_params)
    np.testing.assert_allclose(final_value, 0.0, atol=1e-2)

    # Check that the optimizer state shows that at least one iteration was performed.
    count = otu.tree_get(final_state.state, "count")
    assert count is not None and count > 0
