from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jaxtyping import Array, PyTree


class OptimizerState(NamedTuple):
    """State for Newton-CG optimizer."""

    params: PyTree
    state: PyTree
    updates: PyTree
    value: float
    best_val: float
    best_params: PyTree


def _debug_callback(
    update_norm: Array,
    iter_num: Array,
    value: Array,
    max_iters: Array,
    log_interval: Array,
) -> None:
    if iter_num == 0 or iter_num % int(max_iters * log_interval) == 0:
        print(f"update norm {update_norm} at iter {iter_num} value {value}")


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 6))
def optimize(
    init_params: Array,
    fun: Callable[[Array], Array],
    opt: optax._src.base.GradientTransformationExtraArgs,
    max_iter: int,
    tol: Array,
    verbose: bool = False,
    log_interval: float = 0.1,
    **kwargs: Any,
) -> tuple[Array, OptimizerState]:
    # Define a function that computes both value and gradient of the objective.
    value_and_grad_fun = jax.value_and_grad(fun)

    # Single optimization step.
    def step(carry: OptimizerState) -> OptimizerState:
        value, grad = value_and_grad_fun(carry.params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(grad, carry.state, carry.params, value=carry.value, grad=grad, value_fn=fun, **kwargs)  # Perform update
        params = optax.apply_updates(carry.params, updates)  # Update params

        best_params = jax.tree.map(
            lambda x, y: jnp.where(carry.best_val < value, x, y),
            carry.best_params,
            carry.params,
        )
        best_val = jnp.where(carry.best_val < value, carry.best_val, value)
        return carry._replace(
            params=params,
            state=state,
            updates=updates,
            value=value,
            best_val=best_val,
            best_params=best_params,
        )

    # Stopping condition.
    def continuing_criterion(carry: OptimizerState) -> Any:
        iter_num = otu.tree_get(carry.state, "count")  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = otu.tree_l2_norm(carry.updates)  # Compute update norm
        if verbose:
            jax.debug.callback(_debug_callback, update_norm, iter_num, carry.value, max_iter, log_interval)
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_state = OptimizerState(init_params, opt.init(init_params), init_params, jnp.inf, jnp.inf, init_params)

    # Run the while loop.
    final_opt_state = jax.lax.while_loop(continuing_criterion, step, init_state)

    # Was the last evaluation better than the best?
    best_params = jax.tree.map(
        lambda x, y: jnp.where(final_opt_state.best_val < final_opt_state.value, x, y),
        final_opt_state.best_params,
        final_opt_state.params,
    )
    best_value: float = jnp.where(
        final_opt_state.best_val < final_opt_state.value,
        final_opt_state.best_val,
        final_opt_state.value,
    )  # type: ignore[assignment]
    final_opt_state = final_opt_state._replace(best_params=best_params, best_val=best_value)

    return final_opt_state.best_params, final_opt_state
