import os
import shutil
from typing import Callable, Iterator

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jax_grid_search import DistributedGridSearch


# Automatically clean up the "results" directory before and after each test.
@pytest.fixture(autouse=True)
def clean_results_dir() -> Iterator[None]:
    results_dir = "results"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    yield
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)


# Fixture for the objective function used by grid search.
@pytest.fixture
def objective_function() -> Callable[[Array, Array, Array, Array], dict[str, Array]]:
    def objective_function(
        x: Array,
        y: Array,
        z: Array,
        w: Array,
    ) -> dict[str, Array]:
        value = x**2 + y**2 + z**2 - w**2
        return {"value": value.sum()}

    return objective_function


# Fixture for the default search space.
@pytest.fixture
def search_space() -> dict[str, Array]:
    return {
        "x": jnp.arange(4).reshape(2, 2),
        "y": jnp.arange(4).reshape(2, 2),
        "z": jnp.arange(4).reshape(2, 2),
        "w": jnp.arange(4).reshape(2, 2),
    }


# Fixture for an updated search space.
@pytest.fixture
def updated_search_space() -> dict[str, Array]:
    return {
        "x": jnp.arange(6).reshape(3, 2),
        "y": jnp.arange(6).reshape(3, 2),
        "z": jnp.arange(4).reshape(2, 2),
        "w": jnp.arange(4).reshape(2, 2),
    }


def test_grid_search(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
) -> None:
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=8, progress_bar=True, log_every=0.1)
    grid_search.run()

    results = grid_search.stack_results("results")
    values = results["value"]

    # Assert that the first value is the minimum.
    assert values[0] == jnp.min(values)

    best_x = results["x"][0]
    best_y = results["y"][0]
    best_z = results["z"][0]
    best_w = results["w"][0]

    # Check that the objective function returns the expected value.
    assert jnp.min(values) == objective_function(best_x, best_y, best_z, best_w)["value"]


def test_resume(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
    updated_search_space: dict[str, Array],
) -> None:
    # First run with the initial search space.
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=8, progress_bar=True, log_every=0.1)
    expected_n_combinations = jax.tree.reduce(lambda x, y: x * y.shape[0], search_space, 1)
    assert grid_search.n_combinations == expected_n_combinations

    grid_search.run()
    full_results = grid_search.stack_results("results")

    # Now resume with the same search space.
    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=8,
        progress_bar=True,
        log_every=0.1,
        old_results=full_results,
    )
    # Since all combinations have been processed, we expect 0 remaining.
    assert grid_search.n_combinations == 0

    # Resume with an updated search space.
    new_expected_n_combinations = jax.tree.reduce(lambda x, y: x * y.shape[0], updated_search_space, 1)
    remaining_combinations = new_expected_n_combinations - expected_n_combinations

    grid_search = DistributedGridSearch(
        objective_function,
        updated_search_space,
        batch_size=8,
        progress_bar=True,
        log_every=0.1,
        old_results=full_results,
    )
    print(f"remaining_combinations: {remaining_combinations}")
    print(f"grid_search.n_combinations: {grid_search.n_combinations}")
    assert grid_search.n_combinations == remaining_combinations

    grid_search.run()

    full_results = grid_search.stack_results("results")
    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=8,
        progress_bar=True,
        log_every=0.1,
        old_results=full_results,
    )
    # After resuming, there should be no remaining combinations.
    assert grid_search.n_combinations == 0


def test_suggest_batch(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
) -> None:
    if jax.devices()[0].platform == "cpu":
        pytest.skip("Test only works for GPU devices")
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=None, progress_bar=True, log_every=0.1)

    max_size = grid_search.suggest_batch_size()

    memory_stats = jax.devices()[0].memory_stats()
    max_device_memory = memory_stats["bytes_limit"] - memory_stats["bytes_in_use"]

    # Size of one call.
    sample_params = jax.tree.map(lambda x: x[0], search_space)
    compiled = (
        jax.jit(objective_function)
        .lower(
            sample_params["x"],
            sample_params["y"],
            sample_params["z"],
            sample_params["w"],
        )
        .compile()
    )
    mem_analysis = compiled.memory_analysis()

    one_call_mem = mem_analysis.argument_size_in_bytes + mem_analysis.output_size_in_bytes + mem_analysis.temp_size_in_bytes

    assert (max_device_memory / one_call_mem) - max_size < 0.5


def test_bad_objective_fn(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
) -> None:
    def bad_objective_fn(
        x: Array,
        y: Array,
        z: Array,
        w: Array,
    ) -> Array:
        good_res = objective_function(x, y, z, w)
        return good_res["value"]  # Return only the value and not the dict

    grid_search = DistributedGridSearch(bad_objective_fn, search_space, batch_size=8, progress_bar=True, log_every=0.1)
    with pytest.raises(KeyError):
        grid_search.run()

    def no_val_objective_fn(
        x: Array,
        y: Array,
        z: Array,
        w: Array,
    ) -> dict[str, Array]:
        good_res = objective_function(x, y, z, w)
        return {"not_value": good_res["value"]}  # Return an unexpected key

    grid_search = DistributedGridSearch(no_val_objective_fn, search_space, batch_size=8, progress_bar=True, log_every=0.1)

    with pytest.raises(KeyError):
        grid_search.run()
