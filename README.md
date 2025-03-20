# Distributed Grid Search & Continuous Optimization using JAX

[![Testing](https://github.com/ASKabalan/jax-grid-search/actions/workflows/tests.yml/badge.svg)](https://github.com/ASKabalan/jax-grid-search/actions/workflows/tests.yml)
[![Code Formatting](https://github.com/ASKabalan/jax-grid-search/actions/workflows/formatting.yml/badge.svg)](https://github.com/ASKabalan/jax-grid-search/actions/workflows/formatting.yml)
[![Upload Python Package](https://github.com/ASKabalan/jax-grid-search/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ASKabalan/jax-grid-search/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/jax-grid-search.svg)](https://badge.fury.io/py/jax-grid-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides two complementary optimization tools:

1. **Distributed Grid Search for Discrete Optimization:**
   Explore a parameter space by evaluating a user-defined objective function on a grid of discrete values. The search runs in parallel across available processes, automatically handling batching, progress tracking, and result aggregation.

2. **Continuous Optimization with Optax:**
   Minimize continuous functions using gradient-based methods (such as LBFGS). This routine leverages Optax for iterative parameter updates and includes built-in progress monitoring.

---

## Getting Started

### Installation

Install the required dependencies via pip:

```bash
pip install jax_grid_search
```

---

## Usage Examples

### 1. Distributed Grid Search (Discrete Optimization)

Define your objective function and parameter grid, then run a distributed grid search. The objective function must return a dictionary with a `"value"` key.

```python
import jax.numpy as jnp
from jax_grid_search import DistributedGridSearch

# Define a discrete objective function
def objective_fn(param1, param2):
    # Example: combine sine and cosine evaluations
    result = jnp.sin(param1) + jnp.cos(param2)
    return {"value": result}

# Define the search space (discrete values)
search_space = {
    "param1": jnp.linspace(0, 3.14, 10),
    "param2": jnp.linspace(0, 3.14, 10)
}

# Initialize and run the grid search
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    progress_bar=True,     # Enable progress updates
    log_every=0.1,         # Log progress every 10%
    result_dir="results"   # Directory for intermediate results
)
grid_search.run()

# Retrieve the aggregated results
results = grid_search.stack_results("results")
print("Grid Search Results:", results)
```

#### Resuming a Grid Search

To resume a grid search from a previous checkpoint, simply load the results and pass them to the `DistributedGridSearch` constructor:

```python

results = grid_search.stack_results("results")

# Initialize and run the grid search
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    progress_bar=True,     # Enable progress updates
    log_every=0.1,         # Log progress every 10%
    result_dir="results"   # Directory for intermediate results
    old_results=results    # Pass the previous results to resume the search
)
grid_search.run()
```

#### Running a distributed grid search

To run the grid search across multiple processes, use the mpirun (or srun):

```bash
mpirun -n 4 python grid_search_example.py
```

To run the following code in script

```python
import jax
jax.distributed.initialize()


# Initialize and run the grid search
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    progress_bar=True,     # Enable progress updates
    log_every=0.1,         # Log progress every 10%
    result_dir="results"   # Directory for intermediate results
    old_results=results    # Pass the previous results to resume the search
)
grid_search.run()
```

You need to make sure that the number of combinitions in the search space is divisible by the number of processes.


### 2. Continuous Optimization using Optax

Use the continuous optimization routine to minimize a function with gradient-based methods (e.g., LBFGS). The example below minimizes a simple quadratic function.

```python
import jax.numpy as jnp
import optax
from jax_grid_search import optimize , ProgressBar

# Define a continuous objective function (e.g., quadratic)
def quadratic(x):
    return jnp.sum((x - 3.0) ** 2)

# Initial parameters and an optimizer (e.g., LBFGS)
init_params = jnp.array([0.0])
optimizer = optax.lbfgs()

with ProgressBar() as p:
    # Run continuous optimization with progress monitoring (optional)
    best_params, opt_state = optimize(
        init_params,
        quadratic,
        opt=optimizer,
        max_iter=50,
        tol=1e-10,
        progress=p  # Replace with a ProgressBar instance for visual updates if desired
)

print("Optimized Parameters:", best_params)
```

#### Running multiple optimization tasks with vmap

You can run multiple optimization tasks in parallel using `jax.vmap`. This is useful when optimizing multiple functions or parameters simultaneously.

(This is very usefull for simulating multiple noise realizations for example)

You can use `progress_id` to track the progress of each optimization task running in parallel.

```python
import jax
import jax.numpy as jnp
import optax

# Define multiple objective functions
def objective_fn(x , normal):
    return jnp.sum(((x - 3.0) ** 2) + normal)

with ProgressBar() as p:

    def solve_one(seed):
        init_params = jnp.array([0.0])
        normal = jax.random.normal(jax.random.PRNGKey(seed), init_params.shape)
        optimizer = optax.lbfgs()
        # Run continuous optimization with progress monitoring (optional)
        best_params, opt_state = optimize(
            init_params,
            objective_fn,
            opt=optimizer,
            max_iter=50,
            tol=1e-4,
            progress=p,
            progress_id=seed,
            normal=normal
        )

        return best_params

    jax.vmap(solve_one)(jnp.arange(10))

```
### 3. Optimizing Likelihood parameters and models

You can use the continuous optimization to optimize the parameters of a model that is defined in a function.
For performance purposes, you need to make sure that the discrete parameters that can control the likelihood model can be jitted (using `lax.cond` for example or other lax control flow functions).
