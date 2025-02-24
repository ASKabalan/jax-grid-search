import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.distributed.initialize()

import jax.numpy as jnp
from jax_grid_search import DistributedGridSearch


def objective_function(x , y , z , w):
    value =  x**2 + y**2 + z**2 - w**2
    value = value.sum()
    return {"value": value}

search_space = {"x": jnp.arange(4*2).reshape(4,2),
                "y": jnp.arange(4*2).reshape(4,2),
                "z": jnp.arange(4*2).reshape(4,2),
                "w": jnp.arange(4*2).reshape(4,2),
}


grid_search = DistributedGridSearch(
    objective_function, search_space, batch_size=30, progress_bar=True, log_every=0.1
)


grid_search.run()