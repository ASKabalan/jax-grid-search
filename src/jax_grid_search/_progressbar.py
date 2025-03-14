from functools import partial
from typing import Any, Callable, NamedTuple , Optional

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jaxtyping import Array, PyTree
import time
import jax.numpy as jnp
import jax
from rich.progress import Progress , BarColumn , TimeRemainingColumn
from jax.debug import callback


def _base_cb(id , _):
    return None

@jax.tree_util.register_static
class ProgressBar:
    def __init__(self, *args , **kwargs) -> None:
        self.tasks = {}
        self.progress = Progress(*args , **kwargs)
        self.progress.start()
    
    def create_task(self, id: int , total):
        # Add a new task to the progress bar
        def _create_task(id , total):
            id = int(id)
            if id not in self.tasks:
                self.tasks[id] = self.progress.add_task(f"Running {id}...", total=total)
            else:
                # Reset
                self.progress.reset(self.tasks[id], total=total, start=True)
        return callback(_create_task , id  , total, ordered=True )

    def update(self, idx: int , arguments , desc_cb = _base_cb , total=100, n=1) -> None:
        # Update by n steps (by default, one iteration)
        def _update_task(idx, total, arguments):
            idx = int(idx)
            if idx not in self.tasks:
                self.create_task(idx , total)
            desc = desc_cb (idx , arguments)
            self.progress.update(self.tasks[idx], advance=n , description=desc)
        return callback(_update_task , idx , total, arguments,  ordered=True)
    
    def finish(self, id: int, total) -> None:
        # Mark the progress as complete
        def _finish_task(id, total):
            id = int(id)
            if id not in self.tasks:
                self.create_task(id)
            self.progress.update(self.tasks[id], completed=total)
        return callback(_finish_task , id , total, ordered=True)

    def close(self):
        self.progress.stop()

    def __del__(self):
        self.progress.stop()

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.progress.__exit__(exc_type, exc_value, traceback)