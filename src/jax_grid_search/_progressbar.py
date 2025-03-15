import sys
from typing import Any, Callable, Dict, Optional

import jax
from jax.debug import callback
from rich.progress import Progress, Task, TaskID

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from jaxtyping import Array, PyTree


def _base_cb(id: int, arguments: PyTree[Array]) -> Optional[str]:
    return None


@jax.tree_util.register_static
class ProgressBar:
    def __init__(self: Self, *args: None, **kwargs: None) -> None:
        self.tasks: Dict[TaskID, Task] = {}
        self.progress = Progress(*args, **kwargs)
        self.progress.start()

    def create_task(self: Self, id: int, total: int) -> None:
        # Add a new task to the progress bar
        def _create_task(id: Array, total: Array) -> None:
            id = int(id)  # type: ignore[assignment]
            if id not in self.tasks:
                self.tasks[id] = self.progress.add_task(f"Running {id}...", total=total)
            else:
                # Reset
                self.progress.reset(self.tasks[id], total=total, start=True)

        return callback(_create_task, id, total, ordered=True)

    def update(
        self: Self,
        idx: int,
        arguments: PyTree[Array],
        desc_cb: Callable[[int, Any], Optional[str]] = _base_cb,
        total: int = 100,
        n: int = 1,
    ) -> None:
        # Update by n steps (by default, one iteration)
        def _update_task(idx: int, total: int, arguments: PyTree[Array]) -> None:
            idx = int(idx)
            if idx not in self.tasks:
                self.create_task(idx, total)
            desc = desc_cb(idx, arguments)
            self.progress.update(self.tasks[idx], advance=n, description=desc)

        return callback(_update_task, idx, total, arguments, ordered=True)

    def finish(self: Self, id: int, total: int) -> None:
        # Mark the progress as complete
        def _finish_task(id: int, total: int) -> None:
            id = int(id)
            if id not in self.tasks:
                self.create_task(id, total)
            self.progress.update(self.tasks[id], completed=total)

        return callback(_finish_task, id, total, ordered=True)

    def close(self) -> None:
        self.progress.stop()

    def __del__(self) -> None:
        self.progress.stop()

    def __enter__(self: Self) -> Self:
        self.progress.__enter__()
        return self

    def __exit__(self: Self, exc_type: type[RuntimeError], exc_value: RuntimeError, traceback: Any) -> Any:
        return self.progress.__exit__(exc_type, exc_value, traceback)
