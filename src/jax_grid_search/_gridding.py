import glob
import itertools
import logging
import os
import pickle
import sys
from typing import Any, Callable, Dict, Iterator, Optional

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy import dtype, ndarray
from scipy.interpolate import interp1d
from tqdm import tqdm

logger = logging.getLogger("GRIDDING")


class DistributedGridSearch:
    def __init__(
        self: Self,
        objective_fn: Callable[..., Dict[str, Array]],
        search_space: Dict[str, Array],
        batch_size: Optional[int] = None,
        memory_limit: float = 0.6,
        log_every: float = 0.1,
        progress_bar: bool = True,
        result_dir: str = "results",
        old_results: Optional[Dict[str, Array]] = None,
    ) -> None:
        """
        Initialize the grid search.

        Args:
            objective_fn: The objective function to be evaluated.
            search_space: A dictionary where keys are parameter names and values are lists
                of possible values.
            batch_size: The number of combinations to evaluate in each batch.
                If None, it is determined automatically.
            memory_limit: Fraction of device memory to use for determining batch size.
            verbose: Percentage (0.0 to 1.0) to control logging frequency.
                Logs every 'verbose' fraction of progress.
            use_tqdm: Whether to use tqdm for a progress bar.
            result_dir: Directory to save batch results.
        """
        keys, values = zip(*search_space.items())

        self.param_keys = keys
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.result_dir = result_dir

        # Create an iterator over all parameter combinations
        self.combinations = list(itertools.product(*values))

        if old_results is not None:
            self.reduce_search_space(search_space, old_results)

        self.batch_idx = self.last_batch_idx(self.result_dir)
        self.n_combinations = len(self.combinations)

        if self.n_combinations % jax.process_count() != 0:
            raise ValueError(
                f"Number of combinations ({self.n_combinations}) must be evenly divisible "
                f"by the number of processes ({jax.process_count()})."
            )

        self.batch_size = batch_size
        self.log_every = log_every
        self.progress_bar = progress_bar

        # Automatically determine batch size if None
        if self.batch_size is None:
            if jax.devices()[0].platform == "cpu":
                logger.warning(
                    """
                    Batch size not specified and automatic batch size
                    determination is not supported on CPU.
                    Falling back to default batch size of 64.
                    """
                )
                self.batch_size = 64
            else:
                self.batch_size = int(self.suggest_batch_size() * memory_limit)

        # Make sure that batch size is less than (self.n_combinations // jax.process_count())
        # and that it is a divisor of (self.n_combinations // jax.process_count()
        if self.batch_size > (self.n_combinations // jax.process_count()):
            self.batch_size = self.n_combinations // jax.process_count()

        if self.batch_size != 0:
            while (self.n_combinations // jax.process_count()) % self.batch_size != 0:
                self.batch_size -= 1

        os.makedirs(self.result_dir, exist_ok=True)

        print(f"Selecting batch size of {self.batch_size}")

    def suggest_batch_size(self: Self) -> int:
        """
        Estimate the largest feasible batch size based on device memory constraints.

        Returns:
            The estimated maximum batch size.
        """
        memory_stats = jax.devices()[0].memory_stats()
        if memory_stats is None:
            print("Memory stats not available, defaulting to batch size 64.")
            return 64

        max_device_memory = memory_stats["bytes_limit"] - memory_stats["bytes_in_use"]

        test_batch_sizes = [2, 4, 8, 16, 32]
        memory_usages = []

        for batch_size in test_batch_sizes:
            try:
                memory_usages.append(self._measure_memory_usage(batch_size))
            except Exception as e:
                print(f"Error measuring memory for batch size {batch_size}: {e}")
                break

        if len(memory_usages) < 2:
            raise ValueError("Not enough data points to interpolate memory usage.")

        interpolator = interp1d(
            memory_usages,
            test_batch_sizes[: len(memory_usages)],
            kind="linear",
            fill_value="extrapolate",
        )

        max_batch_size = int(interpolator(max_device_memory))
        return max_batch_size

    def _measure_memory_usage(self: Self, batch_size: int) -> int:
        """
        Measure memory usage of the objective function for a given batch size.

        Args:
            batch_size: The batch size to test.

        Returns:
            Estimated memory usage in bytes.
        """
        param_sample = {key: np.array([val[0]] * batch_size) for key, val in self.search_space.items()}

        mem_analysis = jax.jit(jax.vmap(self.objective_fn)).lower(**param_sample).compile().memory_analysis()

        arg_size: int = mem_analysis.argument_size_in_bytes  # type: ignore[union-attr]
        out_size: int = mem_analysis.output_size_in_bytes  # type: ignore[union-attr]
        temp_size: int = mem_analysis.temp_size_in_bytes  # type: ignore[union-attr]

        return arg_size + out_size + temp_size

    def _batch_generator(self: Self, indx: int = 0, size: int = 1) -> Iterator[list[tuple[Array, Array, Array, Array]]]:
        """Generates batches of parameter combinations."""
        current_slice_combinations = self.combinations[indx * self.n_combinations // size : (indx + 1) * self.n_combinations // size]
        batch_size: int = self.batch_size  # type: ignore[assignment]

        n_batches = len(current_slice_combinations) // batch_size

        for i in range(n_batches):
            yield current_slice_combinations[i * batch_size : (i + 1) * batch_size]

    def run(self: Self) -> None:
        """
        Run the grid search.

        Saves batch results to disk and clears them from memory.
        """
        rank = jax.process_index()
        size = jax.process_count()
        assert self.batch_size is not None

        total_batches = len(self.combinations) // (self.batch_size * size)
        log_interval = max(1, int(self.log_every * total_batches)) if self.log_every > 0 else 0

        progress_bar = tqdm(total=total_batches, desc=f"Processing batches on device {rank}/{size}") if self.progress_bar else None

        sample_batch = next(self._batch_generator(rank, size))
        sample_params = {key: np.array([val[0] for val in values]) for key, values in zip(self.param_keys, sample_batch)}
        # check if function returns a dictionary with value
        sample_result = jax.eval_shape(self.objective_fn, **sample_params)
        if not isinstance(sample_result, dict) or "value" not in sample_result:
            raise KeyError("The objective function must return a dictionary with a 'value' key.")

        for batch_idx, batch in enumerate(self._batch_generator(rank, size)):
            param_dicts = [dict(zip(self.param_keys, combo)) for combo in batch]
            param_arrays = {key: jnp.array([d[key] for d in param_dicts]) for key in self.param_keys}

            values = jax.vmap(lambda **kwargs: self.objective_fn(**kwargs))(**param_arrays)

            if not isinstance(values, dict):
                raise ValueError("The objective function must return a dictionary.")

            batch_results: dict[str, list[Array]] = {key: [] for key in self.param_keys}

            for i, param_dict in enumerate(param_dicts):
                for key in param_dict:
                    batch_results[key].append(param_dict[key])
                for key, val in values.items():
                    if key not in batch_results:
                        batch_results[key] = []
                    batch_results[key].append(val[i])

            batch_log = self.batch_idx + batch_idx
            result_file = os.path.join(self.result_dir, f"result_batch_{batch_log}_rank_{rank}.pkl")
            with open(result_file, "wb") as f:
                pickle.dump(batch_results, f)

            del batch_results

            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                logger.info(f"Rank {rank}: Processed {batch_idx + 1}/{total_batches} batches.")

            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

    def reduce_search_space(self: Self, search_space: dict[str, Array], results: dict[str, Array]) -> None:
        """
        Reduce the search space by removing combinations already processed.

        Args:
            search_space: A dictionary where keys are parameter names and values are arrays of possible values.
            results: A dictionary where keys match search_space keys and values are arrays of completed results.
        """
        # Generate all possible combinations from the search space
        param_names = list(search_space.keys())
        completed_param_results = {}
        for key in param_names:
            completed_param_results[key] = results[key]

        # Create set of completed combinations from the results
        completed_combinations = list(zip(*[results[key].reshape(-1, results[key].shape[-1]) for key in param_names]))

        filter_set = {tuple(tuple(arr.tolist()) for arr in tup) for tup in completed_combinations}

        reduced_combinations = [tup for tup in self.combinations if tuple(tuple(arr.tolist()) for arr in tup) not in filter_set]
        self.combinations = reduced_combinations

    @staticmethod
    def stack_results(result_folder: str) -> dict[str, ndarray[tuple[int, ...], dtype[Any]]]:
        """
        Stack results from a folder of result files.

        Args:
            result_folder: Folder containing .pkl files with batch results.

        Returns:
            A dictionary with stacked results.
        """
        combined_results: dict[str, list[Array]] = {}

        result_files = glob.glob(os.path.join(result_folder, "*.pkl"))

        for file_path in result_files:
            with open(file_path, "rb") as f:
                batch_results = pickle.load(f)

            for key, value in batch_results.items():
                if key not in combined_results:
                    combined_results[key] = []
                combined_results[key].extend(value)

        array_combined_results = {key: np.array(value) for key, value in combined_results.items()}

        assert "value" in array_combined_results
        sorted_indices = np.argsort(array_combined_results["value"])
        sorted_results = {key: value[sorted_indices] for key, value in array_combined_results.items()}
        return sorted_results

    @staticmethod
    def last_batch_idx(result_folder: str) -> int:
        """
        Determine the index of the last batch processed in the result folder.

        Args:
            result_folder: Path to the folder containing result files.

        Returns:
            The maximum batch index found, or 0 if no files are present.
        """
        result_files = glob.glob(os.path.join(result_folder, "*.pkl"))

        if not result_files:
            return 0  # No files found

        # Extract batch_idx from filenames using the known format
        batch_indices = []
        for file in result_files:
            filename = os.path.basename(file)
            try:
                # Parse the batch index from the filename
                batch_idx = int(filename.split("_")[2])
                batch_indices.append(batch_idx)
            except (IndexError, ValueError):
                continue  # Skip files that don't match the expected format

        return max(batch_indices) + 1 if batch_indices else 0
