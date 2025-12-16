# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Loader Evaluator Implementation
Wraps PyTorch DataLoader to provide in-memory caching and performance
monitoring.
"""

import json
import time
from collections import deque
from typing import Any, Dict, Iterator, Optional

from torch.utils.data import DataLoader


class LoaderEvaluator:
    """
    A PyTorch DataLoader wrapper with in-memory caching and performance
    monitoring.

    This class wraps a standard PyTorch DataLoader and adds:
    - Performance metrics collection
    - In-memory batch caching for ideal performance simulation
      (replay mode)
    - Real vs. ideal performance comparison

    Modes:
    - log: Normal iteration while collecting performance metrics
    - replay: Caches batches during construction, then replays them for
      ideal performance
    """

    def __init__(
        self,
        dataloader: DataLoader,
        mode: str = "log",
        num_cached_batches: int = 20,
        metrics_file: Optional[str] = None,
    ):
        """
        Initialize the Loader Evaluator.

        Args:
            dataloader: The PyTorch DataLoader to wrap
            mode: "log" or "replay"
            num_cached_batches: Number of batches to cache in memory
                (for replay mode)
            metrics_file: File to save metrics (optional)
        """
        self.dataloader = dataloader
        self.mode = mode
        self.num_cached_batches = num_cached_batches
        self.metrics_file = metrics_file

        # In-memory cache for batches
        self.cached_batches = deque(maxlen=num_cached_batches)
        self.cache_ready = False

        # Performance metrics
        self.batch_times = []
        self.start_time = None
        self.end_time = None

        # If replay mode, cache batches during construction
        if self.mode == "replay":
            self._cache_batches()

    def __iter__(self) -> Iterator[Any]:
        """Iterate through the dataloader based on the current mode."""
        if self.mode == "log":
            return self._log_mode_iter()
        elif self.mode == "replay":
            return self._replay_mode_iter()
        else:
            # Default to log mode for unknown modes
            return self._log_mode_iter()

    def __len__(self):
        """Return the length of the wrapped dataloader."""
        return len(self.dataloader)

    def _cache_batches(self):
        """Cache batches from the dataloader for replay mode."""
        self.cached_batches.clear()
        self.cache_ready = False

        for batch in self.dataloader:
            self.cached_batches.append(batch)
            # Cache is ready when we have at least one batch cached
            if len(self.cached_batches) > 0:
                self.cache_ready = True

    def _log_mode_iter(self) -> Iterator[Any]:
        """Log mode: iterate normally while collecting metrics."""
        self.start_time = time.time()
        self.batch_times = []

        dataloader_iter = iter(self.dataloader)
        while True:
            try:
                batch_start = time.time()
                batch = next(dataloader_iter)  # This is where the actual data loading happens
                batch_time = time.time() - batch_start
                self.batch_times.append(batch_time)
                yield batch
            except StopIteration:
                break

        self.end_time = time.time()

    def _replay_mode_iter(self) -> Iterator[Any]:
        """Replay mode: replay cached batches for ideal performance
        simulation."""
        if not self.cache_ready or len(self.cached_batches) == 0:
            raise RuntimeError(
                "No cached batches available. This should not happen in\n"
                "replay mode as batches are cached during construction."
            )

        self.start_time = time.time()
        self.batch_times = []

        # Replay cached batches to match original DataLoader length
        original_length = len(self.dataloader)

        for i in range(original_length):
            batch_start = time.time()
            batch = self.cached_batches[i % len(self.cached_batches)]
            batch_time = time.time() - batch_start
            self.batch_times.append(batch_time)
            yield batch

        self.end_time = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected performance metrics."""
        if not self.batch_times:
            return {}

        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0

        return {
            "mode": self.mode,
            "total_batches": len(self.batch_times),
            "total_time": total_time,
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times),
            "min_batch_time": min(self.batch_times),
            "max_batch_time": max(self.batch_times),
            "batch_times": self.batch_times.copy(),
            "cached_batches": len(self.cached_batches),
            "cache_ready": self.cache_ready,
        }

    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to a JSON file."""
        if filename is None:
            filename = self.metrics_file

        if filename is None:
            print("No filename provided for saving metrics")
            return

        metrics = self.get_metrics()

        with open(filename, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {filename}")

    def print_summary(self):
        """Print a summary of the loader and collected metrics."""
        print("Loader Evaluator Summary:")
        print(f"  Mode: {self.mode}")
        print(f"  Dataset size: {len(self.dataloader.dataset)}")
        print(f"  Number of batches: {len(self.dataloader)}")
        cache_size = len(self.cached_batches)
        max_cache = self.num_cached_batches
        print(f"  Cache size: {cache_size}/{max_cache}")
        print(f"  Cache ready: {self.cache_ready}")

        metrics = self.get_metrics()
        if metrics:
            print(f"  Total batches processed: {metrics['total_batches']}")
            print(f"  Total time: {metrics['total_time']:.2f}s")
            avg_time = metrics["avg_batch_time"]
            print(f"  Average batch time: {avg_time:.4f}s")
            min_time = metrics["min_batch_time"]
            print(f"  Min batch time: {min_time:.4f}s")
            print(f"  Max batch time: {metrics['max_batch_time']:.4f}s")
