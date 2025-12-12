# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import tempfile
import unittest
import sys
from io import StringIO

import torch
from torch.utils.data import DataLoader, TensorDataset

from nvidia.dali.plugin.pytorch.loader_evaluator import LoaderEvaluator  # noqa: E402


class TestLoaderEvaluatorBasic(unittest.TestCase):
    """Test basic functionality of LoaderEvaluator."""

    def setUp(self):
        """Create sample data for testing."""
        # Set the seed for reproducibility
        torch.manual_seed(1234)

        data = torch.randn(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 images
        targets = torch.randint(0, 10, (100,))  # 100 labels from 0-9
        self.sample_dataset = TensorDataset(data, targets)
        self.sample_dataloader = DataLoader(
            self.sample_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

    def test_initialization(self):
        """Test that LoaderEvaluator can be initialized with a
        DataLoader."""
        loader = LoaderEvaluator(self.sample_dataloader)
        self.assertIsNotNone(loader)
        self.assertTrue(hasattr(loader, "dataloader"))
        self.assertIs(loader.dataloader, self.sample_dataloader)

    def test_initialization_with_mode(self):
        """Test initialization with different modes."""
        # Test log mode
        loader_log = LoaderEvaluator(self.sample_dataloader, mode="log")
        self.assertEqual(loader_log.mode, "log")

        # Test replay mode
        loader_replay = LoaderEvaluator(self.sample_dataloader, mode="replay")
        self.assertEqual(loader_replay.mode, "replay")

    def test_initialization_with_parameters(self):
        """Test initialization with additional parameters."""
        loader = LoaderEvaluator(
            self.sample_dataloader,
            mode="replay",
            num_cached_batches=10,
            metrics_file="test_metrics.json",
        )
        self.assertEqual(loader.mode, "replay")
        self.assertEqual(loader.num_cached_batches, 10)
        self.assertEqual(loader.metrics_file, "test_metrics.json")

    def test_length_property(self):
        """Test that length property works correctly."""
        loader = LoaderEvaluator(self.sample_dataloader)
        self.assertEqual(len(loader), len(self.sample_dataloader))

    def test_iteration_basic(self):
        """Test that the loader can be iterated over."""
        loader = LoaderEvaluator(self.sample_dataloader)

        # Test that we can iterate through the loader
        batch_count = 0
        for batch in loader:
            batch_count += 1
            # Should get tuples of (data, target) - but DataLoader might
            # return list
            self.assertIsInstance(batch, (tuple, list))
            self.assertEqual(len(batch), 2)
            data, target = batch
            self.assertIsInstance(data, torch.Tensor)
            self.assertIsInstance(target, torch.Tensor)

            # Stop after a few batches to keep test fast
            if batch_count >= 3:
                break

        self.assertGreater(batch_count, 0)

    def test_iteration_consistency(self):
        """Test that iteration is consistent with the original
        DataLoader."""
        # Create loaders without shuffling to ensure consistent data
        original_loader = torch.utils.data.DataLoader(
            self.sample_dataset,
            batch_size=16,
            shuffle=False,  # Disable shuffling for consistency
            num_workers=0,
        )
        wrapped_loader = LoaderEvaluator(original_loader)

        # Get first few batches from both loaders
        original_batches = []
        wrapped_batches = []

        for i, (orig_batch, wrapped_batch) in enumerate(zip(original_loader, wrapped_loader)):
            original_batches.append(orig_batch)
            wrapped_batches.append(wrapped_batch)

            # Stop after a few batches
            if i >= 2:
                break

        # The batches should be identical (since we're just wrapping)
        self.assertEqual(len(original_batches), len(wrapped_batches))
        for orig_batch, wrapped_batch in zip(original_batches, wrapped_batches):
            orig_data, orig_target = orig_batch
            wrapped_data, wrapped_target = wrapped_batch

            # Check that tensors are equal
            self.assertTrue(torch.equal(orig_data, wrapped_data))
            self.assertTrue(torch.equal(orig_target, wrapped_target))


class TestLoaderEvaluatorModes(unittest.TestCase):
    """Test different modes of LoaderEvaluator."""

    def setUp(self):
        """Create sample data for testing."""
        # Create simple tensor data
        data = torch.randn(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 images
        targets = torch.randint(0, 10, (100,))  # 100 labels from 0-9
        self.sample_dataset = TensorDataset(data, targets)
        self.sample_dataloader = DataLoader(
            self.sample_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

    def test_log_mode(self):
        """Test log mode functionality."""
        loader = LoaderEvaluator(self.sample_dataloader, mode="log")
        self.assertEqual(loader.mode, "log")

        # In log mode, should behave like normal DataLoader
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 2:
                break

        self.assertEqual(batch_count, 2)

        # Should have collected metrics
        metrics = loader.get_metrics()
        self.assertEqual(metrics["total_batches"], 2)
        self.assertEqual(metrics["mode"], "log")

    def test_replay_mode_auto_caching(self):
        """Test replay mode automatically caches batches during
        construction."""
        loader = LoaderEvaluator(self.sample_dataloader, mode="replay", num_cached_batches=3)
        self.assertEqual(loader.mode, "replay")

        # Should have automatically cached batches during construction
        self.assertTrue(loader.cache_ready)
        self.assertGreater(len(loader.cached_batches), 0)
        self.assertLessEqual(len(loader.cached_batches), 3)  # Limited by num_cached_batches

    def test_replay_mode_iteration(self):
        """Test replay mode iteration with auto-cached batches."""
        loader = LoaderEvaluator(self.sample_dataloader, mode="replay", num_cached_batches=3)

        # Should be able to iterate and replay cached batches
        # The test should iterate through all available batches (7 total)
        # but we'll limit to 5 to keep test fast
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 5:  # Should replay cached batches multiple times
                break

        self.assertEqual(batch_count, 5)

        # Should have collected metrics
        metrics = loader.get_metrics()
        self.assertEqual(metrics["total_batches"], 5)
        self.assertEqual(metrics["mode"], "replay")

    def test_replay_mode_epoch_length(self):
        """Test that replay mode maintains the same epoch length as
        original DataLoader."""
        # Test replay mode with fewer cached batches than the original dataset
        replay_loader = LoaderEvaluator(self.sample_dataloader, mode="replay", num_cached_batches=2)

        # Count batches from replay mode
        replay_batch_count = 0
        for _ in replay_loader:
            replay_batch_count += 1

        # Should produce the same number of batches as the original DataLoader
        original_length = len(self.sample_dataloader)
        self.assertEqual(replay_batch_count, original_length)

        # Should have collected metrics for all batches
        metrics = replay_loader.get_metrics()
        self.assertEqual(metrics["total_batches"], original_length)
        self.assertEqual(metrics["mode"], "replay")


class TestLoaderEvaluatorMethods(unittest.TestCase):
    """Test methods of LoaderEvaluator."""

    def setUp(self):
        """Create sample data for testing."""
        # Create simple tensor data
        data = torch.randn(100, 3, 32, 32)  # 100 samples, 3 channels, 32x32 images
        targets = torch.randint(0, 10, (100,))  # 100 labels from 0-9
        self.sample_dataset = TensorDataset(data, targets)
        self.sample_dataloader = DataLoader(
            self.sample_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

    def test_get_metrics_empty(self):
        """Test get_metrics method with no processed batches."""
        loader = LoaderEvaluator(self.sample_dataloader)
        metrics = loader.get_metrics()

        # Should return empty dict when no batches processed
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics), 0)

    def test_get_metrics_with_data(self):
        """Test get_metrics method with processed batches."""
        loader = LoaderEvaluator(self.sample_dataloader, mode="log")

        # Process some batches
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 2:
                break

        metrics = loader.get_metrics()

        # Should have collected metrics
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics["total_batches"], 2)
        self.assertEqual(metrics["mode"], "log")
        self.assertIn("avg_batch_time", metrics)
        self.assertIn("total_time", metrics)
        self.assertEqual(len(metrics["batch_times"]), 2)

    def test_save_metrics(self):
        """Test save_metrics method."""
        loader = LoaderEvaluator(self.sample_dataloader, mode="log")

        # Process some batches
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 2:
                break

        # Test saving metrics
        with tempfile.TemporaryDirectory() as tmp_path:
            metrics_file = os.path.join(tmp_path, "test_metrics.json")
            loader.save_metrics(metrics_file)

            # Check that file was created and contains metrics
            self.assertTrue(os.path.exists(metrics_file))

            with open(metrics_file) as f:
                saved_metrics = json.load(f)

            self.assertEqual(saved_metrics["total_batches"], 2)
            self.assertEqual(saved_metrics["mode"], "log")

    def test_save_metrics_no_filename(self):
        """Test save_metrics method without filename."""
        loader = LoaderEvaluator(self.sample_dataloader)

        # Should not raise exception, just print message
        loader.save_metrics()

    def test_print_summary(self):
        """Test print_summary method."""
        loader = LoaderEvaluator(self.sample_dataloader, mode="log")

        # Process some batches
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 2:
                break

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        loader.print_summary()

        sys.stdout = old_stdout
        output = captured_output.getvalue()

        # Should contain summary information
        self.assertIn("Loader Evaluator Summary:", output)
        self.assertIn("Mode: log", output)
        self.assertIn("Dataset size:", output)
        self.assertIn("Number of batches:", output)
        self.assertIn("Total batches processed: 2", output)


class TestLoaderEvaluatorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_dataloader(self):
        """Test with empty DataLoader."""
        # Create empty dataset
        empty_data = torch.empty(0, 3, 32, 32)
        empty_targets = torch.empty(0, dtype=torch.long)
        empty_dataset = torch.utils.data.TensorDataset(empty_data, empty_targets)
        empty_dataloader = torch.utils.data.DataLoader(empty_dataset, batch_size=16)

        loader = LoaderEvaluator(empty_dataloader)
        self.assertEqual(len(loader), 0)

        # Should not raise exception when iterating over empty loader
        batch_count = 0
        for _ in loader:
            batch_count += 1

        self.assertEqual(batch_count, 0)

        # Should have empty metrics
        metrics = loader.get_metrics()
        self.assertEqual(len(metrics), 0)

    def test_single_batch_dataloader(self):
        """Test with DataLoader that has only one batch."""
        # Create dataset with single sample
        single_data = torch.randn(1, 3, 32, 32)
        single_targets = torch.tensor([0])
        single_dataset = torch.utils.data.TensorDataset(single_data, single_targets)
        single_dataloader = torch.utils.data.DataLoader(single_dataset, batch_size=1)

        loader = LoaderEvaluator(single_dataloader)
        self.assertEqual(len(loader), 1)

        # Should be able to iterate once
        batch_count = 0
        for _ in loader:
            batch_count += 1

        self.assertEqual(batch_count, 1)

        # Should have collected metrics
        metrics = loader.get_metrics()
        self.assertEqual(metrics["total_batches"], 1)

    def test_invalid_mode(self):
        """Test with invalid mode (should default to log mode)."""
        # Create sample data
        data = torch.randn(100, 3, 32, 32)
        targets = torch.randint(0, 10, (100,))
        sample_dataset = TensorDataset(data, targets)
        sample_dataloader = DataLoader(sample_dataset, batch_size=16, num_workers=0)

        # This should not raise an exception
        loader = LoaderEvaluator(sample_dataloader, mode="invalid_mode")
        self.assertEqual(loader.mode, "invalid_mode")

        # Should still be iterable (defaults to log mode)
        batch_count = 0
        for _ in loader:
            batch_count += 1
            if batch_count >= 1:
                break

        self.assertEqual(batch_count, 1)

        # Should have collected metrics (log mode behavior)
        metrics = loader.get_metrics()
        self.assertEqual(metrics["total_batches"], 1)

    def test_replay_mode_with_small_dataset(self):
        """Test replay mode with dataset smaller than cache size."""
        # Create dataset with only 2 samples
        small_data = torch.randn(2, 3, 32, 32)
        small_targets = torch.tensor([0, 1])
        small_dataset = torch.utils.data.TensorDataset(small_data, small_targets)
        small_dataloader = torch.utils.data.DataLoader(small_dataset, batch_size=1)

        loader = LoaderEvaluator(small_dataloader, mode="replay", num_cached_batches=5)

        # Should have cached all available batches
        self.assertEqual(len(loader.cached_batches), 2)  # Only 2 batches available
        self.assertTrue(loader.cache_ready)  # Should still be ready

        # Process all batches
        batch_count = 0
        for _ in loader:
            batch_count += 1

        self.assertEqual(batch_count, 2)

    def test_replay_mode_wraps_around_cache(self):
        """Test that replay mode wraps around the cache to match original
        DataLoader length."""
        # Create sample data
        data = torch.randn(100, 3, 32, 32)
        targets = torch.randint(0, 10, (100,))
        sample_dataset = TensorDataset(data, targets)
        sample_dataloader = DataLoader(sample_dataset, batch_size=16, num_workers=0)

        # Test replay mode with fewer cached batches than the original dataset
        replay_loader = LoaderEvaluator(sample_dataloader, mode="replay", num_cached_batches=2)

        # Count all batches from replay mode
        batch_count = 0
        for batch in replay_loader:
            batch_count += 1

        # Should produce the same number of batches as the original DataLoader
        original_length = len(sample_dataloader)
        self.assertEqual(
            batch_count, original_length
        )  # Should wrap around cache to match original length
