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

import numpy as np
from nvidia.dali import fn, pipeline_def
from nvidia.dali.backend import GetSchema
from nvidia.dali.ops import _registry


def test_random_state_multiple_state_values():
    """Test that _random_state can accept different state arrays per sample."""
    batch_size = 3
    state_size = 6  # Number of uint32 values in the state

    @pipeline_def(
        batch_size=batch_size,
        device_id=0,
        num_threads=1,
        seed=42,
    )
    def pipeline():
        random_states = fn.external_source(
            lambda sample_info: np.random.randint(0, 2**32, size=state_size, dtype=np.uint32),
            batch=False,
        )
        result = fn.random.uniform(
            range=[0.0, 1.0],
            shape=[100],
            _random_state=random_states,
        )
        return result

    pipe = pipeline()
    pipe.build()
    outputs = pipe.run()
    out = outputs[0].as_cpu()

    # Each sample should have 100 values
    for i in range(batch_size):
        sample = np.array(out[i])
        assert sample.shape == (100,), f"Expected shape (100,), got {sample.shape}"


def test_operator_random_state_requirements():
    """Test that all DALI operators follow the correct _random_state requirements.

    Based on the requirements:
    - Non-reader operators with non-deprecated seed should have _random_state
    - Reader operators should NOT have _random_state
    - Operators without seed or with deprecated seed should NOT have _random_state
    """
    # Discover all operators
    _registry._discover_ops()
    all_ops = _registry._all_registered_ops()

    missing_random_state = []
    wrong_random_state = []
    for op_name in all_ops:
        schema = GetSchema(op_name)
        assert schema is not None, f"Schema for {op_name} not found"

        # Check if it's a reader
        is_reader = "reader" in op_name.lower()
        has_seed = schema.HasRandomSeedArg()
        has_random_state = schema.HasArgument("_random_state")

        if has_random_state:
            # should be a tensor argument
            assert schema.IsTensorArgument(
                "_random_state"
            ), f"_random_state should be a tensor argument in {op_name}"
            # should not be advertised in the argument names
            assert (
                "_random_state" not in schema.GetArgumentNames()
            ), f"_random_state should not be in argument names for {op_name}"

        should_have_random_state = has_seed and not is_reader
        should_not_have_random_state = is_reader or not has_seed

        if should_have_random_state and not has_random_state:
            missing_random_state.append(
                {
                    "op": op_name,
                    "reason": "Non-reader with seed argument but missing _random_state",
                }
            )

        if should_not_have_random_state and has_random_state:
            wrong_random_state.append(
                {"op": op_name, "reason": "Has _random_state but should not have it"}
            )

    assert len(missing_random_state) + len(wrong_random_state) == 0, (
        f"Operators missing _random_state: ({len(missing_random_state)}):\n"
        + "\n".join(f"  - {item['op']}: {item['reason']}" for item in missing_random_state)
        + f"\nOperators that shouldn't have _random_state: ({len(wrong_random_state)}):\n"
        + "\n".join(f"  - {item['op']}: {item['reason']}" for item in wrong_random_state)
    )
