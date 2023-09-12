# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import jax
import jax.dlpack

from nvidia.dali.backend import TensorGPU


def _to_jax_array(dali_tensor: TensorGPU) -> jax.Array:
    """Converts input DALI tensor to JAX array.

    Args:
        dali_tensor (TensorGPU): DALI GPU tensor to be converted to JAX array.

    Note:
        This function performs deep copy of the underlying data. That will change in
        future releases.

    Warning:
        As private this API may change without notice.

    Returns:
        jax.Array: JAX array with the same values and backing device as
        input DALI tensor.
    """
    jax_array = jax.dlpack.from_dlpack(dali_tensor._expose_dlpack_capsule())

    # For now we need this copy to make sure that underlying memory is available.
    # One solution is to implement full DLPack contract in DALI.
    # TODO(awolant): Remove this copy.
    return jax_array.copy()
