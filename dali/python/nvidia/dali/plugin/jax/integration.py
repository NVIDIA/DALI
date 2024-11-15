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

import jax
import jax.dlpack

from nvidia.dali.backend import TensorGPU
from packaging.version import Version


_jax_has_old_dlpack = Version(jax.__version__) < Version("0.4.16")


if Version(jax.__version__) >= Version("0.4.31"):

    def _jax_device(jax_array):
        return jax_array.device

elif Version(jax.__version__) >= Version("0.4.27"):

    def _jax_device(jax_array):
        devs = jax_array.devices()
        if len(devs) != 1:
            raise RuntimeError("The array must be associated with exactly one device")
        for d in devs:
            return d

else:

    def _jax_device(jax_array):
        return jax_array.device()


def _to_jax_array(dali_tensor: TensorGPU, copy: bool) -> jax.Array:
    """Converts input DALI tensor to JAX array.

    Args:
        dali_tensor (TensorGPU):
            DALI GPU tensor to be converted to JAX array.

        copy (bool):
            If True, the output is copied;
            if False, the output may wrap DLPack capsule obtained from `dali_tensor`.

    Note:
        This function may perform a copy of the data even if `copy==False` when JAX version is
        insufficient (<0.4.16)

    Warning:
        As private this API may change without notice.

    Returns:
        jax.Array: JAX array with the same values and backing device as
        input DALI tensor.
    """
    if _jax_has_old_dlpack:
        copy = True
        jax_array = jax.dlpack.from_dlpack(dali_tensor.__dlpack__(stream=None))
    else:
        jax_array = jax.dlpack.from_dlpack(dali_tensor)

    if copy:
        jax_array = jax_array.copy()
    return jax_array
