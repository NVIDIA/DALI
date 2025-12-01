# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import jax.sharding


def gpu_to_dlpack(tensor: jax.Array, stream):
    devices = list(tensor.devices())
    if not len(devices) == 1:
        raise ValueError(
            f"The function returned array split across multiple devices ({len(devices)}), "
            f"expected single device array."
        )
    if devices[0].platform != "gpu":
        raise ValueError(
            f"The function returned array residing on the device of "
            f"kind `{devices[0].platform}`, expected `gpu`."
        )
    return jax.dlpack.to_dlpack(tensor, stream=stream)


def cpu_to_dlpack(tensor: jax.Array):
    devices = list(tensor.devices())
    if not len(devices) == 1:
        raise ValueError(
            f"The function returned array split across multiple devices ({len(devices)}), "
            f"expected single device array."
        )
    if devices[0].platform != "cpu":
        raise ValueError(
            f"The function returned array residing on the device of "
            f"kind `{devices[0].platform}`, expected `cpu`."
        )
    return jax.dlpack.to_dlpack(tensor)


def with_gpu_dl_tensors_as_arrays(callback):

    def inner(stream, *dl_tensors):
        ts = tuple(jax.dlpack.from_dlpack(t) for t in dl_tensors)
        out = callback(*ts)
        if out is None:
            return
        else:
            out = out if isinstance(out, (tuple, list)) else (out,)
            return tuple(gpu_to_dlpack(t, stream=stream) for t in out)

    return inner


def with_cpu_dl_tensors_as_arrays(callback):

    def inner(*dl_tensors):
        ts = tuple(jax.dlpack.from_dlpack(t) for t in dl_tensors)
        out = callback(*ts)
        if out is None:
            return
        else:
            out = out if isinstance(out, (tuple, list)) else (out,)
            return tuple(cpu_to_dlpack(t) for t in out)

    return inner


def with_sharding(callback, sharding):
    if jax.local_device_count() != 1:
        raise NotImplementedError(
            f"Currently, the `jax_function` supports only global/multiprocessing sharding. "
            f"The number of local devices seen by the process must be 1, "
            f"got {jax.local_device_count()}"
        )

    if not isinstance(sharding, (jax.sharding.NamedSharding, jax.sharding.PositionalSharding)):
        raise ValueError(
            f"The value passed as `sharding` must be an instance of `NamedSharding` or "
            f"`PositionalSharding`, got value of a type {type(sharding)}"
        )

    def as_sharded_array(array: jax.Array) -> jax.Array:
        array_shape = array.shape
        if isinstance(sharding, jax.sharding.NamedSharding):
            global_shape = (sharding.mesh.size * array_shape[0], *array_shape[1:])
        else:
            global_shape = (sharding.shape[0] * array_shape[0], *array_shape[1:])
        return jax.make_array_from_single_device_arrays(global_shape, sharding, [array])

    def as_single_device_array(sharded_array: jax.Array) -> jax.Array:
        local_arrays = [x.data for x in sharded_array.addressable_shards]
        if len(local_arrays) != 1:
            raise ValueError(
                f"The function returned multiple shards, "
                f"expected exactly one, got {len(local_arrays)}."
            )
        return local_arrays[0]

    def inner(*arrays: jax.Array):
        sharded_arrays = tuple(as_sharded_array(array) for array in arrays)
        out = callback(*sharded_arrays)
        if out is None:
            return
        else:
            out = out if isinstance(out, (tuple, list)) else (out,)
            return tuple(as_single_device_array(t) for t in out)

    return inner


def jax_callback_wrapper(function, sharding, device):

    assert device in ("cpu", "gpu")

    dl_pack_wrapper = (
        with_cpu_dl_tensors_as_arrays if device == "cpu" else with_gpu_dl_tensors_as_arrays
    )
    return dl_pack_wrapper(function if sharding is None else with_sharding(function, sharding))
