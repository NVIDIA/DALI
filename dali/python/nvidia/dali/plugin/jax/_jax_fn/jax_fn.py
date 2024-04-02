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

from typing import Optional, Protocol, Tuple, Union

import jax
import jax.dlpack
import jax.sharding

from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import fn

from .function_transform import jax_callback_wrapper as _jax_callback_wrapper


class JaxCallback(Protocol):

    def __call__(self, *args: jax.Array) -> Optional[Tuple[jax.Array, ...]]: ...


class DaliCallback(Protocol):

    def __call__(self, *args: _DataNode) -> Optional[Tuple[_DataNode, ...]]: ...


def jax_fn(
    function: Optional[JaxCallback]=None,
    num_outputs: int = 1,
    output_layouts: Union[None, str, Tuple[str]] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
    device: Optional[str] = None,
    preserve : bool = True,
) -> DaliCallback:
    """
    Transforms the Python function ``function`` that processes ``jax.Array`` objects into
    DALI operator that can be used inside DALI pipeline definition or jax plugin iterator definition.
    The transformed function accepts and returns the same number of inputs and outputs as the
    original `function`. The inputs and outputs of the transformed function are traced by DALI
    and are interoperable with remaining DALI operators.

    If the resulting operator is run with DALI GPU batches, the internal DALI and JAX streams will be
    synchronized. The JAX operations do not need to be further synchronized by the user, they can
    be scheduled and run asynchronously.

    The inputs passed to the ``function`` must not be accessed after the ``function`` completes
    (for example, they should not be stored in some non-local scope).

    .. note::

        This is experimental API and may change in future releases.

    Args
    ----
    function : JaxCallback
        Python callback that accepts and returns zero or more `jax.Array` objects.
        The function will receive batches processed by DALI as `jax.Array` tensors
        (with the leftmost extent corresponding to DALI batch).
        For this reason, the transformed function can only receive DALI batches that
        contain samples of uniform shape.
    num_outputs : int, default=1
        The number of outputs returned by the ``function``.

        Function can return no output, in that case the `num_outputs` must be set to 0.
        If the ``num_outputs`` is 1 (the default), callback should return a single JAX array,
        for ``num_outputs`` > 1, callback should return a tuple of JAX arrays.
    output_layouts: Union[str, Tuple[str]], optional
        The layouts of returned tensors.

        It can be either a list of strings for all of ``num_outputs`` respective outputs or a single string
        to be set to all of the outputs.

        Please note, in DALI, the outermost batch extent is implicit, the layout should
        take into account only the sample dimensions.

        If the argument is not specified, the ``function`` has the same number of inputs and outputs and
        the dimensionality of respective inputs and outputs is preserved, the layout will be propagated
        from the input to the output.
    sharding: jax.sharding.Sharding, optional
        The JAX sharding object (either ``PositionalSharding`` or ``NamedSharding``). If specified, the
        ``jax.Arrays`` passed to the ``function`` will be a global ``jax.Array`` aware of the sharding.

        .. note::

            Currently, only the global sharding is supported, i.e. the number of the local devices
            in the given process must be exactly one.
    device: str, optional
        Either "cpu", "gpu" or None.
        The device kind on which all of the DALI inputs and outputs to the transformed function will be placed.
        If not specified, the device will be deduced based on the DALI inputs passed to the resulting function.
        Currently, the device kind of all the inputs and outputs must be the same.
    preserve: bool, default=True
        If set to False, the returned DALI function may be optimized out of the DALI pipeline, if
        it does not return any outputs or none of the function outputs contribute to the pipeline's output.

    Returns
    -------
    DaliCallback
        The transformed function that processes DALI-traced batches (DataNodes).
    """

    def decorator(function):

        def dali_callback(*args: _DataNode) -> Optional[Tuple[_DataNode, ...]]:
            is_gpu = device == "gpu" or any(getattr(arg, "device", None) == "gpu" for arg in args)
            inferred_device = "gpu" if is_gpu else "cpu"
            actual_callback = _jax_callback_wrapper(function, sharding, inferred_device)
            jax_fn_outputs = fn._jax_function(
                *args,
                function_id=id(actual_callback),
                num_outputs=num_outputs,
                output_layouts=output_layouts,
                preserve=preserve,
            )
            # to make sure the `_actual_callback` lives
            if isinstance(jax_fn_outputs, _DataNode):
                setattr(jax_fn_outputs, "_actual_callback", actual_callback)
            else:
                for jax_fn_output in jax_fn_outputs:
                    setattr(jax_fn_output, "_actual_callback", actual_callback)
            return jax_fn_outputs

        return dali_callback

    if function is None:
        return decorator
    else:
        return decorator(function)
