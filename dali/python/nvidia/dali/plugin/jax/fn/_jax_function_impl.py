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

from packaging.version import Version

import jax
import jax.dlpack
import jax.sharding

from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import ops
import nvidia.dali.ops._python_def_op_utils as _python_def_op_utils
from nvidia.dali.plugin import jax as dax

from ._function_transform import jax_callback_wrapper as _jax_callback_wrapper


class JaxCallback(Protocol):

    def __call__(self, *args: jax.Array) -> Optional[Tuple[jax.Array, ...]]: ...


class DaliCallback(Protocol):

    def __call__(self, *args: _DataNode) -> Optional[Tuple[_DataNode, ...]]: ...


class _JaxFunction(
    ops.python_op_factory("_JaxFunction", "_JaxFunction", "_JaxFunction", generated=False)
):
    _impl_module = "nvidia.dali.plugin.jax.fn"
    ops.register_cpu_op("_JaxFunction")
    ops.register_gpu_op("_JaxFunction")

    def __init__(self, function, device="cpu", sharding=None, **kwargs):
        self.function = _jax_callback_wrapper(function, sharding, device)
        self.sharding = sharding
        super().__init__(
            function_id=id(self.function),
            device=device,
            **kwargs,
        )


ops._wrap_op(_JaxFunction, "fn", "nvidia.dali.plugin.jax")


def jax_function(
    function: Optional[JaxCallback] = None,
    num_outputs: int = 1,
    output_layouts: Union[None, str, Tuple[str]] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
    device: Optional[str] = None,
    preserve: bool = True,
) -> DaliCallback:
    """
    Transforms the Python function `function` that processes ``jax.Array`` objects into
    DALI operator that can be used inside DALI pipeline definition
    or JAX plugin iterator definition.
    The transformed function accepts and returns the same number of inputs and outputs as the
    original `function`, but changes their types: from ``jax.Array`` to DALI-traced ``DataNodes``.
    The resulting function is interoperable with other DALI operators.

    For example, we could implement horizontal flipping operation in JAX as follows::

        import jax
        from nvidia.dali import pipeline_def, fn, types
        from nvidia.dali.plugin import jax as dax

        @dax.fn.jax_function
        def flip_horizontal(image_batch: jax.Array):
            return image_batch[:, :, ::-1, :]  # batch of HWC images

        @pipeline_def(batch_size=4, device_id=0, num_threads=4)
        def pipeline():
            image, _ = fn.readers.file(file_root=jpeg_path_dali_extra)
            image = fn.decoders.image(image, device="mixed", output_type=types.RGB)
            image = fn.resize(image, size=(244, 244))
            flipped = flip_horizontal(image)
            return image, flipped

    The `function` can be transformed with usual JAX transformations, for example
    we can utilize JAX's just-in-time compilation and vectorization adding
    the appropriate decorators in the above example::

        @dax.fn.jax_function
        @jax.jit
        @jax.vmap
        def flip_horizontal(image: jax.Array):
            return image[:, ::-1, :]  # HWC image

    If the resulting function is run with DALI GPU batches, the internal DALI and JAX
    streams will be synchronized. The JAX operations do not need to be further
    synchronized by the user.

    The ``jax.Arrays`` passed to the `function` must not be accessed after the
    `function` completes (for example, they should not be stored in some non-local scope).

    .. note::

        This is experimental API and may change in future releases.

    .. note::

        The `jax_function` requires JAX version 0.4.16 or higher, with GPU support.
        JAX 0.4.16 requires Python 3.9 or higher.

    Args
    ----
    function : JaxCallback
        Python callback that accepts and returns zero or more `jax.Array` objects.
        The function will receive batches processed by DALI as `jax.Array` tensors
        (with the leftmost extent corresponding to DALI batch).
        For this reason, the transformed function can only receive DALI batches that
        contain samples of uniform shape.
    num_outputs : int, default=1
        The number of outputs returned by the `function`.

        Function can return no output, in that case the `num_outputs` must be set to 0.
        If the `num_outputs` is 1 (the default), callback should return a single JAX array,
        for `num_outputs` > 1, callback should return a tuple of JAX arrays.
    output_layouts: Union[str, Tuple[str]], optional
        The layouts of returned tensors.

        It can be either a list of strings for all of `num_outputs` respective outputs
        or a single string to be set to all of the outputs.

        Please note, in DALI, the outermost batch extent is implicit, the layout should
        take into account only the sample dimensions.

        If the argument is not specified and the `function`'s i-th output has the same
        dimensionality as the i-th input, the layout will be propagated from the input to
        the corresponding output.
    sharding: jax.sharding.Sharding, optional
        The JAX sharding object (either ``PositionalSharding`` or ``NamedSharding``).
        If specified, the ``jax.Arrays`` passed to the `function` will be a global
        ``jax.Array`` aware of the sharding.

        .. note::

            Currently, only the global sharding is supported, i.e. the number of the local devices
            in the given process must be exactly one.
    device: str, optional
        Either "cpu", "gpu" or None.
        The device kind on which all of the DALI inputs and outputs to the transformed function
        will be placed. If not specified, the device will be deduced based on the DALI
        inputs passed to the resulting function. Currently, the device kind of all the inputs
        and outputs must be the same.
    preserve: bool, default=True
        If set to False, the returned DALI function may be optimized out of the DALI pipeline,
        if it does not return any outputs or none of the function outputs contribute
        to the pipeline's output.

    Returns
    -------
    DaliCallback
        The transformed function that processes DALI-traced batches (DataNodes).
    """

    if Version(jax.__version__) < Version("0.4.16"):
        raise RuntimeError("DALI `jax_function` requires JAX 0.4.16 or above.")

    def decorator(function):

        def dali_callback(*args: _DataNode) -> Optional[Tuple[_DataNode, ...]]:
            jax_fn_outputs = dax.fn._jax_function(
                *args,
                function=function,
                num_outputs=num_outputs,
                output_layouts=output_layouts,
                sharding=sharding,
                device=device,
                preserve=preserve,
            )
            return jax_fn_outputs

        return dali_callback

    if function is None:
        return decorator
    else:
        return decorator(function)


_jax_function_desc = _python_def_op_utils.PyOpDesc(
    "nvidia.dali.plugin.jax.fn",
    "jax_function",
    ["CPU", "GPU"],
    "Transforms ``jax.Array`` processing function into DALI operator",
)
