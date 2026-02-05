# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from . import _ops
from . import _op_builder
from . import _invocation
from . import _eval_context
from . import _eval_mode
from . import _device
from ._batch import as_batch, batch, Batch, Tensor
import nvtx


def is_uniform(shape):
    return all(x == shape[0] for x in shape[1:])


class BatchToTensor:
    """This is an internal pseudo-operator, not to be used directly.

    This operator is used when calling ``tensor`` or ``as_tensor`` with a ``Batch`` argument.
    Unlike any other operator, it takes a ``Batch`` and returns a single ``Tensor``.
    This operator is special in that, unlike any other, it takes a Batch and returns a
    single Tensor. It needs to be an usable as an operator so we can benefit from the
    ``Invocation`` object and correctly apply ``EvalModes`` policies.
    Only a minimal required subset of ``Operator`` interface is implemented.
    """

    def __call__(self, batch, pad=False, force_copy=False, batch_size=None, device=None):
        if not isinstance(batch, Batch):
            batch = _op_builder._to_batch(batch, batch_size)
        with nvtx.annotate("__call__: construct Invocation", domain="op_builder"):
            invocation = _invocation.Invocation(
                self,
                None,
                [batch],
                {
                    "pad": pad,
                    "force_copy": force_copy,
                    "device": device,
                },
                is_batch=False,
                batch_size=None,
                previous_invocation=None,
            )
        invocation.apply_eval_policy(_op_builder.is_external(batch))
        return Tensor(invocation_result=invocation[0])

    def _infer_num_outputs(self, *inputs, **args):
        return 1

    def _infer_output_devices(self, input, device, **_):
        return (_device.device(device) or input.device,)

    def _run(self, ctx, input_batch, *, pad, force_copy, device, **_):
        if input_batch._storage and input_batch._storage.is_dense_tensor():
            return Tensor(input_batch._storage.as_tensor(), device=device, copy=force_copy)._storage

        input_batch = as_batch(input_batch, device=device).evaluate()
        shape = input_batch.shape
        if input_batch._storage.is_dense_tensor():
            return input_batch.evaluate()._storage.as_tensor()
        elif is_uniform(shape):
            return batch(input_batch).evaluate()._storage.as_tensor()
        else:
            if not pad:
                raise ValueError(
                    "The batch has a non-uniform shape. "
                    "To convert it to a tensor, `pad` argument must be ``True``"
                )
            from . import pad as _pad

            return _pad(input_batch).evaluate()._storage.as_tensor()


_batch_to_tensor_instance = BatchToTensor()


def batch_to_tensor(batch, pad=False, force_copy=False, device=None):
    return _batch_to_tensor_instance(batch, pad=pad, force_copy=force_copy, device=device)
