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


class BatchToTensor(_ops.Operator):
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

        if (
            _eval_mode.EvalMode.current() is _eval_mode.EvalMode.sync_cpu
            or _eval_mode.EvalMode.current() is _eval_mode.EvalMode.sync_full
            or _op_builder.is_external(batch)
        ):
            # Evaluate immediately
            invocation.run(_eval_context.EvalContext.current())
        elif _eval_mode.EvalMode.current() is _eval_mode.EvalMode.eager:
            with nvtx.annotate("__call__: eager scheduling", domain="op_builder"):
                invocation.schedule(_eval_context.EvalContext.current())
        else:
            pass
        return Tensor(invocation_result=invocation[0])

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


_batch_to_tensor_instance = BatchToTensor(max_batch_size=None)


def batch_to_tensor(batch, pad=False, force_copy=False, device=None):
    return _batch_to_tensor_instance(batch, pad=pad, force_copy=force_copy, device=device)
