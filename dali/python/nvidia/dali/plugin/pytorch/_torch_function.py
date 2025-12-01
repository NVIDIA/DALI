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

import torch
import torch.utils.dlpack as torch_dlpack

from nvidia.dali import ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.ops._operators import python_function


class TorchPythonFunction(
    python_function._get_base_impl("TorchPythonFunction", "DLTensorPythonFunctionImpl")
):
    ops.register_cpu_op("TorchPythonFunction")
    ops.register_gpu_op("TorchPythonFunction")

    def _torch_stream_wrapper(self, function, *ins):
        with torch.cuda.stream(self.stream):
            out = function(*ins)
        self.stream.synchronize()
        return out

    def torch_wrapper(self, batch_processing, function, device, *args):
        func = (
            function if device == "cpu" else lambda *ins: self._torch_stream_wrapper(function, *ins)
        )
        if batch_processing:
            return ops.PythonFunction.function_wrapper_batch(
                self.pipeline,
                func,
                self.num_outputs,
                torch.utils.dlpack.from_dlpack,
                torch.utils.dlpack.to_dlpack,
                *args,
            )
        else:
            return ops.PythonFunction.function_wrapper_per_sample(
                self.pipeline,
                func,
                self.num_outputs,
                torch_dlpack.from_dlpack,
                torch_dlpack.to_dlpack,
                *args,
            )

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()._stub()
        if pipeline is None:
            Pipeline._raise_pipeline_required("TorchPythonFunction")
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=pipeline.device_id)
        return super().__call__(*inputs, **kwargs)

    def __init__(self, function, num_outputs=1, device="cpu", batch_processing=False, **kwargs):
        self.stream = None
        super().__init__(
            function=lambda *ins: self.torch_wrapper(batch_processing, function, device, *ins),
            num_outputs=num_outputs,
            device=device,
            batch_processing=batch_processing,
            **kwargs,
        )
