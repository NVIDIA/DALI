// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/util/copy_with_stride.h"

namespace dali {

DALI_SCHEMA(PythonFunctionBase)
    .AddArg("function",
            "Function object.",
            DALI_PYTHON_OBJECT)
    .AddOptionalArg("num_outputs", R"code(Number of outputs.)code", 1)
    .AddOptionalArg<std::vector<TensorLayout>>("output_layouts",
      R"code(Tensor data layouts for the outputs.

This argument can be a list that contains a distinct layout for each output. If the list has
fewer than num_outputs elements, only the first outputs have the layout set and the rest of the
outputs have no layout assigned.)code", nullptr)
    .MakeInternal();

DALI_SCHEMA(PythonFunction)
        .DocStr(R"code(Executes a Python function.

This operator can be used to execute custom Python code in the DALI pipeline.
The function receives the data from DALI as NumPy arrays in case of CPU operators or
as CuPy arrays for GPU operators. It is expected to return the results in the same format. For
a more universal data format, see :meth:`nvidia.dali.fn.dl_tensor_python_function`.
The function should not modify input tensors.

.. warning::
  Currently, this operator can be used only in pipelines with the
  ``exec_async=False`` and ``exec_pipelined=False`` values specified and should only be
  used for prototyping and debugging.

.. warning::
  This operator is not compatible with TensorFlow integration.
)code")
        .NumInput(0, 256)
        .AllowSequences()
        .SupportVolumetric()
        .NoPrune()
        .AddParent("PythonFunctionBase")
        .AddOptionalArg("batch_processing", R"code(Determines whether the function is invoked
once per batch or separately for every sample in the batch.

If set to True, the function will receive its arguments as lists of NumPy or CuPy arrays,
for CPU and GPU backend, respectively.)code", false);

DALI_SCHEMA(TorchPythonFunction)
        .DocStr(R"code(Executes a function that is operating on Torch tensors.

This class is analogous to :meth:`nvidia.dali.fn.python_function` but the tensor data is handled
as PyTorch tensors.)code")
        .NumInput(0, 256)
        .AllowSequences()
        .SupportVolumetric()
        .NoPrune()
        .AddParent("PythonFunctionBase")
        .AddOptionalArg("batch_processing", R"code(Determines whether the function gets
an entire batch as an input.)code", false);

}  // namespace dali
