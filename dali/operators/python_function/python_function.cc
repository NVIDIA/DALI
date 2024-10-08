// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            R"code(A callable object that defines the function of the operator.

.. warning::
    The function must not hold a reference to the pipeline in which it is used. If it does,
    a circular reference to the pipeline will form and the pipeline will never be freed.)code",
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
    This operator is not compatible with TensorFlow integration.

.. warning::
    When the pipeline has conditional execution enabled, additional steps must be taken to
    prevent the ``function`` from being rewritten by AutoGraph.
    There are two ways to achieve this:

        1. Define the function at global scope (i.e. outside of ``pipeline_def`` scope).

        2. If function is a result of another "factory" function, then the factory function
           must be defined outside pipeline definition function and decorated with
           :meth:`@do_not_convert <nvidia.dali.pipeline.do_not_convert>`.

    More details can be found in :meth:`@do_not_convert <nvidia.dali.pipeline.do_not_convert>`
    documentation.
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
an entire batch as an input.)code", true);

}  // namespace dali
