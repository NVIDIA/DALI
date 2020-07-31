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
                R"code(Function object.)code",
                DALI_PYTHON_OBJECT)
        .AddOptionalArg("num_outputs", R"code(Number of outputs.)code", 1)
        .MakeInternal();

DALI_SCHEMA(PythonFunction)
        .DocStr(R"code(Executes a Python function.

This operator can be used to execute custom Python code in the DALI pipeline.
The function that is called gets the data from the tensoras NumPy arrays for CPU operators or
as CuPy arrays for GPU operators. The results should be returned in the same format, but for
a more universal data format, see :meth:`nvidia.dali.ops.DLTensorPythonFunction`.
The function should not modify input tensors.

Important: Currently, this operator can be used only in pipelines with the
``exec_async=False`` and ``exec_pipelined=False`` values specified and should only be
used for prototyping and debugging.)code")
        .NumInput(0, 256)
        .AllowSequences()
        .SupportVolumetric()
        .NoPrune()
        .AddParent("PythonFunctionBase")
        .AddOptionalArg("batch_processing", R"code(Determines whether the function should get
the entire batch as input.)code", false);

DALI_SCHEMA(TorchPythonFunction)
        .DocStr(R"code(Executes a function that is operating on Torch tensors.

This class is analogous to :meth:`nvidia.dali.ops.PythonFunction` but the tensor data is handled
as PyTorch tensors.)code")
        .NumInput(0, 256)
        .AllowSequences()
        .SupportVolumetric()
        .NoPrune()
        .AddParent("PythonFunctionBase")
        .AddOptionalArg("batch_processing", R"code(Determies whether the function should get
the entire batch as the input.)code", false);

}  // namespace dali
