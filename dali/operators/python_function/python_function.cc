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
        .AddOptionalArg("num_outputs", R"code(Number of outputs)code", 1)
        .MakeInternal();

DALI_SCHEMA(PythonFunction)
        .DocStr("Executes a python function. \n"
                "The operator can be used to execute custom python code within the DALI pipeline. "
                "The called function will get tensors' data as NumPy arrays for CPU operators"
                " or as CuPy arrays for GPU operators and should return results in the same format"
                " (for more universal data format see `DLTensorPythonFunction`). "
                "The function should not modify input tensors. \n\n"
                "For now, this operator can be used only in pipelines with "
                "`exec_async=False` and `exec_pipelined=False` specified. Due to "
                "inferior performance, it is intended for prototyping and debugging.")
        .NumInput(0, 256)
        .AllowSequences()
        .SupportVolumetric()
        .NoPrune()
        .AddParent("PythonFunctionBase")
        .AddOptionalArg("batch_processing",
                        "Whether the function should get the whole batch as input.", false);

DALI_SCHEMA(TorchPythonFunction)
        .DocStr("Executes a function operating on Torch tensors. "
                "Analogous to PythonFunction but tensors' data is handled as PyTorch tensors.")
        .NumInput(0, 256)
        .AllowSequences()
        .SupportVolumetric()
        .NoPrune()
        .AddParent("PythonFunctionBase")
        .AddOptionalArg("batch_processing",
                        "Whether the function should get the whole batch as input.", false);

}  // namespace dali
