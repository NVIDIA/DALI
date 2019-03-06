// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/fused/crop_cast_permute.h"

namespace dali {

DALI_SCHEMA(CropCastPermute)
  .DocStr(R"code(Perform a random crop, data type
cast and permute (from NHWC to NCHW).)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AllowSequences()
  .AddOptionalArg("output_dtype",
      R"code(Output data type. If NO_TYPE is specified, the ouput data type is inferred
 from the input data type.)code", DALI_FLOAT)
  .AddOptionalArg("output_layout",
      R"code(Output tensor data layout)code", DALI_NCHW)
  .AddParent("Crop")  // for image type, crop pos and sizes
  .EnforceInputLayout(DALI_NHWC);

// Register operator
DALI_REGISTER_OPERATOR(CropCastPermute, CropCastPermute<CPUBackend>, CPU);

}  // namespace dali
