// Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/debug/dump_image.h"
#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/util/image.h"

namespace dali {

template<>
void DumpImage<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  DALI_ENFORCE(input.ndim() == 3,
               make_string("Input images must have three dimensions, got input with `",
                           input.ndim(), "` dimensions."));

  int h = input.dim(0);
  int w = input.dim(1);
  int c = input.dim(2);
  DALI_ENFORCE(c == 1 || c == 3,
               make_string("Only 3-channel and gray images are supported, got input with `", c,
                           "` channels."));

  WriteHWCImage(input.template data<uint8>(),
      h, w, c, std::to_string(ws.data_idx()) + "-" + suffix_ + "-" + std::to_string(0));

  // Forward the input
  output.Copy(input, 0);
}

DALI_REGISTER_OPERATOR(DumpImage, DumpImage<CPUBackend>, CPU);

DALI_SCHEMA(DumpImage)
  .DocStr(R"code(Save images in batch to disk in PPM format.

Useful for debugging.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("suffix",
      R"code(Suffix to be added to output file names.)code", std::string())
  .AddOptionalArg("input_layout",
      R"code(Layout of input images.)code", TensorLayout("HWC"));

}  // namespace dali
