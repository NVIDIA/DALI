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

#include "dali/pipeline/operators/util/reshape.h"

namespace dali {

template <>
std::vector<Index> Reshape<CPUBackend>::GetNewShapeForSample(SampleWorkspace *ws) {
  if (use_input_for_shapes_) {
    auto &new_shape = ws->Input<CPUBackend>(kInShapeIdx);
    DALI_ENFORCE(new_shape.shape().size() == 1, "New shape should be represented as 1-dim tensor");
    return {new_shape.template data<Index>(),
            new_shape.template data<Index>() + new_shape.shape()[0]};
  }
  return new_shape_;
}

template <>
void Reshape<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(kInTensorIdx);
  auto *output = ws->Output<CPUBackend>(kOutTensorIdx);
  std::vector<Index> new_sample_shape = GetNewShapeForSample(ws);
  DALI_ENFORCE(Product(new_sample_shape) == Product(input.shape()),
               "New shape must represent exactly the same number of elements as old shape.");
  output->set_type(input.type());
  output->Resize(new_sample_shape);

  TypeInfo type = input.type();
  type.Copy<CPUBackend, CPUBackend>(output->raw_mutable_data(), input.raw_data(), input.size(), 0);
}

DALI_REGISTER_OPERATOR(Reshape, Reshape<CPUBackend>, CPU);

DALI_SCHEMA(Reshape)
    .DocStr("Reshape the tensor")
    .NumInput(1, 2)
    .NumOutput(1)
    .AddOptionalArg("new_shape", R"code(New shape for each sample.
Shape [-1] means that new shapes will be passed as second input, one shape per sample.)code",
                    std::vector<Index>{-1})
    .EnforceInputLayout(DALI_NHWC);

}  // namespace dali
