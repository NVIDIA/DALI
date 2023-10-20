// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include "dali/operators/generic/resize/tensor_resize.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"


namespace dali {
namespace tensor_resize {

template <>
void TensorResize<CPUBackend>::InitializeBackend() {
  this->InitializeCPU(num_threads_);
}

class TensorResizeCPU : public TensorResize<CPUBackend> {
 public:
  explicit TensorResizeCPU(const OpSpec &spec)
      : TensorResize<CPUBackend>(spec) {}
};

}  // namespace tensor_resize


DALI_SCHEMA(experimental__TensorResize)
    .DocStr(R"code(Resize tensors.)code")
    .NumInput(1)
    .NumOutput(1)
    .SupportVolumetric()
    .AllowSequences()
    .AddParent("ResamplingFilterAttr")
    .AddParent("TensorResizeAttr");


DALI_REGISTER_OPERATOR(experimental__TensorResize, tensor_resize::TensorResizeCPU, CPU);


}  // namespace dali
