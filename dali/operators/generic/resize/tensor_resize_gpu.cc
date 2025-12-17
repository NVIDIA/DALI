// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
void TensorResize<GPUBackend>::InitializeBackend() {
  this->InitializeGPU(spec_.GetArgument<int>("minibatch_size"),
                      spec_.GetArgument<int64_t>("temp_buffer_hint"));
}

class TensorResizeGPU : public TensorResize<GPUBackend> {
 public:
  explicit TensorResizeGPU(const OpSpec &spec)
      : TensorResize<GPUBackend>(spec) {}
};

}  // namespace tensor_resize

// Kept for backwards compatibility
DALI_REGISTER_OPERATOR(experimental__TensorResize, tensor_resize::TensorResizeGPU, GPU);

DALI_REGISTER_OPERATOR(TensorResize, tensor_resize::TensorResizeGPU, GPU);

}  // namespace dali
