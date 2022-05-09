// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/static_switch.h"
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/pipeline/operator/false_gpu_operator.h"
#include "dali/pipeline/data/views.h"

namespace dali {

class NonsilenceOperatorGpu : public FalseGPUOperator<NonsilenceOperatorCpu> {
 public:
  explicit NonsilenceOperatorGpu(const OpSpec &spec)
      : FalseGPUOperator<NonsilenceOperatorCpu>(spec) {}
  ~NonsilenceOperatorGpu() override = default;
  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorGpu);
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorGpu, GPU);

}  // namespace dali
