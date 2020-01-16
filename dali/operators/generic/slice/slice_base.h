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

#ifndef DALI_OPERATORS_GENERIC_SLICE_SLICE_BASE_H_
#define DALI_OPERATORS_GENERIC_SLICE_SLICE_BASE_H_

#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/scratch.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

#define SLICE_TYPES (uint8_t, int16_t, uint16_t, int32_t, float, float16)
#define CMN_NDIMS (3, 4, 5)

namespace dali {

template <typename Backend>
class SliceBase : public Operator<Backend> {
 public:
  explicit inline SliceBase(const OpSpec &spec)
    : Operator<Backend>(spec)
    , slice_anchors_(batch_size_)
    , slice_shapes_(batch_size_)
    , output_type_(spec.GetArgument<DALIDataType>("output_dtype")) {
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

  void SetupSharedSampleParams(Workspace<Backend> &ws) override {
    const auto &input = ws.template Input<Backend>(0);
    input_type_ = input.type().id();
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = input_type_;
  }

  virtual void DataDependentSetup(Workspace<Backend> &ws) = 0;

  std::vector<std::vector<int64_t>> slice_anchors_, slice_shapes_;
  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;

  // In current implementation scratchpad memory is only used in the GPU kernel
  // In case of using scratchpad in the CPU kernel a scratchpad allocator per thread
  // should be instantiated
  std::conditional_t<std::is_same<Backend, GPUBackend>::value,
    kernels::ScratchpadAllocator, std::vector<kernels::ScratchpadAllocator>> scratch_alloc_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_BASE_H_
