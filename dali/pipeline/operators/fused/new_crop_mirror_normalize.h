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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_NEW_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_NEW_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/crop/crop_attr.h"

namespace dali {

template <typename Backend>
class NewCropMirrorNormalize : public Operator<Backend>, protected CropAttr  {
 public:
  explicit inline NewCropMirrorNormalize(const OpSpec &spec)
      : Operator<Backend>(spec),
        CropAttr(spec),
        output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
        output_layout_(spec.GetArgument<DALITensorLayout>("output_layout")),
        pad_(spec.GetArgument<bool>("pad_output")),
        image_type_(spec.GetArgument<DALIImageType>("image_type")),
        color_(IsColor(image_type_)),
        C_(color_ ? 3 : 1),
        slice_anchors_(batch_size_),
        slice_shapes_(batch_size_) {
    if (!spec.HasTensorArgument("mirror")) {
      mirror_.Resize({batch_size_});
      for (int i = 0; i < batch_size_; ++i) {
        mirror_.mutable_data<int>()[i] = spec.GetArgument<int>("mirror");
      }
    }

    GetSingleOrRepeatedArg(spec, mean_vec_, "mean", C_);
    GetSingleOrRepeatedArg(spec, inv_std_vec_, "std", C_);
    // Inverse the std-deviation
    for (auto &element : inv_std_vec_) {
      element = 1.f / element;
    }
  }

  inline ~NewCropMirrorNormalize() override = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override {
    const auto &input = ws->template Input<Backend>(0);
    input_type_ = input.type().id();
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = input_type_;

    input_layout_ = input.GetLayout();
    if (output_layout_ == DALI_SAME)
      output_layout_ = input_layout_;

    CropAttr::ProcessArguments(ws);
  }

  void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  // Output data type
  DALIDataType output_type_ = DALI_NO_TYPE;

  // Output data layout
  DALITensorLayout output_layout_ = DALI_SAME;

  // Whether to pad output to 4 channels
  bool pad_;

  // Input/output channel meta-data
  DALIImageType image_type_;
  bool color_;
  int C_;

  std::vector<std::vector<int64_t>> slice_anchors_, slice_shapes_;
  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;

  // Tensor to store mean & stddiv
  std::vector<float> mean_vec_, inv_std_vec_;

  DALITensorLayout input_layout_;
  DALITensorLayout output_layout_ = DALI_SAME;

  // In current implementation scratchpad memory is only used in the GPU kernel
  // In case of using scratchpad in the CPU kernel a scratchpad allocator per thread
  // should be instantiated
  typename std::conditional<std::is_same<Backend, GPUBackend>::value,
    kernels::ScratchpadAllocator, std::vector<kernels::ScratchpadAllocator>>::type scratch_alloc_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_NEW_CROP_MIRROR_NORMALIZE_H_
