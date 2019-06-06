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
#include "dali/kernels/scratch.h"

namespace dali {

template <typename Backend>
class NewCropMirrorNormalize : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline NewCropMirrorNormalize(const OpSpec &spec)
      : Operator<Backend>(spec),
        CropAttr(spec),
        output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
        output_layout_(spec.GetArgument<DALITensorLayout>("output_layout")),
        pad_output_(spec.GetArgument<bool>("pad_output")),
        image_type_(spec.GetArgument<DALIImageType>("image_type")),
        C_(IsColor(image_type_) ? 3 : 1),
        slice_anchors_(batch_size_),
        slice_shapes_(batch_size_),
        mirror_(batch_size_) {
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

  void SetupSample(int data_idx, DALITensorLayout layout, const vector<Index> &shape) {
    Index F = 1, H, W, C;
    DALI_ENFORCE(shape.size() == 3 || shape.size() == 4,
      "Unexpected number of dimensions: " + std::to_string(shape.size()));
    switch (layout) {
      case DALI_NHWC:
        std::tie(H, W, C) = std::make_tuple(shape[0], shape[1], shape[2]);
        break;
      case DALI_NCHW:
        std::tie(C, H, W) = std::make_tuple(shape[0], shape[1], shape[2]);
        break;
      case DALI_NFHWC:
        std::tie(F, H, W, C) = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
        break;
      case DALI_NFCHW:
        std::tie(F, C, H, W) = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
        break;
      default:
        DALI_FAIL("Not supported layout");
    }

    DALI_ENFORCE(H >= crop_height_[data_idx] && W >= crop_width_[data_idx],
      "Image dimensions for sample " + std::to_string(data_idx) + " ("
      + std::to_string(H) + ", " + std::to_string(W) + ")"
      + " are smaller than the cropping window" + " ("
      + std::to_string(crop_height_[data_idx]) + ", "
      + std::to_string(crop_width_[data_idx]) + ")");

    auto crop_pos_y_x = CalculateCropYX(crop_y_norm_[data_idx], crop_x_norm_[data_idx],
                                        crop_height_[data_idx], crop_width_[data_idx], H, W);

    auto crop_h = crop_height_[data_idx];
    auto crop_w = crop_width_[data_idx];

    int64_t crop_y, crop_x;
    std::tie(crop_y, crop_x) = crop_pos_y_x;

    switch (layout) {
      case DALI_NHWC:
        slice_anchors_[data_idx] = {crop_y, crop_x, 0};
        slice_shapes_[data_idx] = {crop_h, crop_w, C};
        break;
      case DALI_NCHW:
        slice_anchors_[data_idx] = {0, crop_y, crop_x};
        slice_shapes_[data_idx] = {C, crop_h, crop_w};
        break;
      case DALI_NFHWC:
        slice_anchors_[data_idx] = {0, crop_y, crop_x, 0};
        slice_shapes_[data_idx] = {F, crop_h, crop_w, C};
        break;
      case DALI_NFCHW:
        slice_anchors_[data_idx] = {0, 0, crop_y, crop_x};
        slice_shapes_[data_idx] = {F, C, crop_h, crop_w};
        break;
      default:
        DALI_FAIL("Not supported layout");
    }
  }

  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;

  DALITensorLayout input_layout_ = DALI_NHWC;
  DALITensorLayout output_layout_ = DALI_SAME;

  // Whether to pad output to 4 channels
  bool pad_output_;

  // Input/output channel meta-data
  DALIImageType image_type_;
  int C_;

  std::vector<std::vector<int64_t>> slice_anchors_, slice_shapes_;

  std::vector<float> mean_vec_, inv_std_vec_;
  std::vector<int> mirror_;

  // In current implementation scratchpad memory is only used in the GPU kernel
  // In case of using scratchpad in the CPU kernel a scratchpad allocator per thread
  // should be instantiated
  typename std::conditional<std::is_same<Backend, GPUBackend>::value,
    kernels::ScratchpadAllocator,
    std::vector<kernels::ScratchpadAllocator>>::type scratch_alloc_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_NEW_CROP_MIRROR_NORMALIZE_H_
