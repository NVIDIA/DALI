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

#ifndef DALI_PIPELINE_OPERATORS_CROP_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_CROP_H_

#include <utility>
#include <vector>
#include <tuple>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/crop_attr.h"
#include "dali/pipeline/operators/crop/slice_base.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Crop : public SliceBase<Backend>, protected CropAttr {
 public:
  explicit inline Crop(const OpSpec &spec) :
    SliceBase<Backend>(spec),
    CropAttr(spec),
    C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
  }

 protected:
  void RunImpl(Workspace<Backend> &ws) override {
    SliceBase<Backend>::RunImpl(ws);
  }

  void SetupSharedSampleParams(Workspace<Backend> &ws) override {
    CropAttr::ProcessArguments(ws);
    SliceBase<Backend>::SetupSharedSampleParams(ws);
  }

  void DataDependentSetup(Workspace<Backend> &ws) override;

  using SliceBase<Backend>::slice_anchors_;
  using SliceBase<Backend>::slice_shapes_;
  using SliceBase<Backend>::input_type_;
  using SliceBase<Backend>::output_type_;
  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
  std::size_t C_;

  void SetupSample(int data_idx, DALITensorLayout layout, const kernels::TensorShape<> &shape) {
    Index F = 1, D = 1, H, W, C;
    const int ndims = shape.size();
    const bool is_volumetric_layout = IsVolumetric(layout);
    const bool is_sequence_layout = IsSequence(layout);
    DALI_ENFORCE((ndims == 4 && is_sequence_layout) ||
                 (ndims == 4 && is_volumetric_layout) ||
                 (ndims == 3 && (layout == DALI_NHWC || layout == DALI_NCHW)),
                 "Unexpected number of dimensions [" + std::to_string(ndims) +
                 "] or layout [" + std::to_string(layout) + "]");
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
      case DALI_NDHWC:
        std::tie(D, H, W, C) = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
        break;
      case DALI_NCDHW:
        std::tie(C, D, H, W) = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
        break;
      default:
        DALI_FAIL("Not supported layout[" + std::to_string(layout)
                  + "] for given number of dimensions");
    }

    auto crop_h = crop_height_[data_idx];
    if (crop_h <= 0)
        crop_h = H;
    auto crop_w = crop_width_[data_idx];
    if (crop_w <= 0)
        crop_w = W;

    if (is_volumetric_layout) {
      auto crop_d = (!crop_depth_.empty() && crop_depth_[data_idx] > 0) ? crop_depth_[data_idx] : D;
      if (crop_d <= 0)
          crop_d = D;
      auto crop_z_norm = !crop_z_norm_.empty() ? crop_z_norm_[data_idx] : 0;

      float anchor_norm[3] =
        {crop_z_norm, crop_y_norm_[data_idx], crop_x_norm_[data_idx]};
      auto crop_anchor = CalculateAnchor(make_span(anchor_norm),
                                         {crop_d, crop_h, crop_w},
                                         {D, H, W});
      int64_t crop_z = crop_anchor[0];
      int64_t crop_y = crop_anchor[1];
      int64_t crop_x = crop_anchor[2];
      switch (layout) {
        case DALI_NDHWC:
          slice_anchors_[data_idx] = {crop_z, crop_y, crop_x, 0};
          slice_shapes_[data_idx] = {crop_d, crop_h, crop_w, C};
          break;
        case DALI_NCDHW:
          slice_anchors_[data_idx] = {0, crop_z, crop_y, crop_x};
          slice_shapes_[data_idx] = {C, crop_d, crop_h, crop_w};
          break;
        default:
          DALI_FAIL("Not supported layout[" + std::to_string(layout)
                    + "] for given number of dimensions");
      }
    } else if (is_sequence_layout) {
      float anchor_norm[2] =
        {crop_y_norm_[data_idx], crop_x_norm_[data_idx]};
      auto crop_anchor = CalculateAnchor(make_span(anchor_norm),
                                         {crop_h, crop_w},
                                         {H, W});
      int64_t crop_y = crop_anchor[0];
      int64_t crop_x = crop_anchor[1];
      switch (layout) {
        case DALI_NFHWC:
          slice_anchors_[data_idx] = {0, crop_y, crop_x, 0};
          slice_shapes_[data_idx] = {F, crop_h, crop_w, C};
          break;
        case DALI_NFCHW:
          slice_anchors_[data_idx] = {0, 0, crop_y, crop_x};
          slice_shapes_[data_idx] = {F, C, crop_h, crop_w};
          break;
        default:
          DALI_FAIL("Not supported layout[" + std::to_string(layout)
                    + "] for given number of dimensions");
      }
    } else {
      float anchor_norm[2] = {crop_y_norm_[data_idx], crop_x_norm_[data_idx]};
      auto crop_anchor = CalculateAnchor(make_span(anchor_norm), {crop_h, crop_w}, {H, W});
      int64_t crop_y = crop_anchor[0];
      int64_t crop_x = crop_anchor[1];
      switch (layout) {
        case DALI_NHWC:
          slice_anchors_[data_idx] = {crop_y, crop_x, 0};
          slice_shapes_[data_idx] = {crop_h, crop_w, C};
          break;
        case DALI_NCHW:
          slice_anchors_[data_idx] = {0, crop_y, crop_x};
          slice_shapes_[data_idx] = {C, crop_h, crop_w};
          break;
        default:
          DALI_FAIL("Not supported layout[" + std::to_string(layout)
                    + "] for given number of dimensions");
      }
    }
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_
