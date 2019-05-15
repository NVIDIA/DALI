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

#ifndef DALI_PIPELINE_OPERATORS_CROP_NEW_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_NEW_CROP_H_

#include <utility>
#include <vector>
#include <tuple>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"
#include "dali/pipeline/operators/crop/crop_attr.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class NewCrop : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline NewCrop(const OpSpec &spec) :
    Operator<Backend>(spec),
    CropAttr(spec),
    C_(IsColor(spec.GetArgument<DALIImageType>("image_type")) ? 3 : 1) {
    Init(batch_size_);
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  std::vector<std::array<int64_t, 3>> slice_anchors_, slice_shapes_;
  DALIDataType input_type_;
  DALIDataType output_type_;
  std::size_t C_;

  USE_OPERATOR_MEMBERS();

 private:
  void DataDependentSetup(Workspace<Backend> *ws, int idx);

  void DataDependentSetup(int data_idx, DALITensorLayout layout, const vector<Index> &shape) {
    Index H, W, C;
    DALI_ENFORCE(shape.size() == 3, "Expected 3-dimensional input");
    switch (layout) {
      case DALI_NHWC:
        std::tie(H, W, C) = std::make_tuple(shape[0], shape[1], shape[2]);
        break;
      case DALI_NCHW:
        std::tie(C, H, W) = std::make_tuple(shape[0], shape[1], shape[2]);
        break;
      default:
        DALI_FAIL("Not supported layout");
    }

    DALI_ENFORCE(H >= crop_height_[data_idx] && W >= crop_width_[data_idx],
      "Image dimensions for sample " + std::to_string(data_idx)
      + " (" + std::to_string(H)
      + ", " + std::to_string(W) + ")"
      + " are smaller than the cropping window"
      + " (" + std::to_string(crop_height_[data_idx])
      + ", " + std::to_string(crop_width_[data_idx]) + ")");

    auto crop_pos_y_x = CalculateCropYX(
      crop_y_norm_[data_idx],
      crop_x_norm_[data_idx],
      crop_height_[data_idx],
      crop_width_[data_idx],
      H, W);

    auto crop_h = crop_height_[data_idx];
    auto crop_w = crop_width_[data_idx];
    auto crop_y = crop_pos_y_x.first;
    auto crop_x = crop_pos_y_x.second;

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
        DALI_FAIL("Not supported layout");
    }
  }

  void Init(int size) {
    slice_anchors_.resize(size);
    slice_shapes_.resize(size);
    output_type_ = DALI_NO_TYPE;
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_NEW_CROP_H_
