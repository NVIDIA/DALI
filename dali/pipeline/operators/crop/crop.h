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
    const int ndims = shape.size();
    const bool is_volumetric_layout = IsVolumetric(layout);
    const bool is_sequence_layout = IsSequence(layout);
    DALI_ENFORCE((ndims == 4 && is_sequence_layout) ||
                 (ndims == 4 && is_volumetric_layout) ||
                 (ndims == 3 && (layout == DALI_NHWC || layout == DALI_NCHW)),
                 "Unexpected number of dimensions [" + std::to_string(ndims) +
                 "] or layout [" + std::to_string(layout) + "]");

    auto crop_window_gen = GetCropWindowGenerator(data_idx);
    CropWindow win;
    switch (layout) {
      case DALI_NHWC:
      {
        int64_t H = shape[0];
        int64_t W = shape[1];
        int64_t C = shape[2];
        win = crop_window_gen({H, W});
        slice_shapes_[data_idx] = {win.shape[0], win.shape[1], C};
        slice_anchors_[data_idx] = {win.anchor[0], win.anchor[1], 0};
      }
      break;
      case DALI_NCHW:
      {
        int64_t C = shape[0];
        int64_t H = shape[1];
        int64_t W = shape[2];
        win = crop_window_gen({H, W});
        slice_shapes_[data_idx] = {C, win.shape[0], win.shape[1]};
        slice_anchors_[data_idx] = {0, win.anchor[0], win.anchor[1]};
      }
      break;
      case DALI_NFHWC:
      {
        int64_t F = shape[0];
        int64_t H = shape[1];
        int64_t W = shape[2];
        int64_t C = shape[3];
        win = crop_window_gen({H, W});
        slice_shapes_[data_idx] = {F, win.shape[0], win.shape[1], C};
        slice_anchors_[data_idx] = {0, win.anchor[0], win.anchor[1], 0};
      }
      break;
      case DALI_NFCHW:
      {
        int64_t F = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        win = crop_window_gen({H, W});
        slice_shapes_[data_idx] = {F, C, win.shape[0], win.shape[1]};
        slice_anchors_[data_idx] = {0, 0, win.anchor[0], win.anchor[1]};
      }
      break;
      case DALI_NDHWC:
      {
        int64_t D = shape[0];
        int64_t H = shape[1];
        int64_t W = shape[2];
        int64_t C = shape[3];
        win = crop_window_gen({D, H, W});
        slice_shapes_[data_idx] = {win.shape[0], win.shape[1], win.shape[2], C};
        slice_anchors_[data_idx] = {win.anchor[0], win.anchor[1], win.anchor[2], 0};
      }
      break;
      case DALI_NCDHW:
      {
        int64_t C = shape[0];
        int64_t D = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        win = crop_window_gen({D, H, W});
        slice_shapes_[data_idx] = {C, win.shape[0], win.shape[1], win.shape[2]};
        slice_anchors_[data_idx] = {0, win.anchor[0], win.anchor[1], win.anchor[2]};
      }
      break;
      default:
        DALI_FAIL("Not supported layout[" + std::to_string(layout)
                  + "] for given number of dimensions[" + std::to_string(shape.size()) + "]");
    }
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_
