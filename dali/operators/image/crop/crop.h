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

#ifndef DALI_OPERATORS_IMAGE_CROP_CROP_H_
#define DALI_OPERATORS_IMAGE_CROP_CROP_H_

#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/scratch.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/generic/slice/slice_base.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

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

  void SetupSample(int data_idx, const TensorLayout &layout, const TensorShape<> &shape) {
    int64_t F = 1, D = 1, H, W, C;
    DALI_ENFORCE(shape.size() >= 3 || shape.size() <= 5,
      "Unexpected number of dimensions: " + std::to_string(shape.size()));
    DALI_ENFORCE(layout.ndim() == shape.size());

    int d_dim = layout.find('D');
    int h_dim = layout.find('H');
    int w_dim = layout.find('W');
    int c_dim = layout.find('C');
    int f_dim = layout.find('F');

    DALI_ENFORCE(h_dim >= 0 && w_dim >= 0 && c_dim >= 0,
      "Height, Width and Channel must be present in the layout. Got: " + layout.str());
    if (d_dim >= 0)
      D = shape[d_dim];
    H = shape[h_dim];
    W = shape[w_dim];
    C = shape[c_dim];
    if (f_dim >= 0)
      F = shape[f_dim];

    int spatial_ndim = ImageLayoutInfo::NumSpatialDims(layout);
    assert(spatial_ndim >= 2);  // bug-check: should never occur with h_dim, w_dim >= 0

    // Special case.
    // This allows using crop_d to crop on the sequence dimension,
    // by treating video inputs as a volume instead of a sequence
    if (has_crop_d_ && F > 1 && D == 1) {
      std::swap(d_dim, f_dim);
      std::swap(D, F);
      spatial_ndim++;
    }

    auto crop_window_gen = GetCropWindowGenerator(data_idx);
    auto win = spatial_ndim == 3 ?
      crop_window_gen({D, H, W}, "DHW") : crop_window_gen({H, W}, "HW");

    int ndim = shape.sample_dim();
    slice_anchors_[data_idx].resize(ndim);
    slice_shapes_[data_idx].resize(ndim);

    if (d_dim >= 0) {
      slice_anchors_[data_idx][d_dim] = win.anchor[spatial_ndim - 3];
      slice_shapes_[data_idx][d_dim] = win.shape[spatial_ndim - 3];
    }

    slice_anchors_[data_idx][h_dim] = win.anchor[spatial_ndim - 2];
    slice_shapes_[data_idx][h_dim] = win.shape[spatial_ndim - 2];

    slice_anchors_[data_idx][w_dim] = win.anchor[spatial_ndim - 1];
    slice_shapes_[data_idx][w_dim] = win.shape[spatial_ndim - 1];

    slice_anchors_[data_idx][c_dim] = 0;
    slice_shapes_[data_idx][c_dim] = C;

    if (f_dim >= 0) {
      slice_anchors_[data_idx][f_dim] = 0;
      slice_shapes_[data_idx][f_dim] = F;
    }
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_CROP_H_
