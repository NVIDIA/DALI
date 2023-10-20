// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_TENSOR_RESIZE_ATTR_H_
#define DALI_OPERATORS_IMAGE_RESIZE_TENSOR_RESIZE_ATTR_H_

#include <string>
#include <vector>
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/operators/image/resize/resize_attr_base.h"
#include "dali/operators/util/axes_utils.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

class DLL_PUBLIC TensorResizeAttr {
 public:
  explicit TensorResizeAttr(const OpSpec &spec);

  void PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                           const TensorListShape<> &input_shape, const TensorLayout &layout);

  /**
   * @brief Gets the shape after resizing.
   */
  template <int out_ndim, int in_ndim>
  void GetResizedShape(TensorListShape<out_ndim> &out_shape,
                       const TensorListShape<in_ndim> &in_shape) const {
    int N = in_shape.num_samples();
    assert(static_cast<int>(params_.size()) == N);
    out_shape = in_shape;
    for (int i = 0; i < N; i++) {
      auto out_sample_shape = out_shape.tensor_shape_span(i);
      for (int d = 0; d < spatial_ndim_; d++)
        out_sample_shape[d + first_spatial_dim_] = params_[i].dst_size[d];
    }
  }

  int NumSpatialDims() const {
    return spatial_ndim_;
  }

  int FirstSpatialDim() const {
    return first_spatial_dim_;
  }

  span<const ResizeParams> Params() const {
    return make_cspan(params_);
  }

 private:
  const float* PrepareMaxSize(const OpSpec &spec, span<const int> axes);

  AxesHelper axes_helper_;
  bool has_sizes_ = false;
  bool has_scales_ = false;
  bool has_max_size_ = false;
  bool has_mode_ = false;
  bool has_roi_ = false;
  bool has_alignment_ = false;
  bool roi_relative_ = false;
  bool subpixel_scale_ = true;
  ResizeMode mode_ = ResizeMode::Stretch;
  vector<float> sizes_, sizes_arg_;
  vector<float> scales_, scales_arg_;
  vector<float> roi_start_, roi_start_arg_, roi_end_, roi_end_arg_;
  vector<float> alignment_, alignment_arg_;
  vector<float> max_size_, max_size_arg_;
  std::function<int(float)> scale_round_fn_;
  int spatial_ndim_ = -1;
  int first_spatial_dim_ = -1;
  vector<ResizeParams> params_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_TENSOR_RESIZE_ATTR_H_
