// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_H_

#include <vector>
#include "dali/core/small_vector.h"
#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/operators/image/resize/resize_mode.h"

namespace dali {

struct ResizeParams {
  void resize(int ndim) {
    dst_size.resize(ndim);
    src_lo.resize(ndim);
    src_hi.resize(ndim);
  }
  int size() const { return dst_size.size(); }
  SmallVector<int, 6> dst_size;
  SmallVector<float, 6> src_lo, src_hi;
};

class DLL_PUBLIC ResizeAttr {
 public:
  void PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                           const TensorListShape<> &input_shape);

  void PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                           const TensorListShape<> &input_shape,
                           TensorLayout input_layout) {
    SetLayout(input_layout);
    PrepareResizeParams(spec, ws, input_shape);
  }

  void SetLayout(const TensorLayout &layout) {
    ParseLayout(spatial_ndim_, first_spatial_dim_, layout);
  }

  static void ParseLayout(int &spatial_ndim, int &first_spatial_dim, const TensorLayout &layout);

  bool HasSeparateSizeArgs() const {
    return has_resize_x_ || has_resize_y_ || has_resize_z_;
  }

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

  vector<ResizeParams> params_;

  /**
   * Maximum size - used together with with mode NotSmaller to limit the size for
   * very thin images
   */
  vector<float> max_size_;

  bool has_size_ = false;
  bool has_max_size_ = false;
  bool has_mode_ = false;
  bool has_resize_shorter_ = false;
  bool has_resize_longer_ = false;
  bool has_resize_x_ = false;
  bool has_resize_y_ = false;
  bool has_resize_z_ = false;
  bool has_roi_ = false;
  bool roi_relative_ = false;

  int spatial_ndim_ = 2;
  int first_spatial_dim_ = 0;
  bool subpixel_scale_ = true;

  ResizeMode mode_ = ResizeMode::Stretch;

 private:
  vector<float> size_arg_;
  vector<float> res_x_, res_y_, res_z_;
  vector<float> roi_start_, roi_end_;

  void SetFlagsAndMode(const OpSpec &spec);

  void AdjustOutputSize(float *out_size, const float *in_size);

  void CalculateInputRoI(SmallVector<float, 3> &in_lo,
                         SmallVector<float, 3> &in_hi,
                         const TensorListShape<> &input_shape,
                         int sample_idx) const;

  // pass sizes by value - the function will modify them internally
  void CalculateSampleParams(ResizeParams &params,
                             SmallVector<float, 3> requested_size,
                             SmallVector<float, 3> in_lo,
                             SmallVector<float, 3> in_hi,
                             bool adjust_roi,
                             bool empty_input);
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_H_
