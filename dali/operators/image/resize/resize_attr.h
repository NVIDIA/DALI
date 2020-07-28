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
  ResizeAttr(const OpSpec &spec);
  void Initialize(const OpSpec &spec);

  void PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                           const TensorListShape<> &input_shape);

  void PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                           const TensorListShape<> &input_shape,
                           TensorLayout input_layout) {
    ParseLayout(input_layout);
    PrepareResizeParams(spec, ws, input_shape);
  }

  void ParseLayout(const TensorLayout &layout) {
    ParseLayout(spatial_ndim_, first_spatial_dim_, layout);
  }

  static void ParseLayout(int &spatial_ndim, int &first_spatial_dim, const TensorLayout &layout);

  bool HasSeparateSizeArgs() const {
    return has_resize_x_ || has_resize_y_ || has_resize_z_;
  }


  vector<ResizeParams> params_;

  /**
   * Maximum size - used together with with mode NotSmaller to limit the size for
   * very thin images
   */
  vector<float> max_size_;

  // pass sizes by value - the function will modify them internally
  void CalculateSampleParams(ResizeParams &params,
                             SmallVector<float, 3> requested_size,
                             SmallVector<float, 3> input_size);

  bool has_size_ = false;
  bool has_max_size_ = false;
  bool has_mode_ = false;
  bool has_resize_shorter_ = false;
  bool has_resize_longer_ = false;
  bool has_resize_x_ = false;
  bool has_resize_y_ = false;
  bool has_resize_z_ = false;

  int spatial_ndim_ = 2;
  int first_spatial_dim_ = 0;

  ResizeMode mode_ = ResizeMode::Stretch;

 private:
  TensorListShape<> size_arg_;
  vector<float> res_x_, res_y_, res_z_;

  void AdjustOutputSize(float *out_size, const float *in_size);
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_H_
