// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/operators/image/resize/resize_mode.h"

namespace dali {

struct ResizeParams {
  SmallVector<int> dst_size;
  SmallVector<float> src_lo, src_hi;
};

class ResizeAttr {
 public:
  ResizeAttr(const OpSpec &spec);

  void PrepareParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                     const TensorListshape<> &input_shape,
                     TensorLayout input_layout = {});

  vector<ResizeParams> params_;
  /// Output size - only spatial dimensions (no channels, frames, etc.)
  TensorListShape<> out_size_;

  bool has_size_ = false;
  bool has_mode_ = false;
  bool has_resize_shorter_ = false;
  bool has_resize_longer_ = false;
  bool has_resize_x_ = false;
  bool has_resize_y_ = false;
  bool has_resize_z_ = false;

};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_H_
