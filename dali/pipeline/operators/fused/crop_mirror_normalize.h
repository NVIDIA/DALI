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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_

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
class CropMirrorNormalize : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec)
      : Operator<Backend>(spec),
        CropAttr(spec),
        output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
        output_layout_(spec.GetArgument<DALITensorLayout>("output_layout")),
        pad_output_(spec.GetArgument<bool>("pad_output")),
        slice_anchors_(batch_size_),
        slice_shapes_(batch_size_),
        mirror_(batch_size_) {
    if (!spec.TryGetRepeatedArgument(mean_vec_, "mean")) {
      mean_vec_ = { spec.GetArgument<float>("mean") };
    }

    if (!spec.TryGetRepeatedArgument(inv_std_vec_, "std")) {
      inv_std_vec_ = { spec.GetArgument<float>("std") };
    }

    // Inverse the std-deviation
    for (auto &element : inv_std_vec_) {
      element = 1.f / element;
    }
  }

  inline ~CropMirrorNormalize() override = default;

 protected:
  void RunImpl(Workspace<Backend> *ws) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override {
    const auto &input = ws->template Input<Backend>(0);
    input_type_ = input.type().id();
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = input_type_;

    input_layout_ = input.GetLayout();
    DALI_ENFORCE(input_layout_ == DALI_NHWC || input_layout_ == DALI_NCHW ||
                 input_layout_ == DALI_NFHWC || input_layout_ == DALI_NFCHW,
      "Unexpected data layout");
    if (output_layout_ == DALI_SAME)
      output_layout_ = input_layout_;

    CropAttr::ProcessArguments(ws);
  }

  void DataDependentSetup(Workspace<Backend> *ws);

  void SetupSample(int data_idx, DALITensorLayout layout, const kernels::TensorShape<> &shape) {
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

    const bool is_whole_image = IsWholeImage();
    int crop_h = is_whole_image ? H : crop_height_[data_idx];
    int crop_w = is_whole_image ? W : crop_width_[data_idx];
    auto crop_pos_y_x = CalculateCropYX(crop_y_norm_[data_idx], crop_x_norm_[data_idx],
                                        crop_h, crop_w, H, W);

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

  std::vector<std::vector<int64_t>> slice_anchors_, slice_shapes_;

  std::vector<float> mean_vec_, inv_std_vec_;
  std::vector<int> mirror_;

  // In current implementation scratchpad memory is only used in the GPU kernel
  // In case of using scratchpad in the CPU kernel a scratchpad allocator per thread
  // should be instantiated
  std::conditional_t<std::is_same<Backend, GPUBackend>::value,
    kernels::ScratchpadAllocator,
    std::vector<kernels::ScratchpadAllocator>> scratch_alloc_;

  USE_OPERATOR_MEMBERS();
};

namespace detail {

inline size_t horizontal_dim_idx(DALITensorLayout layout) {
  switch (layout) {
    case DALI_NHWC:
      return 1;
    case DALI_NCHW:
      return 2;
    case DALI_NFHWC:
      return 2;
    case DALI_NFCHW:
      return 3;
    default:
      DALI_FAIL("not supported layout: " + std::to_string(layout));
  }
}

template <size_t Dims>
inline std::array<int64_t, Dims> permuted_dims(DALITensorLayout in_layout,
                                               DALITensorLayout out_layout) {
  std::array<int64_t, Dims> perm_dims;
  for (size_t d = 0; d < Dims; d++) {
    perm_dims[d] = d;
  }

  if (in_layout != out_layout) {
    if (in_layout == DALI_NHWC && out_layout == DALI_NCHW) {
      perm_dims[0] = 2;
      perm_dims[1] = 0;
      perm_dims[2] = 1;
    } else if (in_layout == DALI_NCHW && out_layout == DALI_NHWC) {
      perm_dims[0] = 1;
      perm_dims[1] = 2;
      perm_dims[2] = 0;
    } else if (in_layout == DALI_NFHWC && out_layout == DALI_NFCHW) {
      perm_dims[1] = 3;
      perm_dims[2] = 1;
      perm_dims[3] = 2;
    } else if (in_layout == DALI_NFCHW && out_layout == DALI_NFHWC) {
      perm_dims[1] = 2;
      perm_dims[2] = 3;
      perm_dims[3] = 1;
    } else {
      DALI_FAIL("layout conversion from " + std::to_string(in_layout) + " to "
        + std::to_string(out_layout) + " not supported");
    }
  }

  return perm_dims;
}

inline size_t channels_dim(DALITensorLayout in_layout) {
  switch (in_layout) {
    case DALI_NHWC:
      return 2;
    case DALI_NCHW:
      return 0;
    case DALI_NFHWC:
      return 3;
    case DALI_NFCHW:
      return 1;
    default:
      DALI_FAIL("not supported layout: " + std::to_string(in_layout));
  }
}

}  // namespace detail


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
