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

#ifndef DALI_OPERATORS_CROP_SLICE_ATTR_H_
#define DALI_OPERATORS_CROP_SLICE_ATTR_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_layout.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/crop_window.h"
#include "dali/core/tensor_layout.h"

namespace dali {

class SliceAttr {
 protected:
  explicit inline SliceAttr(const OpSpec &spec)
      : batch_size__(spec.GetArgument<int>("batch_size"))
      , normalized_anchor_(spec.GetArgument<bool>("normalized_anchor"))
      , normalized_shape_(spec.GetArgument<bool>("normalized_shape"))
      , crop_window_generators_(batch_size__) {
    const bool has_dims_arg = spec.HasArgument("dims");
    const bool has_dim_names_arg = spec.HasArgument("dim_names");
    if (has_dim_names_arg) {
      // Process dim names
      dim_names_ = spec.GetArgument<TensorLayout>("dim_names");
    } else {
      // Process dims
      dims_ = spec.GetRepeatedArgument<int>("dims");
    }
  }

  void ProcessArguments(MixedWorkspace &ws) {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    for (std::size_t data_idx = 0; data_idx < batch_size__; data_idx++) {
      const auto &images = ws.Input<CPUBackend>(0, data_idx);
      const auto &crop_anchor = ws.Input<CPUBackend>(1, data_idx);
      const auto &crop_shape = ws.Input<CPUBackend>(2, data_idx);
      auto args_shape = crop_anchor.shape();
      DALI_ENFORCE(args_shape == crop_shape.shape());
      ProcessArgumentsHelper(data_idx, images.GetLayout(), images.shape(), args_shape,
                             crop_anchor.data<float>(), crop_shape.data<float>());
    }
  }

  void ProcessArguments(DeviceWorkspace &ws) {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    const auto &images = ws.Input<GPUBackend>(0);
    const auto &crop_anchor = ws.Input<CPUBackend>(1);
    const auto &crop_shape = ws.Input<CPUBackend>(2);
    for (std::size_t data_idx = 0; data_idx < batch_size__; data_idx++) {
      auto args_shape = crop_anchor.tensor_shape(data_idx);
      DALI_ENFORCE(args_shape == crop_shape.tensor_shape(data_idx));
      ProcessArgumentsHelper(data_idx, images.GetLayout(), images.tensor_shape(data_idx),
                             args_shape, crop_anchor.tensor<float>(data_idx),
                             crop_shape.tensor<float>(data_idx));
    }
  }

  void ProcessArguments(const SampleWorkspace &ws) {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    const auto &images = ws.Input<CPUBackend>(0);
    const auto &crop_anchor = ws.Input<CPUBackend>(1);
    const auto &crop_shape = ws.Input<CPUBackend>(2);
    auto args_shape = crop_anchor.shape();
    DALI_ENFORCE(args_shape == crop_shape.shape());
    ProcessArgumentsHelper(ws.data_idx(), images.GetLayout(), images.shape(), args_shape,
                           crop_anchor.data<float>(), crop_shape.data<float>());
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

 private:
  void ProcessArgumentsHelper(int data_idx,
                              TensorLayout in_layout,
                              const kernels::TensorShape<> &shape,
                              const kernels::TensorShape<> &args_shape,
                              const float *slice_anchor_data,
                              const float *slice_shape_data) {
    // TODO(janton): verify args_shape is as expected
    std::cout << "args shape " << args_shape[0] << std::endl;
    std::cout << "batch_size_ " << batch_size__ << std::endl;
    std::cout << "dims size " << dims_.size() << std::endl;

    auto slice_shape = shape;
    auto slice_anchor = shape;
    for (auto &x : slice_anchor)
      x = 0;

    if (normalized_shape_) {
      for (auto dim : dims_) {
        float anchor_val = slice_anchor_data[dim];
        float shape_val = slice_shape_data[dim];
        DALI_ENFORCE(anchor_val + shape_val <= 1.0f,
          make_string("anchor[", dim, "] + crop[", dim, "] must be <= 1.0f"));
        slice_anchor[dim] = static_cast<int64_t>(anchor_val * shape[dim] + 0.5f);
        slice_shape[dim] = static_cast<int64_t>(shape_val * shape[dim] + 0.5f);
        assert(anchor_val + shape_val <= shape[dim]);
      }
    } else {
      for (auto dim : dims_) {
        auto anchor_val = static_cast<int64_t>(slice_anchor_data[dim]);
        auto shape_val = static_cast<int64_t>(slice_shape_data[dim]);
        DALI_ENFORCE(anchor_val + shape_val <= shape[dim],
          make_string("anchor[", dim, "] + crop[", dim, "] must be <=", shape[dim]));
        slice_anchor[dim] = anchor_val;
        slice_shape[dim] = shape_val;
      }
    }

    crop_window_generators_[data_idx] =
      [this, slice_shape, slice_anchor](const kernels::TensorShape<> &shape) {
        CropWindow crop_window;
        crop_window.anchor = slice_anchor;
        crop_window.shape = slice_shape;
        DALI_ENFORCE(crop_window.IsInRange(shape));
        return crop_window;
      };
  }

  size_t batch_size__;
  bool normalized_anchor_, normalized_shape_;
  std::vector<CropWindowGenerator> crop_window_generators_;
  std::vector<int> dims_;
  TensorLayout dim_names_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_CROP_SLICE_ATTR_H_
