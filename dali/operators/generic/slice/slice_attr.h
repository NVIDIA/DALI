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

#ifndef DALI_OPERATORS_GENERIC_SLICE_SLICE_ATTR_H_
#define DALI_OPERATORS_GENERIC_SLICE_SLICE_ATTR_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/crop_window.h"
#include "dali/core/static_switch.h"

#define SLICE_ARGS_TYPES (int32_t, int64_t, float)

namespace dali {

class SliceAttr {
 public:
  explicit inline SliceAttr(const OpSpec &spec)
      : normalized_anchor_(spec.GetArgument<bool>("normalized_anchor")),
        normalized_shape_(spec.GetArgument<bool>("normalized_shape")),
        crop_window_generators_(spec.GetArgument<int>("max_batch_size")) {
    const bool has_axes_arg = spec.HasArgument("axes");
    const bool has_axis_names_arg = spec.HasArgument("axis_names");
    // Process `axis_names` if provided, or if neither `axis_names` nor `axes` are
    if (has_axis_names_arg || !has_axes_arg) {
      axis_names_ = spec.GetArgument<TensorLayout>("axis_names");
      axes_ = {};
    } else {
      // Process `axes` only if provided and `axis_names` isn't
      axes_ = spec.GetRepeatedArgument<int>("axes");
      axis_names_ = TensorLayout{};
    }
  }

  template <typename Backend>
  void ProcessArguments(const workspace_t<Backend> &ws) {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    // slice args are CPU inputs
    auto curr_batch_size = ws.GetInputBatchSize(0);
    const auto &crop_anchor = ws.template InputRef<CPUBackend>(1);
    const auto &crop_shape = ws.template InputRef<CPUBackend>(2);
    DALI_ENFORCE(crop_anchor.type().id() == crop_shape.type().id(),
                 make_string("Anchor and shape should have the same type. Got: ",
                             crop_anchor.type().id(), " and ", crop_shape.type().id()));
    auto args_dtype = crop_anchor.type().id();
    TYPE_SWITCH(args_dtype, type2id, ArgsType, SLICE_ARGS_TYPES, (
      auto anchor_view = view<const ArgsType>(crop_anchor);
      auto shape_view = view<const ArgsType>(crop_shape);
      for (int data_idx = 0; data_idx < curr_batch_size; data_idx++) {
        VerifyArgsShape(anchor_view.tensor_shape(data_idx), shape_view.tensor_shape(data_idx));
        ProcessArgumentsHelper(data_idx,
                               anchor_view.tensor_data(data_idx),
                               shape_view.tensor_data(data_idx));
      }
    ), DALI_FAIL(make_string("Unsupported type of anchor and shape arguments: ", args_dtype)));  // NOLINT
  }

  void ProcessArguments(const SampleWorkspace &ws) {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    const auto &crop_anchor = ws.Input<CPUBackend>(1);
    const auto &crop_shape = ws.Input<CPUBackend>(2);
    DALI_ENFORCE(crop_anchor.type().id() == crop_shape.type().id(),
                  make_string("Anchor and shape should have the same type. Got: ",
                              crop_anchor.type().id(), " and ", crop_shape.type().id()));
    auto args_dtype = crop_anchor.type().id();
    VerifyArgsShape(crop_anchor.shape(), crop_shape.shape());
    TYPE_SWITCH(args_dtype, type2id, ArgsType, SLICE_ARGS_TYPES, (
      ProcessArgumentsHelper(ws.data_idx(),
                             crop_anchor.data<ArgsType>(),
                             crop_shape.data<ArgsType>());
    ), DALI_FAIL(make_string("Unsupported type of anchor and shape arguments: ", args_dtype)));  // NOLINT
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

 private:
  template <typename T>
  void ProcessArgumentsHelper(int data_idx,
                              const T *slice_anchor_data,
                              const T *slice_shape_data) {
    bool normalized_anchor = normalized_anchor_;
    bool normalized_shape  = normalized_shape_;
    // If integer anchor and shape were provided, assume absolute coordinates silently
    if (!std::is_floating_point<T>::value) {
      normalized_anchor = false;
      normalized_shape = false;
    }
    assert(std::is_floating_point<T>::value || (!normalized_anchor && !normalized_shape));

    crop_window_generators_[data_idx] =
      [this, slice_anchor_data, slice_shape_data, normalized_anchor, normalized_shape]
      (const TensorShape<> &shape, const TensorLayout& shape_layout) {
        CropWindow slice;
        slice.anchor = std::vector<int64_t>(shape.size(), 0);
        slice.shape = shape;

        auto axes = axes_;
        if (!axis_names_.empty()) {
          axes = GetDimIndices(shape_layout, axis_names_).to_vector();
        }

        for (size_t i = 0; i < axes.size(); i++) {
          auto dim = axes[i];
          double anchor_val = slice_anchor_data[i];
          double shape_val = slice_shape_data[i];
          int64_t slice_end;
          // special case - minimize the floating point error by multiplying only once after sum
          if (normalized_anchor && normalized_shape) {
            slice_end = std::llround((anchor_val + shape_val) * shape[dim]);
            anchor_val *= shape[dim];
            shape_val *= shape[dim];
          } else {
            if (normalized_anchor) {
              anchor_val *= shape[dim];
            }
            if (normalized_shape) {
              shape_val *= shape[dim];
            }
            slice_end = std::llround(anchor_val + shape_val);
          }
          slice.anchor[dim] = std::llround(anchor_val);
          slice.shape[dim] = slice_end - slice.anchor[dim];
        }

        return slice;
      };
  }

  void VerifyArgsShape(const TensorShape<>& crop_anchor_shape,
                       const TensorShape<>& crop_shape_shape) {
    DALI_ENFORCE(crop_anchor_shape == crop_shape_shape);
    DALI_ENFORCE(crop_anchor_shape.sample_dim() <= 1,
                 "Anchor and shape must be 1D tensors or scalars");
    size_t args_size = volume(crop_anchor_shape);
    auto axes_size = !axis_names_.empty() ? axis_names_.size() : axes_.size();
    DALI_ENFORCE(args_size == axes_size,
      make_string("Unexpected number of arguments ", args_size, " vs ", axes_size));
  }

 private:
  bool normalized_anchor_, normalized_shape_;
  std::vector<CropWindowGenerator> crop_window_generators_;
  std::vector<int> axes_;
  TensorLayout axis_names_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_ATTR_H_
