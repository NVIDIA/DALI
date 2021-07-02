// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <limits>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/util/crop_window.h"
#include "dali/core/static_switch.h"

#define SLICE_ARGS_TYPES (int32_t, int64_t, float)

namespace dali {

/**
 * @brief Process axes arguments.
 * @return true if there are axes defined, false otherwise (use all axes).
 */
bool ProcessAxesArgs(std::vector<int>& axes, TensorLayout& axis_names, const OpSpec& spec,
                     const char* axes_arg_name = "axes",
                     const char* axis_names_arg_name = "axis_names");

class NamedSliceAttr {
 public:
  explicit inline NamedSliceAttr(const OpSpec &spec,
                                 const char* start_name = "start",
                                 const char* rel_start_name = "rel_start",
                                 const char* end_name = "end",
                                 const char* rel_end_name = "rel_end",
                                 const char* shape_name = "shape",
                                 const char* rel_shape_name = "rel_shape",
                                 const char* axes_arg_name = "axes",
                                 const char* axis_names_arg_name = "axis_names")
      : spec_(spec),
        start_(start_name, spec),
        rel_start_(rel_start_name, spec),
        end_(end_name, spec),
        rel_end_(rel_end_name, spec),
        shape_(shape_name, spec),
        rel_shape_(rel_shape_name, spec) {
    int max_batch_sz = spec.GetArgument<int>("max_batch_size");
    crop_window_generators_.resize(max_batch_sz);
    use_all_axes_ = !ProcessAxesArgs(axes_, axis_names_, spec, axes_arg_name, axis_names_arg_name);

    has_start_ = start_.IsDefined() || rel_start_.IsDefined();

    if ((start_.IsDefined() + rel_start_.IsDefined()) > 1)
      DALI_FAIL(make_string("\"", start_name, "\" and \"", rel_start_name,
                            "\" arguments are mutually exclusive"));

    has_end_ = end_.IsDefined() || rel_end_.IsDefined();
    has_shape_ = shape_.IsDefined() || rel_shape_.IsDefined();

    if ((end_.IsDefined() + rel_end_.IsDefined() + shape_.IsDefined() + rel_shape_.IsDefined()) > 1)
      DALI_FAIL(make_string("\"", end_name, "\", \"", rel_end_name, "\", \"", shape_name,
                            "\", and \"", rel_shape_name, "\" arguments are mutually exclusive"));
  }

  template <typename Backend>
  bool ProcessArguments(const workspace_t<Backend> &ws, int curr_batch_size = -1, int ndim = -1) {
    if (curr_batch_size < 0)
      curr_batch_size = ws.GetInputBatchSize(0);
    if (ndim < 0)
      ndim = ws.GetInputDim(0);

    auto args_shape = TensorShape<1>(ndim);
    if (!use_all_axes_) {
      args_shape[0] = std::max(static_cast<int>(axes_.size()),
                               static_cast<int>(axis_names_.size()));
    }

    if (start_.IsDefined())
      start_.Acquire(spec_, ws, curr_batch_size, args_shape);
    else if (rel_start_.IsDefined())
      rel_start_.Acquire(spec_, ws, curr_batch_size, args_shape);

    if (end_.IsDefined())
      end_.Acquire(spec_, ws, curr_batch_size, args_shape);
    else if (rel_end_.IsDefined())
      rel_end_.Acquire(spec_, ws, curr_batch_size, args_shape);
    else if (shape_.IsDefined())
      shape_.Acquire(spec_, ws, curr_batch_size, args_shape);
    else if (rel_shape_.IsDefined())
      rel_shape_.Acquire(spec_, ws, curr_batch_size, args_shape);

    for (int data_idx = 0; data_idx < curr_batch_size; data_idx++) {
      ProcessNamedArgs(data_idx);
    }
    return has_start_ || has_end_ || has_shape_;
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

 private:
  void ProcessNamedArgs(int data_idx) {
    crop_window_generators_[data_idx] =
      [this, data_idx](const TensorShape<> &shape, const TensorLayout& shape_layout) {
        CropWindow slice;
        slice.anchor = std::vector<int64_t>(shape.size(), 0);
        slice.shape = shape;

        auto axes = axes_;
        if (!axis_names_.empty()) {
          axes = GetDimIndices(shape_layout, axis_names_).to_vector();
        } else if (use_all_axes_) {
          axes.resize(shape.sample_dim());
          std::iota(axes.begin(), axes.end(), 0);
        }

        constexpr double i64min = static_cast<double>(std::numeric_limits<int64_t>::min());
        constexpr double i64max = static_cast<double>(std::numeric_limits<int64_t>::max());

        for (size_t i = 0; i < axes.size(); i++) {
          auto dim = axes[i];

          double anchor_val = 0;
          if (start_.IsDefined()) {
            anchor_val = start_[data_idx].data[i];
          } else if (rel_start_.IsDefined()) {
            anchor_val = rel_start_[data_idx].data[i] * shape[dim];
          }
          DALI_ENFORCE(anchor_val >= i64min && anchor_val <= i64max,
                       make_string("anchor value out of range [", i64min, ", ", i64max,
                                   "]. Got: ", anchor_val));

          double end_val = shape[dim];
          if (end_.IsDefined()) {
            end_val = end_[data_idx].data[i];
          } else if (rel_end_.IsDefined()) {
            end_val = rel_end_[data_idx].data[i] * shape[dim];
          } else if (shape_.IsDefined()) {
            double shape_val = shape_[data_idx].data[i];
            DALI_ENFORCE(shape_val >= 0 && shape_val <= i64max,
              make_string("shape value out of range [", 0, ", ", i64max, "]. Got: ", shape_val));

            end_val = anchor_val + shape_val;
          } else if (rel_start_.IsDefined() && rel_shape_.IsDefined()) {
            // special case - minimize the floating point error by multiplying only once after sum
            double rel_start_val = rel_start_[data_idx].data[i];
            double rel_shape_val = rel_shape_[data_idx].data[i];
            DALI_ENFORCE(rel_shape_val >= 0,
              make_string("negative shapes are not allowed. Got: ", rel_shape_val));

            end_val = (rel_start_val + rel_shape_val) * shape[dim];
          } else if (rel_shape_.IsDefined()) {
            double shape_val = rel_shape_[data_idx].data[i] * shape[dim];
            DALI_ENFORCE(shape_val >= 0 && shape_val <= i64max,
                         make_string("shape value out of range [", 0, ", ", i64max,
                                     "]. Got: ", shape_val));
            end_val = anchor_val + shape_val;
          }
          DALI_ENFORCE(end_val >= i64min && end_val <= i64max,
                       make_string("end coordinates out of range [", i64min, ", ", i64max,
                                   "]. Got: ", end_val));

          DALI_ENFORCE(end_val >= anchor_val,
                       make_string("end coordinates can't be before start coordinates. Got: start=",
                                   anchor_val, " end=", end_val));

          slice.anchor[dim] = std::llround(anchor_val);
          slice.shape[dim] = std::llround(end_val) - slice.anchor[dim];
        }
        return slice;
      };
  }

 private:
  const OpSpec &spec_;

  std::vector<int> axes_;
  TensorLayout axis_names_;

  ArgValue<int, 1> start_;
  ArgValue<float, 1> rel_start_;

  ArgValue<int, 1> end_;
  ArgValue<float, 1> rel_end_;

  ArgValue<int, 1> shape_;
  ArgValue<float, 1> rel_shape_;

  std::vector<CropWindowGenerator> crop_window_generators_;

  bool has_start_, has_end_, has_shape_;
  bool use_all_axes_ = false;
};


class PositionalSliceAttr {
 public:
  explicit inline PositionalSliceAttr(const OpSpec& spec) : spec_(spec) {
    normalized_anchor_ = spec.GetArgument<bool>("normalized_anchor");
    normalized_shape_ = spec.GetArgument<bool>("normalized_shape");
    int max_batch_sz = spec.GetArgument<int>("max_batch_size");
    crop_window_generators_.resize(max_batch_sz);
    use_all_axes_ = !ProcessAxesArgs(axes_, axis_names_, spec, "axes", "axis_names");
  }

  template <typename Backend>
  bool ProcessArguments(const workspace_t<Backend> &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    int ndim = ws.GetInputDim(0);

    auto args_shape = TensorShape<1>(ndim);
    if (!use_all_axes_) {
      args_shape[0] = std::max(static_cast<int>(axes_.size()),
                               static_cast<int>(axis_names_.size()));
    }

    if (ws.NumInput() != (spec_.GetSchema().MinNumInput() + 2))
      return false;

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
        ProcessPositionalInputArgs(data_idx,
                                    anchor_view.tensor_data(data_idx),
                                    shape_view.tensor_data(data_idx));
      }
    ), DALI_FAIL(make_string("Unsupported type of anchor and shape arguments: ", args_dtype)));  // NOLINT
    return true;
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

 private:
  template <typename AnchorT, typename ShapeT>
  void ProcessPositionalInputArgs(int data_idx,
                                  const AnchorT *slice_anchor_data,
                                  const ShapeT *slice_shape_data) {
    bool normalized_anchor = std::is_floating_point<AnchorT>::value && normalized_anchor_;
    bool normalized_shape  = std::is_floating_point<ShapeT>::value && normalized_shape_;
    crop_window_generators_[data_idx] =
      [this, slice_anchor_data, slice_shape_data,
       normalized_anchor, normalized_shape]
      (const TensorShape<> &shape, const TensorLayout& shape_layout) {
        CropWindow slice;
        slice.anchor = std::vector<int64_t>(shape.size(), 0);
        slice.shape = shape;

        auto axes = axes_;
        if (!axis_names_.empty()) {
          axes = GetDimIndices(shape_layout, axis_names_).to_vector();
        } else if (use_all_axes_) {
          axes.resize(shape.sample_dim());
          std::iota(axes.begin(), axes.end(), 0);
        }

        constexpr double i64min = static_cast<double>(std::numeric_limits<int64_t>::min());
        constexpr double i64max = static_cast<double>(std::numeric_limits<int64_t>::max());

        for (size_t i = 0; i < axes.size(); i++) {
          auto dim = axes[i];
          double anchor_val = slice_anchor_data[i];
          double shape_val = slice_shape_data ? slice_shape_data[i] : 0;
          double end_val = 0.0;
          // special case - minimize the floating point error by multiplying only once after sum
          if (normalized_anchor && normalized_shape) {
            end_val = (anchor_val + shape_val) * shape[dim];
            anchor_val *= shape[dim];
          } else {
            if (normalized_anchor) {
              anchor_val *= shape[dim];
            }
            if (normalized_shape) {
              shape_val *= shape[dim];
            }
            end_val = anchor_val + shape_val;
          }

          DALI_ENFORCE(anchor_val >= i64min && anchor_val <= i64max,
                       make_string("anchor value out of range [", i64min, ", ", i64max,
                                   "]. Got: ", anchor_val));
          DALI_ENFORCE(end_val >= i64min && end_val <= i64max,
                       make_string("end coordinates value out of range [", i64min, ", ", i64max,
                                   "]. Got: ", end_val));
          DALI_ENFORCE(anchor_val <= end_val,
                       make_string("end coordinates can't be before start coordinates. Got: start=",
                                   anchor_val, " end=", end_val));

          slice.anchor[dim] = std::llround(anchor_val);
          slice.shape[dim] = std::llround(end_val) - slice.anchor[dim];
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
  const OpSpec &spec_;
  bool normalized_anchor_, normalized_shape_;
  std::vector<int> axes_;
  TensorLayout axis_names_;
  std::vector<CropWindowGenerator> crop_window_generators_;
  bool use_all_axes_ = false;
};

class SliceAttr {
 public:
  explicit inline SliceAttr(const OpSpec &spec)
      : spec_(spec),
        named_slice_attr_(spec),
        pos_slice_attr_(spec) {
  }

  template <typename Backend>
  void ProcessArguments(const workspace_t<Backend> &ws) {
    use_named_args_ = named_slice_attr_.template ProcessArguments<Backend>(ws);
    if (use_named_args_) {
      if (spec_.HasArgument("normalized_anchor") || spec_.HasArgument("normalized_shape")) {
        DALI_WARN(
            "Warning: ``normalized_anchor``/``normalized_shape`` is only relevant "
            "when using positional slice arguments");
      }
      DALI_ENFORCE(ws.NumInput() == spec_.GetSchema().MinNumInput(),
                  "Named arguments start/end/shape are not compatible with positional"
                  " anchor and shape inputs");
    } else if (ws.NumInput() == (spec_.GetSchema().MinNumInput() + 2)) {
      bool processed_pos_args = pos_slice_attr_.template ProcessArguments<Backend>(ws);
      DALI_ENFORCE(processed_pos_args, "Failed to process positional arguments (start, shape)");
    } else {
      DALI_FAIL(
          make_string("Expected named slice arguments (e.g. start/end, start/shape) "
                      "or positional inputs (start, shape). Got ",
                      ws.NumInput(), " inputs."));
    }
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    if (use_named_args_)
      return named_slice_attr_.GetCropWindowGenerator(data_idx);
    else
      return pos_slice_attr_.GetCropWindowGenerator(data_idx);
  }

 private:
  const OpSpec &spec_;
  NamedSliceAttr named_slice_attr_;
  PositionalSliceAttr pos_slice_attr_;
  bool use_named_args_ = false;
};


}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_ATTR_H_
