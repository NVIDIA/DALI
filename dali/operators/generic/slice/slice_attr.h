// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/operators/util/axis_args.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/util/crop_window.h"

#define SLICE_ARGS_TYPES (int32_t, int64_t, float)

namespace dali {

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
      : axis_args_(spec, axes_arg_name, axis_names_arg_name),
        start_(start_name, spec),
        rel_start_(rel_start_name, spec),
        end_(end_name, spec),
        rel_end_(rel_end_name, spec),
        shape_(shape_name, spec),
        rel_shape_(rel_shape_name, spec) {
    int max_batch_sz = spec.GetArgument<int>("max_batch_size");
    crop_window_generators_.resize(max_batch_sz);

    has_start_ = start_.HasExplicitValue() || rel_start_.HasExplicitValue();

    if ((start_.HasExplicitValue() + rel_start_.HasExplicitValue()) > 1)
      DALI_FAIL(make_string("\"", start_name, "\" and \"", rel_start_name,
                            "\" arguments are mutually exclusive"));

    has_end_ = end_.HasExplicitValue() || rel_end_.HasExplicitValue();
    has_shape_ = shape_.HasExplicitValue() || rel_shape_.HasExplicitValue();

    if ((end_.HasExplicitValue() + rel_end_.HasExplicitValue() + shape_.HasExplicitValue() +
         rel_shape_.HasExplicitValue()) > 1)
      DALI_FAIL(make_string("\"", end_name, "\", \"", rel_end_name, "\", \"", shape_name,
                            "\", and \"", rel_shape_name, "\" arguments are mutually exclusive"));
  }

  template <typename Backend>
  bool ProcessArguments(const OpSpec& spec, const Workspace &ws,
                        int curr_batch_size = -1, int ndim = -1) {
    if (curr_batch_size < 0)
      curr_batch_size = ws.GetInputBatchSize(0);
    if (ndim < 0)
      ndim = ws.GetInputDim(0);

    axis_args_.Acquire(spec, ws, curr_batch_size, ndim);
    auto args_sh = axis_args_.AxesShape();

    ArgValueFlags flags = ArgValue_AllowEmpty;
    if (start_.HasExplicitValue())
      start_.Acquire(spec, ws, curr_batch_size, args_sh, flags);
    else if (rel_start_.HasExplicitValue())
      rel_start_.Acquire(spec, ws, curr_batch_size, args_sh, flags);

    if (end_.HasExplicitValue())
      end_.Acquire(spec, ws, curr_batch_size, args_sh, flags);
    else if (rel_end_.HasExplicitValue())
      rel_end_.Acquire(spec, ws, curr_batch_size, args_sh, flags);
    else if (shape_.HasExplicitValue())
      shape_.Acquire(spec, ws, curr_batch_size, args_sh, flags);
    else if (rel_shape_.HasExplicitValue())
      rel_shape_.Acquire(spec, ws, curr_batch_size, args_sh, flags);

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

        int ndim = shape.sample_dim();
        auto axes = axis_args_.Get(data_idx, ndim, shape_layout);

        constexpr double i64min = static_cast<double>(std::numeric_limits<int64_t>::min());
        constexpr double i64max = static_cast<double>(std::numeric_limits<int64_t>::max());

        for (size_t i = 0; i < axes.size(); i++) {
          auto dim = axes[i];

          double anchor_val = 0;
          if (start_ && !start_.IsEmpty(data_idx)) {
            anchor_val = start_[data_idx].data[i];
          } else if (rel_start_ && !rel_start_.IsEmpty(data_idx)) {
            anchor_val = rel_start_[data_idx].data[i] * shape[dim];
          }
          DALI_ENFORCE(anchor_val >= i64min && anchor_val <= i64max,
                       make_string("anchor value out of range [", i64min, ", ", i64max,
                                   "]. Got: ", anchor_val));

          double end_val = shape[dim];
          if (end_ && !end_.IsEmpty(data_idx)) {
            end_val = end_[data_idx].data[i];
          } else if (rel_end_ && !rel_end_.IsEmpty(data_idx)) {
            end_val = rel_end_[data_idx].data[i] * shape[dim];
          } else if (shape_ && !shape_.IsEmpty(data_idx)) {
            double shape_val = shape_[data_idx].data[i];
            DALI_ENFORCE(shape_val >= 0 && shape_val <= i64max,
              make_string("shape value out of range [", 0, ", ", i64max, "]. Got: ", shape_val));

            end_val = anchor_val + shape_val;
          } else if (rel_start_ && rel_shape_ && !rel_start_.IsEmpty(data_idx) &&
                     !rel_shape_.IsEmpty(data_idx)) {
            // special case - minimize the floating point error by multiplying only once after sum
            double rel_start_val = rel_start_[data_idx].data[i];
            double rel_shape_val = rel_shape_[data_idx].data[i];
            DALI_ENFORCE(rel_shape_val >= 0,
              make_string("negative shapes are not allowed. Got: ", rel_shape_val));

            end_val = (rel_start_val + rel_shape_val) * shape[dim];
          } else if (rel_shape_ && !rel_shape_.IsEmpty(data_idx)) {
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
  AxisArgs axis_args_;

  ArgValue<int, 1> start_;
  ArgValue<float, 1> rel_start_;

  ArgValue<int, 1> end_;
  ArgValue<float, 1> rel_end_;

  ArgValue<int, 1> shape_;
  ArgValue<float, 1> rel_shape_;

  std::vector<CropWindowGenerator> crop_window_generators_;

  bool has_start_, has_end_, has_shape_;
};


class PositionalSliceAttr {
 public:
  explicit inline PositionalSliceAttr(const OpSpec& spec)
      : axis_args_(spec, "axes", "axis_names") {
    normalized_anchor_ = spec.GetArgument<bool>("normalized_anchor");
    normalized_shape_ = spec.GetArgument<bool>("normalized_shape");
    int max_batch_sz = spec.GetArgument<int>("max_batch_size");
    crop_window_generators_.resize(max_batch_sz);

    crop_anchor_cpu_.set_pinned(true);
    crop_shape_cpu_.set_pinned(true);
  }

  template <typename Backend>
  bool ProcessArguments(const OpSpec &spec, const Workspace &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    int ndim = ws.GetInputDim(0);

    if (ws.NumInput() != (spec.GetSchema().MinNumInput() + 2))
      return false;

    axis_args_.Acquire(spec, ws, curr_batch_size, ndim);

    auto crop_anchor_type = ws.GetInputDataType(1);
    auto crop_shape_type = ws.GetInputDataType(2);
    DALI_ENFORCE(crop_anchor_type == crop_shape_type,
                 make_string("Anchor and shape should have the same type. Got: ",
                             crop_anchor_type, " and ", crop_shape_type));
    TYPE_SWITCH(crop_anchor_type, type2id, ArgsType, SLICE_ARGS_TYPES, (
      TensorListView<StorageCPU, const ArgsType> anchor, shape;
      GetPositionalSliceArgsCPU<const ArgsType, Backend>(anchor, shape, ws);
      for (int data_idx = 0; data_idx < curr_batch_size; data_idx++) {
        ProcessPositionalInputArgs(data_idx,
                                   sample_as_span(anchor, data_idx),
                                   sample_as_span(shape, data_idx));
      }
    ), DALI_FAIL(make_string("Unsupported type of anchor and shape arguments: ", crop_anchor_type)));  // NOLINT
    return true;
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

 private:
  template <typename T>
  span<T> sample_as_span(const TensorListView<StorageCPU, T>& v,
                         int sample_idx) {
    ptrdiff_t vol = volume(v.tensor_shape_span(sample_idx));
    return span<T>(v.tensor_data(sample_idx), vol);
  }

  template <typename T, typename Backend>
  void GetPositionalSliceArgsCPU(TensorListView<StorageCPU, T> &anchor,
                                 TensorListView<StorageCPU, T> &shape,
                                 const Workspace &ws) {
    AccessOrder order;
    auto in_cpu_view = [&](int idx, TensorList<CPUBackend>& cpu_buffer) {
      if (ws.InputIsType<CPUBackend>(idx)) {
        return view<T>(ws.Input<CPUBackend>(idx));
      } else {
        const auto& arg = ws.Input<GPUBackend>(idx);
        if (!order)
          order = AccessOrder(ws.stream());
        cpu_buffer.set_order(order);
        cpu_buffer.Copy(arg);
        return view<T>(cpu_buffer);
      }
    };

    anchor = in_cpu_view(1, crop_anchor_cpu_);
    shape = in_cpu_view(2, crop_shape_cpu_);

    // Sync with stream used for copy, only if needed
    if (order)
      AccessOrder::host().wait(order);
  }

  template <typename AnchorT, typename ShapeT>
  void ProcessPositionalInputArgs(int data_idx,
                                  span<const AnchorT> slice_anchor_data,
                                  span<const ShapeT> slice_shape_data) {
    bool normalized_anchor = std::is_floating_point<AnchorT>::value && normalized_anchor_;
    bool normalized_shape  = std::is_floating_point<ShapeT>::value && normalized_shape_;
    DALI_ENFORCE(
        slice_anchor_data.size() == slice_shape_data.size(),
        make_string("Slice anchor and shape should have the same number of arguments. Got: ",
                    slice_anchor_data.size(), " and ", slice_shape_data.size()));

    crop_window_generators_[data_idx] =
      [this, slice_anchor_data, slice_shape_data,
       normalized_anchor, normalized_shape, data_idx]
      (const TensorShape<> &shape, const TensorLayout& shape_layout) {
        CropWindow slice;
        slice.anchor = std::vector<int64_t>(shape.size(), 0);
        slice.shape = shape;

        int ndim = shape.sample_dim();
        auto axes = axis_args_.Get(data_idx, ndim, shape_layout);

        // checking anchor is enough (we already checked they have the same size earlier)
        DALI_ENFORCE(static_cast<int>(axes.size()) == slice_anchor_data.size(),
                     make_string("Expected ", axes.size(),
                                 " elements for slice arguments (start/shape). Got ",
                                 slice_anchor_data.size()));

        constexpr double i64min = static_cast<double>(std::numeric_limits<int64_t>::min());
        constexpr double i64max = static_cast<double>(std::numeric_limits<int64_t>::max());

        for (size_t i = 0; i < axes.size(); i++) {
          auto dim = axes[i];

          double anchor_val = slice_anchor_data[i];
          double shape_val = slice_shape_data.data() ? slice_shape_data[i] : 0;
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

 private:
  bool normalized_anchor_, normalized_shape_;
  AxisArgs axis_args_;
  std::vector<CropWindowGenerator> crop_window_generators_;

  // To be used to copy GPU arguments
  TensorList<CPUBackend> crop_anchor_cpu_, crop_shape_cpu_;
};

class SliceAttr {
 public:
  explicit inline SliceAttr(const OpSpec &spec)
      : named_slice_attr_(spec),
        pos_slice_attr_(spec) {
  }

  template <typename Backend>
  void ProcessArguments(const OpSpec &spec, const Workspace &ws) {
    use_named_args_ = named_slice_attr_.template ProcessArguments<Backend>(spec, ws);
    if (use_named_args_) {
      if (spec.HasArgument("normalized_anchor") || spec.HasArgument("normalized_shape")) {
        DALI_WARN(
            "Warning: ``normalized_anchor``/``normalized_shape`` is only relevant "
            "when using positional slice arguments");
      }
      DALI_ENFORCE(ws.NumInput() == spec.GetSchema().MinNumInput(),
                  "Named arguments start/end/shape are not compatible with positional"
                  " anchor and shape inputs");
    } else if (ws.NumInput() == (spec.GetSchema().MinNumInput() + 2)) {
      bool processed_pos_args = pos_slice_attr_.template ProcessArguments<Backend>(spec, ws);
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
  NamedSliceAttr named_slice_attr_;
  PositionalSliceAttr pos_slice_attr_;
  bool use_named_args_ = false;
};


}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_ATTR_H_
