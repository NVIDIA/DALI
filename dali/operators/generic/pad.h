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

#ifndef DALI_OPERATORS_GENERIC_PAD_H_
#define DALI_OPERATORS_GENERIC_PAD_H_

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/any.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/operator/operator.h"

#define PAD_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                             uint64_t, int64_t, float, float16)
#define PAD_SUPPORTED_NDIMS (1, 2, 3, 4, 5)

namespace dali {

template <typename Backend>
class Pad : public Operator<Backend> {
 public:
  inline explicit Pad(const OpSpec &spec)
      : Operator<Backend>(spec)
      , fill_value_(spec.GetArgument<float>("fill_value")) {
    bool has_axis_names = spec.HasArgument("axis_names");
    bool has_axes = spec.HasArgument("axes");
    if (has_axis_names && has_axes) {
      DALI_FAIL("Arguments axis_names and axes are mutually exclusive");
    } else if (has_axis_names) {
      axis_names_ = spec.GetArgument<TensorLayout>("axis_names");
    } else if (has_axes) {
      axes_ = spec.GetRepeatedArgument<int>("axes");
    }
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

 private:
  void ReadArguments(const OpSpec &spec, const workspace_t<Backend> &ws) {
    const auto &input = ws.template InputRef<Backend>(0);
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto in_shape = input.shape();
    auto in_layout = input.GetLayout();
    int ndim = in_shape.sample_dim();
    int nsamples = in_shape.num_samples();

    if (!axis_names_.empty()) {
      axes_ = GetDimIndices(in_layout, axis_names_).to_vector();
    }

    for (auto axis : axes_) {
      DALI_ENFORCE(axis >= 0 && axis < ndim,
        make_string("specified axis is out of bounds. axis=", axis, ", ndim=", ndim));
    }

    // If no axes are provided, use all
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    }

    if (spec.ArgumentDefined("shape")) {
      GetShapeArgument(shape_, spec, "shape", ws, curr_batch_size);
    }
    if (spec.ArgumentDefined("align")) {
      GetShapeArgument(align_, spec, "align", ws, curr_batch_size);
    }

    if (shape_.empty())
      shape_ = uniform_list_shape(nsamples, TensorShape<>(std::vector<int64_t>(axes_.size(), -1)));
    // If a single *align* value is provided, use this value for all the axes
    for (int i = 0; i < nsamples; i++) {
      if (!align_.empty()) {
        for (auto &a : align_.tensor_shape_span(i)) {
          DALI_ENFORCE(a > 0, "Values of `align` argument must be positive.");
        }
      }
      auto shape = shape_.tensor_shape_span(i);
      DALI_ENFORCE(static_cast<int>(axes_.size()) == shape.size(),
          make_string(
              "If explicit shape is provided, there should be a value per axis to be padded. "
              "Expected ",
              axes_.size(), " values for shape, got ", shape.size()));
    }
  }

  static inline int64_t aligned_extent(int64_t extent, int alignment) {
    int64_t remainder = extent % alignment;
    if (remainder > 0)
      extent += alignment - remainder;
    return extent;
  }

  template <typename Args>
  std::vector<Args>& FillArgs(TensorListShape<> in_shape, TensorLayout in_layout) {
    int nsamples = in_shape.num_samples();
    int ndim = in_shape.sample_dim();
    int naxes = axes_.size();
    assert(naxes <= ndim);

    TensorShape<> largest_shape = std::vector<int64_t>(ndim, -1);
    for (int i = 0; i < naxes; i++) {
      auto axis = axes_[i];
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        auto data_shape = in_shape.tensor_shape_span(sample_idx);
        largest_shape[axis] = std::max(largest_shape[axis], data_shape[axis]);
      }
    }

    if (!kernel_sample_args_.has_value()) {
      kernel_sample_args_ = std::vector<Args>();
    }

    auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
    kernel_sample_args.clear();
    kernel_sample_args.resize(nsamples);

    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      auto &sample_args = kernel_sample_args[sample_idx];
      const auto &sample_shape = in_shape.tensor_shape_span(sample_idx);
      auto req_shape = shape_.tensor_shape_span(sample_idx);
      for (int d = 0; d < sample_args.anchor.size(); d++) {
        sample_args.anchor[d] = 0;
        sample_args.shape[d] = sample_shape[d];
      }
      sample_args.fill_values.resize(1);
      sample_args.fill_values[0] = fill_value_;

      for (int i = 0; i < naxes; i++) {
        auto req_extent = req_shape[i];
        auto axis = axes_[i];
        auto &extent = sample_args.shape[axis];
        // Adjust padded extent only if it is bigger than the sample's extent
        // That is, we are not cropping the image
        if (req_extent > 0) {  // explicitly requested padded shape
          extent = std::max(req_extent, extent);
        } else {  // pad to largest
          extent = std::max(largest_shape[axis], extent);
        }
        // Adjust alignment if required
        if (!align_.empty() && !align_.tensor_shape_span(sample_idx).empty()) {
          auto align = align_.tensor_shape_span(sample_idx);
          extent = aligned_extent(extent, align.size() == 1 ? align[0] : align[i]);
        }
      }
    }

    return kernel_sample_args;
  }

  float fill_value_;
  TensorLayout axis_names_;
  std::vector<int> axes_;
  TensorListShape<> align_;
  TensorListShape<> shape_;
  kernels::KernelManager kmgr_;
  any kernel_sample_args_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_PAD_H_
