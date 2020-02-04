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

    if (spec.HasArgument("align")) {
      align_ = spec.GetRepeatedArgument<int>("align");
    }

    if (spec.HasArgument("shape")) {
      auto shape_arg = spec.GetRepeatedArgument<int>("shape");
      shape_ = std::vector<int64_t>{std::begin(shape_arg), std::end(shape_arg)};
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
  template <typename Args>
  std::vector<Args>& FillArgs(TensorListShape<> in_shape, TensorLayout in_layout) {
    int nsamples = in_shape.num_samples();
    if (!kernel_sample_args_.has_value()) {
      kernel_sample_args_ = std::vector<Args>();
    }

    auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
    kernel_sample_args.clear();
    kernel_sample_args.reserve(nsamples);
    for (int i = 0; i < nsamples; i++) {
      kernel_sample_args.emplace_back(in_shape[i]);
    }

    int ndim = in_shape.sample_dim();

    for (auto axis : axes_) {
      DALI_ENFORCE(axis >= 0 && axis < ndim,
        make_string("specified axis is out of bounds. axis=", axis, ", ndim=", ndim));
    }

    if (!axis_names_.empty()) {
      axes_ = GetDimIndices(in_layout, axis_names_).to_vector();
    }

    // If no axes are provided, use all
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    }

    // If a single *align* value is provided, use this value for all the axes
    if (align_.size() == 1 && axes_.size() > 1) {
      align_.resize(axes_.size(), align_[0]);
    }

    TensorShape<> padded_shape;
    padded_shape.resize(ndim);

    // If no shape arg is provided, mark all axis as -1 (automatic shape)
    if (shape_.empty()) {
      shape_ = std::vector<int64_t>(axes_.size(), -1);
    }

    DALI_ENFORCE(static_cast<int>(axes_.size()) == shape_.size(),
      make_string("If explicit shape is provided, there should be a value per axis to be padded. "
                  "Expected ", axes_.size(), " values for shape, got ", shape_.size()));

    assert(static_cast<int>(axes_.size()) <= ndim);
    for (int i = 0; i < static_cast<int>(axes_.size()); i++) {
      auto axis = axes_[i];
      int64_t shape_val = shape_[i];
      if (shape_val > 0) {
        padded_shape[axis] = shape_val;
      } else {
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
          auto data_shape = in_shape[sample_idx];
          if (data_shape[axis] > padded_shape[axis])
            padded_shape[axis] = data_shape[axis];
        }
        assert(padded_shape[axis] > 0);
        if (!align_.empty()) {
          int64_t align_val = align_[i];
          int64_t remainder = padded_shape[axis] % align_val;
          if (remainder > 0) {
            padded_shape[axis] += align_val - remainder;
          }
        }
      }
    }

    for (int i = 0; i < nsamples; i++) {
      auto &sample_args = kernel_sample_args[i];
      sample_args.padding_val = fill_value_;
      for (auto axis : axes_) {
        assert(padded_shape[axis] > 0);
        sample_args.padded_shape[axis] = padded_shape[axis];
      }
    }

    return kernel_sample_args;
  }

  float fill_value_;
  TensorLayout axis_names_;
  std::vector<int> axes_;
  std::vector<int> align_;
  TensorShape<> shape_;
  kernels::KernelManager kmgr_;
  any kernel_sample_args_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_PAD_H_
