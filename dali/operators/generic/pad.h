// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <any>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"
#include "dali/operators/util/axis_args.h"
#include "dali/pipeline/operator/operator.h"

#define PAD_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                             uint64_t, int64_t, float, float16, bool)
#define PAD_SUPPORTED_NDIMS (1, 2, 3, 4, 5)

namespace dali {

template <typename Backend>
class Pad : public Operator<Backend> {
 public:
  inline explicit Pad(const OpSpec &spec)
      : Operator<Backend>(spec)
      , axis_args_(spec, "axes", "axis_names")
      , shape_("shape", spec)
      , align_("align", spec)
      , fill_value_("fill_value", spec) {
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  using Operator<Backend>::RunImpl;
  void RunImpl(Workspace &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

 private:
  void ReadArguments(const OpSpec &spec, const Workspace &ws) {
    const auto &input = ws.Input<Backend>(0);
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto in_shape = input.shape();
    int ndim = in_shape.sample_dim();
    int nsamples = in_shape.num_samples();

    assert(fill_value_);
    fill_value_.Acquire(spec, ws, curr_batch_size);

    axis_args_.Acquire(spec, ws, curr_batch_size, ndim);
    auto axes_sh = axis_args_.AxesShape();

    if (shape_)
      shape_.Acquire(spec, ws, curr_batch_size, axes_sh, ArgValue_AllowEmpty);

    if (align_) {
      align_.Acquire(spec, ws, curr_batch_size, axes_sh, ArgValue_AllowEmpty);
      for (int i = 0; i < nsamples; i++) {
        const auto &a = align_[i];
        for (int k = 0; k < a.num_elements(); k++)
          DALI_ENFORCE(a.data[k] > 0, "Values of `align` argument must be positive.");
      }
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

    TensorShape<> largest_shape = std::vector<int64_t>(ndim, -1);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      auto data_shape = in_shape.tensor_shape_span(sample_idx);
      for (int d = 0; d < ndim; d++) {
        largest_shape[d] = std::max(largest_shape[d], data_shape[d]);
      }
    }

    if (!kernel_sample_args_.has_value()) {
      kernel_sample_args_ = std::vector<Args>();
    }

    auto &kernel_sample_args = std::any_cast<std::vector<Args>&>(kernel_sample_args_);
    kernel_sample_args.clear();
    kernel_sample_args.resize(nsamples);

    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      auto axes = axis_args_.Get(sample_idx, ndim, in_layout);

      bool has_req_shape = shape_ && !shape_.IsEmpty(sample_idx);
      bool has_align = align_ && !align_.IsEmpty(sample_idx);

      auto &sample_args = kernel_sample_args[sample_idx];
      const auto &sample_shape = in_shape.tensor_shape_span(sample_idx);
      for (int d = 0; d < sample_args.anchor.size(); d++) {
        sample_args.anchor[d] = 0;
        sample_args.shape[d] = sample_shape[d];
      }
      sample_args.fill_values.resize(1);
      sample_args.fill_values[0] = fill_value_[sample_idx].data[0];

      int naxes = axes.size();
      for (int i = 0; i < naxes; i++) {
        int64_t req_extent = has_req_shape ? shape_[sample_idx].data[i] : -1;
        auto req_align = has_align ? align_[sample_idx].data[i] : 1;
        auto axis = axes[i];
        auto &extent = sample_args.shape[axis];
        // Adjust padded extent only if it is bigger than the sample's extent
        // That is, we are not cropping the image
        if (req_extent > 0) {  // explicitly requested padded shape
          extent = std::max(req_extent, extent);
        } else {  // pad to largest
          extent = std::max(largest_shape[axis], extent);
        }
        // Adjust alignment if required
        if (req_align > 1) {
          extent = aligned_extent(extent, align_[sample_idx].data[i]);
        }
      }
    }

    return kernel_sample_args;
  }

  AxisArgs axis_args_;
  ArgValue<int, 1> shape_;
  ArgValue<int, 1> align_;
  ArgValue<float> fill_value_;

  kernels::KernelManager kmgr_;
  std::any kernel_sample_args_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_PAD_H_
