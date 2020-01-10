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

#ifndef DALI_OPERATORS_UTIL_PAD_H_
#define DALI_OPERATORS_UTIL_PAD_H_

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
    , fill_value_(spec.GetArgument<float>("fill_value"))
    , axes_(spec.GetRepeatedArgument<int>("axes")) {}

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  using Operator<Backend>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

 private:
  template <typename Args>
  std::vector<Args>& FillArgs(TensorListShape<> in_shape) {
    int nsamples = in_shape.num_samples();
    if (!kernel_sample_args_.has_value()) {
      kernel_sample_args_ = std::vector<Args>();
      auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
      kernel_sample_args.reserve(nsamples);
      for (int i = 0; i < nsamples; i++) {
        kernel_sample_args.emplace_back(in_shape[i]);
      }
    }

    auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
    assert(static_cast<int>(kernel_sample_args.size()) == nsamples);

    int ndim = in_shape.sample_dim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    }
    TensorShape<> padded_shape;
    padded_shape.resize(ndim);

    assert(static_cast<int>(axes_.size()) <= ndim);
    for (int i = 0; i < nsamples; i++) {
      auto shape = in_shape[i];
      for (auto axis : axes_) {
        if (shape[axis] > padded_shape[axis])
          padded_shape[axis] = shape[axis];
      }
    }

    for (int i = 0; i < nsamples; i++) {
      auto &sample_args = kernel_sample_args[i];
      sample_args.padding_val = fill_value_;
      for (auto axis : axes_) {
        sample_args.padded_shape[axis] = padded_shape[axis];
        assert(sample_args.padded_shape[axis] > 0);
      }
    }

    return kernel_sample_args;
  }

  float fill_value_;
  std::vector<int> axes_;
  kernels::KernelManager kmgr_;
  any kernel_sample_args_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_PAD_H_
