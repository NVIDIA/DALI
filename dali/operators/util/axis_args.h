// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_UTIL_AXIS_ARGS_H_
#define DALI_OPERATORS_UTIL_AXIS_ARGS_H_

#include <memory>
#include <vector>

#include "dali/core/small_vector.h"
#include "dali/core/tensor_layout.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

class AxisArgs {
 public:
  enum Flags : unsigned {
    AllowEmpty = 0b00000001,
    AllowMultiple = 0b00000010,
    AllIfEmpty = 0b00000100,
    AllowNegative = 0b000001000,
    AllowNonUniformLen = 0b00010000,
    DefaultFlags = AllowEmpty | AllowMultiple | AllIfEmpty | AllowNegative | AllowNonUniformLen
  };

  AxisArgs(const OpSpec &spec, const char *axes_arg, const char *axis_names_arg,
           unsigned int flags = DefaultFlags);

  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples);
  TensorListShape<1> AxesShape();

  SmallVector<int, 6> Get(int data_idx, int ndim, const TensorLayout &layout);

 private:
  void Process(int ndim, SmallVector<int, 6> &axes);

  unsigned int flags_ = 0;
  std::unique_ptr<ArgValue<int, 1>> axes_;

  bool use_axis_names_ = false;
  bool per_sample_axes_ = false;
  TensorLayout axis_names_;
  SmallVector<int, 6> const_axes_;
  TensorListShape<1> shape_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_AXIS_ARGS_H_
