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
    AllowEmpty = 0b0001,
    AllowMultiple = 0b0010,
    AllIfEmpty = 0b0100,
    AllowNegative = 0b1000,
    DefaultFlags = AllowEmpty | AllowMultiple | AllIfEmpty | AllowNegative
  };

  AxisArgs(const OpSpec &spec, const char *axes_arg, const char *axis_names_arg,
           unsigned int flags = DefaultFlags);

  /**
   * @brief Acquire axes information from the workspace
   *
   * @param spec
   * @param ws
   * @param nsamples
   * @param ndim Number of dimensions. If AllIfEmpty, it will be used to calculate the
   *             effective shape of the axes when the argument is empty
   *             (as many elements as ndim)
   */
  void Acquire(const OpSpec &spec, const ArgumentWorkspace &ws, int nsamples, int ndim);

  /**
   * @brief Returns the effective axes shape, given the input dimensionality.
   * @remarks When AllIfEmpty is selected, the effective axes shape will have as many
   *          elements as ndim provided to Acquire.
   *
   * @return TensorListShape<1>
   */
  TensorListShape<1> AxesShape();

  /**
   * @brief Retrieve the final axes for a particular sample,
   *        given its input dimensionalityand layout.
   *
   * @param data_idx
   * @param ndim
   * @param layout
   * @return SmallVector<int, 6>
   */
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
