// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_H_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_H_

#include <memory>
#include <functional>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/common/join/tensor_join_shape.h"

namespace dali {
namespace kernels {
namespace tensor_join {

template <typename T, bool new_axis>
class DLL_PUBLIC TensorJoinImplGPU {
 public:
  static_assert(std::is_same<T, type_of_size<sizeof(T)>>::value,
                "This class must be used with a type proudced by `type_of_size<size>`");
  static_assert(sizeof(T) >= 1 && sizeof(T) <= 16 && (sizeof(T)&(sizeof(T)-1)) == 0,
                "TensorJoin works only for types of size 1, 2, 4, 8 and 16 bytes.");

  /**
   * @param output_shape    shape of the result of joining the inputs
   * @param se              scratchpad requirements are added here
   * @param get_input_shape a function called with an input index; returns a reference to a shape
   *                        of the input at given index
   * @param num_inputs      number of joined tensors
   * @param axis            The axis along which the tensors are concatenated or stacked;
   *                        when `new_axis` is true, a new axis with the length equal to the number
   *                        of inputs is inserted at this position.
   *                        Valid range:
   *                          * 0 to sample_dim when `new_axis` is true
   *                          * 0 to sample_dim - 1 when `new_axis` is false
   *                        where sample_dim is the dimensionality of the inputs.
   *
   * @remarks Inputs must have the same dimensionality.
   * Respective tensors in the input must have the same shape (if new_axis == `true`) or can
   * differ at index `axis` (if new_axis == `false`).
   */
  void Setup(TensorListShape<> &output_shape,
             ScratchpadEstimator &se,
             const std::function<const TensorListShape<> *(int)> &get_input_shape,
             int num_inputs,
             int axis);

  void Run(KernelContext &ctx, const OutListGPU<T> &out, span<const InListGPU<T> *const> in_lists);

 private:
  int axis_  = -1;
};

}  // namespace tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_H_
