// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_H_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_H_

#include "dali/kernels/kernel.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.h"

namespace dali {
namespace kernels {

/**
 * @brief Joins (concatenates or stacks) batches of tensors
 *
 * This kernel takes a number of batches of tensors as the input and produces a single
 * batch of tensors, where each tensor is a result of joining (concatenating or stacking)
 * the respective tensors in the inputs.
 *
 * @tparam T        type of the tensor element. sizeof(T) must be 1,2,4,8 or 16
 * @tparam new_axis if true, stacking is performed, otherwise - concatenation
 *
 */
template <typename T, bool new_axis>
class TensorJoinGPU : public tensor_join::TensorJoinImplGPU<type_of_size<sizeof(T)>, new_axis> {
  using U = type_of_size<sizeof(T)>;
  using InListU = InListGPU<U>;
 public:
  using Base = tensor_join::TensorJoinImplGPU<U, new_axis>;

  /**
   * @param ctx             kernel context, not used
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
  KernelRequirements Setup(KernelContext &ctx,
                           const std::function<const TensorListShape<> *(int)> &get_input_shape,
                           int num_inputs,
                           int axis) {
    ScratchpadEstimator se;
    KernelRequirements req;
    req.output_shapes.resize(1);
    se.add<mm::memory_kind::host, const InListU *>(num_inputs);
    Base::Setup(req.output_shapes[0], se, get_input_shape, num_inputs, axis);
    req.scratch_sizes = se.sizes;
    return req;
  }

  /**
   * @param ctx             kernel context, not used
   * @param inputs          views to the inputs
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
  KernelRequirements Setup(KernelContext &ctx,
                           span<const InListGPU<T>> inputs,
                           int axis) {
    return Setup(ctx, [&](int idx){ return &inputs[idx].shape; }, inputs.size(), axis);
  }

  /**
   * @brief Concatenates or stacks the tensors in `in_lists` and stores the result in `out`.
   *
   * @param ctx       Kernel context (CUDA stream, scratchpad)
   * @param out       Output tensor list
   * @param in_lists  List of pointers to the inputs
   */
  template <int in_ndim>
  void Run(KernelContext &ctx, const OutListGPU<T> &out,
           span<const InListGPU<T, in_ndim> *const> in_lists) {
    auto *lists = reinterpret_cast<const InListGPU<U, in_ndim> *const*>(in_lists.data());
    Base::Run(ctx,
        reinterpret_cast<const OutListGPU<U> &>(out),
        make_span(lists, in_lists.size()));
  }

  /**
   * @brief Concatenates or stacks the tensors in `in_lists` and stores the result in `out`.
   *
   * @param ctx       Kernel context (CUDA stream, scratchpad)
   * @param out       Output tensor list
   * @param in_lists  List of pointers to the inputs
   */
  template <int in_ndim>
  void Run(KernelContext &ctx, const OutListGPU<T> &out,
           span<const InListGPU<T, in_ndim>> in_lists) {
    int njoin = in_lists.size();
    auto *in_list_ptrs = ctx.scratchpad->AllocateHost<const InListU *>(njoin);
    for (int i = 0; i < njoin; i++)
      in_list_ptrs[i] = reinterpret_cast<const InListU *>(&in_lists[i]);
    Base::Run(ctx, reinterpret_cast<const OutListGPU<U> &>(out), make_span(in_list_ptrs, njoin));
  }
};

template <typename T>
using TensorStackGPU = TensorJoinGPU<T, true>;

template <typename T>
using TensorConcatGPU = TensorJoinGPU<T, false>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_H_
