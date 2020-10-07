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

#include "dali/kernels/common/join/tensor_join_gpu_impl.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.cuh"

namespace dali {
namespace kernels {
namespace tensor_join {


template <typename T, bool new_axis>
KernelRequirements TensorJoinImplGPU<T, new_axis>::Setup(
        KernelContext &ctx,
        const std::function<const TensorListShape<> &(int)> &get_input_shape,
        int num_inputs,
        int axis) {
  KernelRequirements req;
  req.output_shapes.resize(1);
  JoinedShape(req.output_shapes[0], get_input_shape, num_inputs, axis, new_axis);
  ScratchpadEstimator se;
  int N = req.output_shapes[0].num_samples();
  se.add<OutputDesc<T>>(AllocType::GPU, N);
  se.add<InputDesc<T>>(AllocType::GPU, num_inputs * N);
  req.scratch_sizes = se.sizes;
  return req;
}

template <typename T, bool new_axis>
void TensorJoinImplGPU<T, new_axis>::Run(
        KernelContext &ctx,
        OutListGPU<T> &out,
        span<const InListGPU<T> *const> &in_lists) {

}

}  // namespace tensor_join
}  // namespace kernels
}  // namespace dali
