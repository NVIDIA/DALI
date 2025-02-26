// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
void TensorJoinImplGPU<T, new_axis>::Setup(
        TensorListShape<> &output_shape,
        const std::function<const TensorListShape<> *(int)> &get_input_shape,
        int num_inputs,
        int axis) {
  JoinedShape(output_shape, get_input_shape, num_inputs, axis, new_axis);
  int N = output_shape.num_samples();
  axis_ = axis;
}

template <typename T, bool new_axis>
void TensorJoinImplGPU<T, new_axis>::Run(
        KernelContext &ctx,
        const OutListGPU<T> &out,
        span<const InListGPU<T> *const> in_lists) {
  int njoin = in_lists.size();
  int N = out.num_samples();
  int N_in = N * njoin;

  auto output_descs_cpu = make_span(ctx.scratchpad->AllocatePinned<OutputDesc<T>>(N), N);
  auto input_descs_cpu  = make_span(ctx.scratchpad->AllocatePinned<InputDesc<T>>(N_in), N_in);

  FillDescs(output_descs_cpu, input_descs_cpu, out, in_lists, axis_);

  OutputDesc<T> *out_descs_gpu = nullptr;
  InputDesc<T> *in_descs_gpu = nullptr;

  std::tie(out_descs_gpu, in_descs_gpu) = ctx.scratchpad->ToContiguousGPU(
    ctx.gpu.stream, output_descs_cpu, input_descs_cpu);

  int64_t avg_size = out.num_elements() / N;
  dim3 grid(std::max(static_cast<int>(avg_size / 2048), 32), N);
  dim3 block(256);  // tuned!

  JoinTensorsKernel<<<grid, block, 0, ctx.gpu.stream>>>(
    out_descs_gpu, in_descs_gpu, njoin);
}

template class TensorJoinImplGPU<type_of_size<1>, false>;
template class TensorJoinImplGPU<type_of_size<1>, true>;
template class TensorJoinImplGPU<type_of_size<2>, false>;
template class TensorJoinImplGPU<type_of_size<2>, true>;
template class TensorJoinImplGPU<type_of_size<4>, false>;
template class TensorJoinImplGPU<type_of_size<4>, true>;
template class TensorJoinImplGPU<type_of_size<8>, false>;
template class TensorJoinImplGPU<type_of_size<8>, true>;
template class TensorJoinImplGPU<type_of_size<16>, false>;
template class TensorJoinImplGPU<type_of_size<16>, true>;

}  // namespace tensor_join
}  // namespace kernels
}  // namespace dali
