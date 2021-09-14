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

#ifndef DALI_KERNELS_REDUCE_REDUCE_ALL_KERNEL_GPU_H_
#define DALI_KERNELS_REDUCE_REDUCE_ALL_KERNEL_GPU_H_

#include <memory>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/kernels/reduce/reduce_all_gpu_impl.cuh"

namespace dali {
namespace kernels {
namespace reduce {

template <typename Out, typename In, typename Reduction, typename Preprocessor = identity>
class DLL_PUBLIC ReduceAllGPU {
 public:
  // TODO(janton): remove this when implemented
  static_assert(std::is_same<Preprocessor, identity>::value,
                "Preprocessing is not yet implemented");
  using Acc = Out;

  DLL_PUBLIC ~ReduceAllGPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<In, DynamicDimensions> &in) {
    auto num_samples = in.size();
    ScratchpadEstimator se;

    int64_t max_sample_size = 0;
    for (int i = 0; i < num_samples; i++) {
      max_sample_size = std::max(max_sample_size, volume(in.tensor_shape(i)));
    }
    if (max_sample_size <= 4096) {
      blocks_per_sample_ = 1;
    } else {
      blocks_per_sample_ = static_cast<int>(std::sqrt(max_sample_size));
    }

    se.add<mm::memory_kind::host, const In*>(num_samples);
    se.add<mm::memory_kind::host, int64_t>(num_samples);
    se.add<mm::memory_kind::device, const In*>(num_samples);
    se.add<mm::memory_kind::device, int64_t>(num_samples);
    tmp_buffer_size_ = num_samples * blocks_per_sample_;
    se.add<mm::memory_kind::device, Out>(tmp_buffer_size_);

    KernelRequirements req;
    req.scratch_sizes = se.sizes;
    req.output_shapes = {TensorListShape<0>(num_samples)};
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutListGPU<Out, 0> &out,
                      const InListGPU<In, DynamicDimensions> &in) {
    DALI_ENFORCE(out.is_contiguous(), "Reduce all kernel expects the output to be contiguous");
    auto* out_start = out[0].data;

    auto num_samples = in.size();
    auto* sample_data = context.scratchpad->AllocateHost<const In*>(num_samples);
    auto* sample_size = context.scratchpad->AllocateHost<int64_t>(num_samples);
    for (int i = 0; i < num_samples; i++) {
      sample_data[i] = in.tensor_data(i);
      sample_size[i] = volume(in.tensor_shape(i));
    }

    auto* sample_data_gpu = context.scratchpad->AllocateGPU<const In*>(num_samples);
    auto* sample_size_gpu = context.scratchpad->AllocateGPU<int64_t>(num_samples);
    // Single memcpy, since the data in the scratchpad is contiguous
    CUDA_CALL(
      cudaMemcpyAsync(sample_data_gpu, sample_data,
                      num_samples * sizeof(const In*) + num_samples * sizeof(int64_t),
                      cudaMemcpyHostToDevice, context.gpu.stream));

    auto* buffer_gpu = context.scratchpad->AllocateGPU<Out>(tmp_buffer_size_);

    // The reduction is divided into two stages. To minimize the precision error due to
    // accumulating numbers sequentially, we use `blocks_per_sample_ = sqrt(max_sample_size)`
    // so that the first kernel reduces sample_size numbers to sqrt(max_sample_size) numbers
    // and the second one reduces sqrt(max_sample_size) numbers to a single number.

    dim3 block(32, 32);
    dim3 grid(blocks_per_sample_, num_samples);

    if (blocks_per_sample_ == 1) {
      // For small inputs, we reduce in one step
      ReduceAllBatchedKernel<Acc><<<grid, block, 0, context.gpu.stream>>>(
          out_start, sample_data_gpu, sample_size_gpu, reduction);
    } else {
      ReduceAllBatchedKernel<Acc><<<grid, block, 0, context.gpu.stream>>>(
          buffer_gpu, sample_data_gpu, sample_size_gpu, reduction);

      dim3 grid2(1, num_samples);
      ReduceAllBlockwiseKernel<Acc><<<grid2, block, 0, context.gpu.stream>>>(
          out_start, buffer_gpu, blocks_per_sample_, reduction);
    }
  }

 private:
  int blocks_per_sample_;
  int tmp_buffer_size_;
  Reduction reduction;
};

}  // namespace reduce
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_ALL_KERNEL_GPU_H_
