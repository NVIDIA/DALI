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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cuda_impl.cuh"
namespace dali {
namespace kernels {

template <typename OutputType, typename InputType, int Dims>
class SliceFlipNormalizePermutePadGpu {
 private:
  static constexpr size_t kBlockDim = 512;
  static constexpr size_t kBlockSize = 64 * kBlockDim;
  size_t block_count_ = 0;

 public:
  using Args = SliceFlipNormalizePermutePadArgs<Dims>;
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<Args> &args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    const size_t num_samples = in.size();
    se.add<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    se.add<detail::SampleDesc<Dims>>(AllocType::GPU, num_samples);

    DALI_ENFORCE(args[0].mean.size() == args[0].inv_stddev.size());
    size_t norm_args_size = args[0].mean.size();
    if (norm_args_size > 0) {
      se.add<float>(AllocType::Host, 2 * norm_args_size);
      se.add<float>(AllocType::GPU,  2 * norm_args_size);
    }

    block_count_ = 0;
    for (auto &elem : args) {
      size_t sample_size = volume(elem.padded_shape);
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<detail::BlockDesc>(AllocType::Host, block_count_);
    se.add<detail::BlockDesc>(AllocType::GPU, block_count_);
    req.scratch_sizes = se.sizes;

    auto in_shapes = in.shape;
    TensorListShape<Dims> output_shapes(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      TensorShape<Dims> out_shape(args[i].padded_shape);
      CheckValidOutputShape(in_shapes[i], out_shape, args[i]);
      out_shape = detail::permute(out_shape, args[i].permuted_dims);
      output_shapes.set_tensor_shape(i, out_shape);
    }
    req.output_shapes = { output_shapes };
    return req;
  }

  void Run(KernelContext &context,
           OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<Args> &args) {
    const auto num_samples = in.size();

    auto mean_data = args[0].mean;
    auto inv_stddev_data = args[0].inv_stddev;
    DALI_ENFORCE(mean_data.size() == inv_stddev_data.size());

    detail::SampleDesc<Dims>* sample_descs_cpu =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(AllocType::Host,
                                                             num_samples);
    float *norm_add_cpu = mean_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(AllocType::Host, mean_data.size());
    float *norm_mul_cpu = inv_stddev_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(AllocType::Host, inv_stddev_data.size());
    detail::BlockDesc *block_descs_cpu =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    for (size_t i = 0; i < mean_data.size(); i++) {
      norm_add_cpu[i] = -mean_data[i] * inv_stddev_data[i];
      norm_mul_cpu[i] = inv_stddev_data[i];
    }
    int normalization_dim = Dims + 1;
    std::vector<size_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      auto processed_args = detail::ProcessArgs(args[i], in_shape);
      if (i == 0) {
        normalization_dim = processed_args.normalization_dim;
      } else {
        DALI_ENFORCE(normalization_dim == processed_args.normalization_dim);
      }
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = processed_args.in_strides;
      sample_desc.out_strides = processed_args.out_strides;
      sample_desc.out_shape = processed_args.out_shape;
      sample_desc.padded_out_shape = processed_args.padded_out_shape;
      sample_desc.padding_val = processed_args.padding_val;
      sample_desc.in = in.tensor_data(i) + processed_args.input_offset;
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(processed_args.padded_out_shape);
    }

    size_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      size_t offset = 0;
      size_t remaining = sample_sizes[i];
      while (remaining > 0) {
        size_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    detail::SampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(
        AllocType::GPU, num_samples);

    float *norm_add = mean_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(
        AllocType::GPU, mean_data.size());

    float *norm_mul = inv_stddev_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(
        AllocType::GPU, inv_stddev_data.size());

    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(
        AllocType::GPU, block_count_);

    // Memory is allocated contiguously, so we launch only one cudaMemcpyAsync
    size_t total_bytes = num_samples * sizeof(detail::SampleDesc<Dims>)
      + mean_data.size() * sizeof(float)
      + inv_stddev_data.size() * sizeof(float)
      + block_count_ * sizeof(detail::BlockDesc);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu,
                    total_bytes,
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const auto grid = block_count_;
    if (norm_add != nullptr && norm_mul != nullptr) {
      detail::SliceFlipNormalizePermutePadKernel<OutputType, InputType, Dims, true>
          <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs, norm_add,
                                                       norm_mul, normalization_dim);
    } else {
      detail::SliceFlipNormalizePermutePadKernel<OutputType, InputType, Dims, false>
          <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs, norm_add,
                                                       norm_mul, normalization_dim);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_
