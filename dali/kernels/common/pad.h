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

#ifndef DALI_KERNELS_COMMON_PAD_H_
#define DALI_KERNELS_COMMON_PAD_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <vector>
#include "dali/kernels/alloc.h"
#include "dali/core/span.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_common.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cuda_impl.cuh"

namespace dali {
namespace kernels {

template <typename Type, int Dims>
class DLL_PUBLIC PadGPU {
 private:
  static constexpr size_t kBlockDim = 512;
  static constexpr size_t kBlockSize = 64 * kBlockDim;
  size_t block_count_ = 0;

 public:
  DLL_PUBLIC PadGPU() = default;

  /**
  * @brief Find the biggest dimension on a given `axis`.
  * This dimension `biggest_dim_size` will be the new dimension
  * on the axis for all the samples of the batch.
  */
  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<Type, Dims> &in,
                                      int axis) {
    auto sample_dim = in.tensor_shape(0).sample_dim();
    KernelRequirements req;
    if (in.num_elements() == 0) {
      req.output_shapes = {in.shape};
    } else {
      // Keeping the code flexible to be able to extend for tuple of axes
      std::vector<size_t> padding_axes;
      if (axis < 0) {
        padding_axes.resize(sample_dim);
        std::iota(padding_axes.begin(), padding_axes.end(), 0);
      } else {
        padding_axes.push_back(axis);
      }

      using DimType =
        typename std::remove_reference<decltype(in.tensor_shape(0)[0])>::type;

      std::vector<DimType> biggest_dim_sizes(sample_dim, 0);
      auto num_samples = static_cast<size_t>(in.num_samples());
      for (size_t i = 0; i < num_samples; ++i) {
        for (auto axis : padding_axes) {
          auto dim = in.tensor_shape(i)[axis];
          if (dim > biggest_dim_sizes[axis]) {
            biggest_dim_sizes[axis] = dim;
          }
        }
      }

      block_count_ = 0;
      std::vector<TensorShape<DynamicDimensions>> tl_shape;
      tl_shape.reserve(num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        auto ts = in.tensor_shape(i);
        for (auto axis : padding_axes) {
          ts[axis] = biggest_dim_sizes[axis];
        }
        block_count_ += std::ceil((volume(ts)) / static_cast<float>(kBlockSize));
        tl_shape.push_back(ts);
      }

      req.output_shapes = {TensorListShape<DynamicDimensions>(tl_shape)};

      ScratchpadEstimator se;
      se.add<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
      se.add<detail::SampleDesc<Dims>>(AllocType::GPU, num_samples);

      se.add<detail::BlockDesc>(AllocType::Host, block_count_);
      se.add<detail::BlockDesc>(AllocType::GPU, block_count_);
      req.scratch_sizes = se.sizes;
    }
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &context, OutListGPU<Type, Dims> &out,
                      const InListGPU<Type, Dims> &in, Type padding_val) {
    auto num_samples = static_cast<size_t>(in.num_samples());

    detail::SampleDesc<Dims>* sample_descs_cpu =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(AllocType::Host,
                                                             num_samples);
    detail::BlockDesc *block_descs_cpu =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    std::vector<size_t> sample_sizes(in.size());

    for (int i = 0; i < in.size(); i++) {
      auto out_shape = out.tensor_shape(i);
      auto out_strides = GetStrides<Dims>(out_shape);
      auto out_vol = volume(out_shape);
      const auto in_shape = in.tensor_shape(i);
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = GetStrides<Dims>(in_shape);
      sample_desc.out_strides = out_strides;
      sample_desc.out_shape = in_shape.shape;
      sample_desc.padded_out_shape = out_shape.shape;
      sample_desc.padding_val = padding_val;
      sample_desc.in = in.tensor_data(i);
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = out_vol;
    }

    // Calculate super-block offset and size
    size_t block_idx = 0;
    for (int i = 0; i < static_cast<int>(num_samples); i++) {
      size_t offset = 0;
      size_t remaining = sample_sizes[i];
      while (remaining > 0) {
        size_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    // Transfer to GPU
    detail::SampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(
        AllocType::GPU, num_samples);
    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(
        AllocType::GPU, block_count_);

    // Memory is allocated contiguously, so we launch only one cudaMemcpyAsync
    size_t total_bytes = num_samples * sizeof(detail::SampleDesc<Dims>)
                         + block_count_ * sizeof(detail::BlockDesc);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu,
                    total_bytes,
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const auto grid = block_count_;
    detail::SliceFlipNormalizePermutePadKernel<Type, Type, Dims, false>
        <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs, nullptr,
                                                     nullptr, 0);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_PAD_H_
