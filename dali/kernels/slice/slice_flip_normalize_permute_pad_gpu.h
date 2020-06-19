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
#include "dali/core/static_switch.h"
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
  
  int norm_args_size_ = -1;
  bool has_channels_ = false;
  bool need_normalize_ = false;
  bool default_fill_values_ = false;
  int nfill_values_ = 0;

 public:
  using Args = SliceFlipNormalizePermutePadArgs<Dims>;
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<Args> &args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    const size_t num_samples = in.size();

    norm_args_size_ = -1;
    has_channels_ = false;
    need_normalize_ = false;
    for (const auto& sample_args : args) {
      if (norm_args_size_ == -1) {
        norm_args_size_ = sample_args.mean.size();
        need_normalize_ = norm_args_size_ > 0;
        has_channels_ = norm_args_size_ > 1;
      }

      if (sample_args.mean.size() != norm_args_size_ || sample_args.inv_stddev.size() != norm_args_size_)
        throw std::invalid_argument("Normalization arguments should have the same size for all the samples");

      if (has_channels_) {
        if (sample_args.channel_dim < 0 || sample_args.channel_dim >= Dims)
          throw std::invalid_argument(
              "Channel dim must be valid for multi-channel normalization arguments");
        if (norm_args_size_ != sample_args.shape[sample_args.channel_dim])
          throw std::invalid_argument(
              "The number of normalization arguments should match the number of channels in the output slice");
      }
    }
    if (need_normalize_) {
      se.add<float>(AllocType::Host, num_samples * norm_args_size_);
      se.add<float>(AllocType::Host, num_samples * norm_args_size_);
      se.add<float>(AllocType::GPU,  num_samples * norm_args_size_);
      se.add<float>(AllocType::GPU,  num_samples * norm_args_size_);
    }

    nfill_values_ = 0;
    for (const auto& sample_args : args) {
      if (nfill_values_ == 0) {
        nfill_values_ = sample_args.fill_values.size();
      } else {
        if (nfill_values_ != sample_args.fill_values.size())
          throw std::invalid_argument(
              "The number of fill values should be the same for all the samples");
      }
    }
    if (nfill_values_ == 0) {
      default_fill_values_ = true;
      nfill_values_ = norm_args_size_;
    } else if (nfill_values_ > 1) {
      if (norm_args_size_ > 0 && norm_args_size_ != nfill_values_)
        throw std::invalid_argument(
            "Number of channels for fill values doesn't match the number of channels in the "
            "normalization arguments");
      for (const auto &sample_args : args) {
        if (sample_args.channel_dim < 0 || sample_args.channel_dim >= Dims)
          throw std::invalid_argument(
              "Channel dim must be valid for multi-channel fill values");
        if (nfill_values_ != sample_args.shape[sample_args.channel_dim])
          throw std::invalid_argument(
              "The number of fill values should match the number of channels in the output slice");
      }
    }

    se.add<OutputType>(AllocType::Host, num_samples * nfill_values_);
    se.add<OutputType>(AllocType::GPU, num_samples * nfill_values_);

    se.add<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    se.add<detail::SampleDesc<Dims>>(AllocType::GPU, num_samples);

    block_count_ = 0;
    for (auto &elem : args) {
      size_t sample_size = volume(elem.shape);
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<detail::BlockDesc>(AllocType::Host, block_count_);
    se.add<detail::BlockDesc>(AllocType::GPU, block_count_);
    req.scratch_sizes = se.sizes;

    auto in_shapes = in.shape;
    TensorListShape<Dims> output_shapes(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      TensorShape<Dims> out_shape(args[i].shape);
      CheckValidOutputShape(in_shapes[i], out_shape, args[i]);
      out_shape = permute(out_shape, args[i].permuted_dims);
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

    float *norm_add_cpu = nullptr, *norm_mul_cpu = nullptr;
    float *norm_add_gpu = nullptr, *norm_mul_gpu = nullptr;
    if (need_normalize_) {
      norm_add_cpu = context.scratchpad->Allocate<float>(AllocType::Host, num_samples * norm_args_size_);
      norm_mul_cpu = context.scratchpad->Allocate<float>(AllocType::Host, num_samples * norm_args_size_);
      for (int i = 0; i < num_samples; i++) {
        auto &sample_args = args[i];
        auto *norm_add_data = norm_add_cpu + i * norm_args_size_;
        auto *norm_mul_data = norm_mul_cpu + i * norm_args_size_;
        for (int d = 0; d < norm_args_size_; d++) {
          norm_add_data[d] = -sample_args.mean[d] * sample_args.inv_stddev[d];
          norm_mul_data[d] = sample_args.inv_stddev[d];
        }
      }

      norm_add_gpu = context.scratchpad->ToGPU(
          context.gpu.stream, make_span(norm_add_cpu, num_samples * norm_args_size_));
      CUDA_CALL(cudaGetLastError());

      norm_mul_gpu = context.scratchpad->ToGPU(
          context.gpu.stream, make_span(norm_mul_cpu, num_samples * norm_args_size_));
      CUDA_CALL(cudaGetLastError());

    }

    OutputType *fill_values_cpu =
        context.scratchpad->Allocate<OutputType>(AllocType::Host, num_samples * nfill_values_);
    for (int i = 0; i < num_samples; i++) {
      auto *fill_values = fill_values_cpu + i * nfill_values_;
      if (default_fill_values_) {
        for (int d = 0; d < nfill_values_; d++)
          fill_values[d] = OutputType(0);
      } else {
        for (int d = 0; d < nfill_values_; d++)
          fill_values[d] = args[i].fill_values[d];
      }
    }
    OutputType *fill_values_gpu = context.scratchpad->ToGPU(
        context.gpu.stream, make_span(fill_values_cpu, num_samples * nfill_values_));
    CUDA_CALL(cudaGetLastError());

    // Host memory
    auto *sample_descs_cpu =
        context.scratchpad->Allocate<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    auto *block_descs_cpu =
        context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    int channel_dim = -1;
    bool need_pad = false;
    std::vector<size_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      auto processed_args = detail::ProcessArgs(args[i], in_shape);
      if (has_channels_) {
        if (i == 0)
          channel_dim = processed_args.channel_dim;
        else if (channel_dim != processed_args.channel_dim)
          throw std::invalid_argument("Channel dim should be the same for every sample");
      }
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = processed_args.in_strides;
      for (int d = 0; d < Dims; d++) sample_desc.out_strides[d] = processed_args.out_strides[d];
      sample_desc.in_shape = processed_args.in_shape;
      sample_desc.anchor = processed_args.anchor;
      sample_desc.norm_add = norm_add_gpu + i * norm_args_size_;
      sample_desc.norm_mul = norm_mul_gpu + i * norm_args_size_;
      sample_desc.fill_values = fill_values_gpu + i * norm_args_size_;
      sample_desc.channel_dim = channel_dim;
      sample_desc.in = in.tensor_data(i) + processed_args.input_offset;
      sample_desc.out = out.tensor_data(i);
      sample_desc.need_pad =
          NeedPad(Dims, processed_args.anchor.data(), processed_args.in_shape.data(),
                  processed_args.out_shape.data());
      need_pad |= sample_desc.need_pad;
      sample_sizes[i] = volume(processed_args.out_shape);
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

    auto *sample_descs_gpu = context.scratchpad->ToGPU(
        context.gpu.stream, make_span(sample_descs_cpu, num_samples));
    CUDA_CALL(cudaGetLastError());

    auto *block_descs_gpu = context.scratchpad->ToGPU(
        context.gpu.stream, make_span(block_descs_cpu, block_count_));
    CUDA_CALL(cudaGetLastError());

    VALUE_SWITCH(need_pad ? 1 : 0, NeedPadInt, (0, 1), (
      VALUE_SWITCH(need_normalize_ ? 1 : 0, NeedNormalizeInt, (0, 1), (
        constexpr bool NeedPad = static_cast<bool>(NeedPadInt);
        constexpr bool NeedNormalize = static_cast<bool>(NeedNormalizeInt);
        auto grid = block_count_;
        detail::SliceFlipNormalizePermutePadKernel<NeedPad, NeedNormalize, OutputType, InputType, Dims>
          <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs_gpu, block_descs_gpu);
      ), ());
    ), ());
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_
