// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_rt_utils.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cuda_impl.cuh"

namespace dali {
namespace kernels {

template <typename OutputType, typename InputType, int Dims>
class SliceFlipNormalizePermutePadGpu {
 private:
  static constexpr size_t kBlockDim = 256;
  size_t block_size_ = 32 * kBlockDim;
  size_t block_count_ = 0;

  using ProcessedArgs = slice_impl::SliceFlipNormalizePermutePadProcessedArgs<Dims>;
  std::vector<ProcessedArgs> processed_args_;
  int norm_args_size_ = -1;
  bool has_channels_ = false;
  bool need_normalize_ = false;
  int nfill_values_ = 0;
  int channel_dim_ = -1;

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
    nfill_values_ = 0;
    channel_dim_ = -1;

    processed_args_.clear();
    processed_args_.reserve(args.size());
    for (size_t i = 0; i < num_samples; i++) {
      auto in_shape = in.tensor_shape(i);
      processed_args_.emplace_back(slice_impl::ProcessArgs(args[i], in_shape));
      auto &sample_args = processed_args_.back();
      if (i == 0) {
        norm_args_size_ = sample_args.mean.size();
        nfill_values_ = sample_args.fill_values.size();
        need_normalize_ = norm_args_size_ > 0;
        has_channels_ = norm_args_size_ > 1 || nfill_values_ > 1;
        channel_dim_ = sample_args.channel_dim;
      } else {
        // Checking all the samples are consistent
        if (norm_args_size_ != static_cast<int>(sample_args.mean.size()) ||
            norm_args_size_ != static_cast<int>(sample_args.inv_stddev.size()))
          throw std::invalid_argument(
            "Normalization arguments should have the same size for all the samples");
        if (nfill_values_ != static_cast<int>(sample_args.fill_values.size()))
          throw std::invalid_argument("Fill values should have the same size for all the samples");
        if (channel_dim_ != sample_args.channel_dim)
          throw std::invalid_argument("channel dim should be the same for all the samples");
      }
    }

    if (need_normalize_) {
      se.add<mm::memory_kind::pinned, float>(num_samples * norm_args_size_);
      se.add<mm::memory_kind::pinned, float>(num_samples * norm_args_size_);
      se.add<mm::memory_kind::device, float>(num_samples * norm_args_size_);
      se.add<mm::memory_kind::device, float>(num_samples * norm_args_size_);
    }

    assert(nfill_values_ > 0);
    se.add<mm::memory_kind::pinned, OutputType>(num_samples * nfill_values_);
    se.add<mm::memory_kind::device, OutputType>(num_samples * nfill_values_);

    se.add<mm::memory_kind::pinned, slice_impl::SampleDesc<Dims>>(num_samples);
    se.add<mm::memory_kind::device, slice_impl::SampleDesc<Dims>>(num_samples);

    int blocks_per_sm_;
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm_,
      slice_impl::SliceFlipNormalizePermutePadKernel<
          false, false, false, OutputType, InputType, Dims>,
      kBlockDim,
      0));

    block_count_ = 0;
    auto number_of_blocks = GetSmCount() * blocks_per_sm_;
    size_t all_sample_sizes = 0;

    for (auto &elem : args) {
      size_t sample_size = volume(elem.shape);
      all_sample_sizes += sample_size;
      block_count_ += static_cast<size_t>(std::ceil(
        sample_size / static_cast<float>(block_size_)));
    }
    auto block_remainder = block_count_ % number_of_blocks;
    if (block_remainder != 0) {
      block_count_ += number_of_blocks - block_remainder;
    }
    block_size_ = div_ceil(all_sample_sizes, block_count_);

    block_count_ = 0;
    for (auto &elem : args) {
      size_t sample_size = volume(elem.shape);
      block_count_ += static_cast<size_t>(std::ceil(
        sample_size / static_cast<float>(block_size_)));
    }

    se.add<mm::memory_kind::pinned, slice_impl::BlockDesc>(block_count_);
    se.add<mm::memory_kind::device, slice_impl::BlockDesc>(block_count_);
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

  bool CanCollapseLastDim(slice_impl::SampleDesc<Dims> &sample_desc, int last_dim) {
    // We fuse the last dimension with the previous IF:
    // There are at least 2 dimensions
    bool at_least_2_dims = last_dim > 0;
    // Last dimension is not sliced/padded/permuted
    // if out_stride[last_dim] == abs(in_stride[last_dim]), then it is not permuted
    // if anchor[last_dim] == 0 && out_shape[last_dim] == in_shape[last_dim],
    // then it is not sliced/padded
    bool no_permute = sample_desc.out_strides[last_dim] ==
                      static_cast<uint64_t>(abs(sample_desc.in_strides[last_dim]));
    bool no_sliced_or_pad = sample_desc.anchor[last_dim] == 0 &&
                            sample_desc.out_shape[last_dim] == sample_desc.in_shape[last_dim];
    // Neither of the two dimensions to be merged are the channel dimension
    bool no_channel_dim =
        sample_desc.channel_dim != last_dim && sample_desc.channel_dim != (last_dim - 1);
    // The in_strides of the last dimension and the former one align
    // in_strides[last_dim] * in_shape[last_dim] == in_strides[last_dim - 1]
    // (this check also makes sure that we collapse dims that have the same sign of strides
    // meaning we can merge two dimensions if they are either not flipped or both of them are)
    bool can_collapse = sample_desc.in_strides[last_dim] * sample_desc.in_shape[last_dim] ==
                        sample_desc.in_strides[last_dim - 1];
    return at_least_2_dims && no_permute && no_sliced_or_pad && no_channel_dim && can_collapse;
  }

  void Run(KernelContext &context,
           const OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<Args> &args) {
    (void) args;
    if (block_count_ == 0) {
      return;  // no data to copy
    }

    const auto num_samples = in.size();

    float *norm_add_cpu = nullptr, *norm_mul_cpu = nullptr;
    float *norm_add_gpu = nullptr, *norm_mul_gpu = nullptr;
    if (need_normalize_) {
      norm_add_cpu = context.scratchpad->AllocatePinned<float>(num_samples * norm_args_size_);
      norm_mul_cpu = context.scratchpad->AllocatePinned<float>(num_samples * norm_args_size_);
      for (int i = 0; i < num_samples; i++) {
        auto &sample_args = processed_args_[i];
        auto *norm_add_data = norm_add_cpu + i * norm_args_size_;
        auto *norm_mul_data = norm_mul_cpu + i * norm_args_size_;
        for (int d = 0; d < norm_args_size_; d++) {
          norm_add_data[d] = -sample_args.mean[d] * sample_args.inv_stddev[d];
          norm_mul_data[d] = sample_args.inv_stddev[d];
        }
      }
    }

    assert(nfill_values_ > 0);
    OutputType *fill_values_cpu = context.scratchpad->AllocatePinned<OutputType>(
          num_samples * nfill_values_);
    for (int i = 0; i < num_samples; i++) {
      auto *fill_values = fill_values_cpu + i * nfill_values_;
      auto &sample_args = processed_args_[i];
      for (int d = 0; d < nfill_values_; d++)
        fill_values[d] = sample_args.fill_values[d];
    }
    OutputType *fill_values_gpu;
    std::tie(norm_add_gpu, norm_mul_gpu, fill_values_gpu) = context.scratchpad->ToContiguousGPU(
      context.gpu.stream,
      make_span(norm_add_cpu, num_samples * norm_args_size_),
      make_span(norm_mul_cpu, num_samples * norm_args_size_),
      make_span(fill_values_cpu, num_samples * nfill_values_));

    auto *sample_descs_cpu =
        context.scratchpad->AllocatePinned<slice_impl::SampleDesc<Dims>>(num_samples);
    auto *block_descs_cpu =
        context.scratchpad->AllocatePinned<slice_impl::BlockDesc>(block_count_);

    bool need_pad = false, need_flip = false;
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      auto &processed_args = processed_args_[i];
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = processed_args.in_strides;
      for (int d = 0; d < Dims; d++) sample_desc.out_strides[d] = processed_args.out_strides[d];
      sample_desc.out_shape = processed_args.out_shape;
      sample_desc.in_shape = processed_args.in_shape;
      sample_desc.anchor = processed_args.anchor;
      sample_desc.norm_add = norm_add_gpu + i * norm_args_size_;
      sample_desc.norm_mul = norm_mul_gpu + i * norm_args_size_;
      sample_desc.fill_values = fill_values_gpu + i * norm_args_size_;
      sample_desc.channel_dim = processed_args.channel_dim;
      sample_desc.in = in.tensor_data(i) + processed_args.input_offset;
      sample_desc.out = out.tensor_data(i);
      sample_desc.need_pad =
          NeedPad(Dims, processed_args.anchor.data(), processed_args.in_shape.data(),
                  processed_args.out_shape.data());
      need_pad |= sample_desc.need_pad;
      sample_desc.need_flip = false;
      for (int d = 0; d < Dims; d++)
        sample_desc.need_flip |= processed_args.in_strides[d] < 0;
      need_flip |= sample_desc.need_flip;

      int last_dim = Dims - 1;
      while (CanCollapseLastDim(sample_desc, last_dim)) {
        last_dim--;
      }

      if (last_dim < Dims - 1) {
        int64_t stride = sample_desc.out_strides[last_dim];  // same as abs(in_strides[last_dim])
        sample_desc.anchor[last_dim]    *= stride;
        sample_desc.in_shape[last_dim]  *= stride;
        sample_desc.out_shape[last_dim] *= stride;
        sample_desc.out_strides[last_dim] = 1;
        sample_desc.in_strides[last_dim] = sample_desc.in_strides[last_dim] < 0 ? -1 : 1;
      }
      sample_desc.effective_ndim = last_dim + 1;
    }

    size_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      size_t offset = 0;
      size_t remaining = volume(processed_args_[i].out_shape);
      while (remaining > 0) {
        size_t size = remaining < block_size_ ? remaining : block_size_;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    slice_impl::SampleDesc<Dims> *sample_descs_gpu = nullptr;
    slice_impl::BlockDesc *block_descs_gpu = nullptr;
    std::tie(sample_descs_gpu, block_descs_gpu) = context.scratchpad->ToContiguousGPU(
        context.gpu.stream,
        make_span(sample_descs_cpu, num_samples),
        make_span(block_descs_cpu, block_count_));

    BOOL_SWITCH(need_pad, NeedPad, (
      BOOL_SWITCH(need_flip, NeedFlip, (
        BOOL_SWITCH(need_normalize_, NeedNormalize, (
          auto grid = block_count_;
          // need to handle __half due to compilation differences
          slice_impl::SliceFlipNormalizePermutePadKernel
            <NeedPad, NeedFlip, NeedNormalize,
            OutputType, InputType, Dims>
            <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs_gpu, block_descs_gpu);
        ));  // NOLINT
      ));  // NOLINT
    ));  // NOLINT

    CUDA_CALL(cudaGetLastError());
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_PAD_GPU_H_
