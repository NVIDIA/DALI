// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_GPU_CUH_
#define DALI_KERNELS_SLICE_SLICE_GPU_CUH_

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/core/fast_div.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_kernel_utils.h"

__device__ DALI_FORCEINLINE bool __ldg(const bool *ptr) {
  return __ldg(reinterpret_cast<const dali::kernels::type_of_size<sizeof(bool)> *>(ptr));
}

namespace dali {
namespace kernels {

namespace {  // NOLINT

DALI_HOST_DEV DALI_FORCEINLINE bool is_out_of_bounds(int64_t idx, int64_t data_extent) {
  // check idx < 0 and idx >= data_extent at once
  return static_cast<uint64_t>(idx) >= static_cast<uint64_t>(data_extent);
}

}  // namespace


namespace slice_impl {

template <int Dims>
struct SliceSampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;

  TensorShape<Dims> out_shape;
  TensorShape<Dims> in_shape;
  TensorShape<Dims> anchor;
  TensorShape<Dims> step;

  const void *__restrict__ fill_values;
  int channel_dim;
  bool need_pad;

  fast_div<uint64_t> out_strides[Dims];
  TensorShape<Dims> in_strides;
};

struct SliceBlockDesc {
  int sampleIdx;
  uint64_t offset;
  uint64_t size;
};

template <typename T>
union PackedBuffer {
  using PackedType = uint32_t;
  static constexpr size_t kCapacity =
      sizeof(T) >= sizeof(PackedType) ? 1 : sizeof(PackedType) / sizeof(T);

  T values[kCapacity];
  PackedType raw;

  __device__ inline void store(T *mem, size_t count) {
    if (kCapacity == 1) {
      *mem = *values;
    } else if (count == kCapacity && reinterpret_cast<uintptr_t>(mem) % sizeof(PackedType) == 0) {
      *reinterpret_cast<PackedType *>(mem) = raw;
    } else {
      #pragma unroll
      for (size_t i = 0; i < count; i++) {
        mem[i] = values[i];
      }
    }
  }
};

/**
 * @brief Simplified algorithm when no padding is necessary
 * @remarks `in` should have `anchor` pre-applied and `stride` should have `step` pre-applied
 */
template <int Dims, typename OutputType, typename InputType>
__device__ void SliceFuncNoPad(OutputType *__restrict__ out, const InputType *__restrict__ in,
                               const fast_div<uint64_t> *out_strides, const int64_t *in_strides,
                               const int64_t *anchor, const int64_t *step, uint64_t offset,
                               uint64_t block_end) {
  if (Dims > 1 && step[Dims - 1] == 1 && step[Dims - 2] == 1 && anchor[Dims - 1] == 0 &&
      out_strides[Dims - 1] == static_cast<uint32_t>(in_strides[Dims - 1])) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFuncNoPad<NextDims, OutputType, InputType>(out, in, out_strides, in_strides, anchor, step,
                                                    offset, block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x * PackedBuffer<OutputType>::kCapacity) {
    PackedBuffer<OutputType> result;

    uint64_t i;
    #pragma unroll
    for (i = 0; i < PackedBuffer<OutputType>::kCapacity; i++) {
      uint64_t idx = offset + i;
      if (idx >= block_end)
        break;
      uint64_t in_idx = 0;

      #pragma unroll
      for (int d = 0; d < Dims; d++) {
        int i_d = div_mod(idx, idx, out_strides[d]);
        in_idx += i_d * in_strides[d];
      }
      in_idx += idx * step[Dims - 1];
      result.values[i] = clamp<OutputType>(in[in_idx]);
    }
    result.store(&out[offset], i);
  }
}

/**
 * @brief General algorithm that allows for padding in any dimension
 * @remarks `in` refers to the beginning of the input (not the slice anchor)
 */
template <int Dims, typename OutputType, typename InputType, bool AllDims = true>
__device__ void SliceFunc(OutputType *__restrict__ out, const InputType *__restrict__ in,
                          const fast_div<uint64_t> *out_strides, const int64_t *in_strides,
                          const int64_t *out_shape, const int64_t *in_shape, const int64_t *anchor,
                          const int64_t *step, const OutputType *__restrict__ fill_values,
                          int channel_dim, uint64_t offset, uint64_t block_end) {
  if (Dims > 1 && step[Dims - 1] == 1 && step[Dims - 2] == 1 && anchor[Dims - 1] == 0 &&
      in_shape[Dims - 1] == out_shape[Dims - 1] && channel_dim != Dims - 1) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFunc<NextDims, OutputType, InputType, false>(out, in, out_strides, in_strides, out_shape,
                                                      in_shape, anchor, step, fill_values,
                                                      channel_dim, offset, block_end);
    return;
  }

  constexpr int LastDim = Dims - 1;
  int64_t inner_in_anchor = anchor[LastDim];
  int64_t inner_in_extent = in_shape[LastDim];
  if (!AllDims) {  // if we fused dimensions, adjust inner dimension's anchor and extent
    inner_in_anchor = anchor[LastDim] * in_strides[LastDim];
    inner_in_extent = Dims > 1 ? in_strides[LastDim - 1] : in_shape[LastDim] * in_strides[LastDim];
  }

  for (; offset < block_end; offset += blockDim.x * PackedBuffer<OutputType>::kCapacity) {
    PackedBuffer<OutputType> result;

    uint64_t i;
#ifndef __clang__
    #pragma unroll
#endif
    for (i = 0; i < PackedBuffer<OutputType>::kCapacity; i++) {
      uint64_t idx = offset + i;
      if (idx >= block_end)
        break;

      // If no dimensions were skipped (AllDims=true) we can avoid division in the last dimension,
      // because know the strides are 1 (or we treat them as 1 if we fused dimensions)
      int i_c = 0;
      int i_d;
      bool out_of_bounds = false;
      uint64_t in_idx = 0;

      #pragma unroll
      for (int d = 0; d < Dims - 1; d++) {
        i_d = div_mod(idx, idx, out_strides[d]);
        if (d == channel_dim)
          i_c = i_d;
        i_d = anchor[d] + i_d * step[d];
        out_of_bounds |= is_out_of_bounds(i_d, in_shape[d]);
        in_idx += i_d * in_strides[d];
      }

      constexpr int d = LastDim;
      i_d = idx;
      if (AllDims && d == channel_dim)
        i_c = i_d;
      i_d = inner_in_anchor + i_d * step[d];
      out_of_bounds |= is_out_of_bounds(i_d, inner_in_extent);
      in_idx += i_d;

      // Fill values are reused a lot, so let's make sure they are cached (by using __ldg())
      OutputType value = __ldg(&fill_values[i_c]);
      if (!out_of_bounds)
        value = clamp<OutputType>(in[in_idx]);
      result.values[i] = value;
    }
    result.store(&out[offset], i);
  }
}

template <typename OutputType, typename InputType, int Dims, bool SupportPad>
__global__ void SliceKernel(const SliceSampleDesc<Dims> *samples, const SliceBlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  uint64_t offset = blocks[blockIdx.x].offset + threadIdx.x * PackedBuffer<OutputType>::kCapacity;
  uint64_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<OutputType *>(sample.out);
  auto *in = static_cast<const InputType *>(sample.in);
  auto *out_strides = sample.out_strides;
  auto *in_strides = sample.in_strides.data();
  auto *anchor = sample.anchor.data();
  auto *step = sample.step.data();
  if (SupportPad && sample.need_pad) {
    auto *in_shape = sample.in_shape.data();
    auto *out_shape = sample.out_shape.data();
    auto *fill_values = static_cast<const OutputType *>(sample.fill_values);
    auto channel_dim = sample.channel_dim;
    SliceFunc<Dims>(out, in, out_strides, in_strides, out_shape, in_shape, anchor, step,
                    fill_values, channel_dim, offset, block_end);
  } else {
    SliceFuncNoPad<Dims>(out, in, out_strides, in_strides, anchor, step, offset, block_end);
  }
}

}  // namespace slice_impl

template <typename OutputType, typename InputType, int Dims>
class SliceGPU {
 private:
  static constexpr uint64_t kBlockDim = 256;
  static constexpr uint64_t kMinBlockSize = 4 * kBlockDim;
  static constexpr uint64_t kMaxBlockSize = 64 * kBlockDim;

  uint64_t block_size_ = kMaxBlockSize;
  uint64_t block_count_ = 0;
  int blocks_per_sm_ = 0;

 public:
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<SliceArgs<OutputType, Dims>> &slice_args) {
    KernelRequirements req;
    auto num_samples = in.size();

    nfill_values_ = 0;
    for (const auto &args : slice_args) {
      if (nfill_values_ == 0) {
        nfill_values_ = args.fill_values.size();
      } else {
        if (nfill_values_ != static_cast<int>(args.fill_values.size()))
          throw std::invalid_argument(
              "The number of fill values should be the same for all the samples");
      }
    }
    if (nfill_values_ == 0) {
      default_fill_values_ = true;
      nfill_values_ = 1;
    } else if (nfill_values_ > 1) {
      for (const auto &args : slice_args) {
        if (args.channel_dim < 0 || args.channel_dim >= Dims)
          throw std::invalid_argument(
              "Channel dim must be valid for multi-channel fill values");
        if (nfill_values_ != args.shape[args.channel_dim])
          throw std::invalid_argument(
              "The number of fill values should match the number of channels in the output slice");
      }
    }

    std::vector<int64_t> sample_sizes;
    sample_sizes.reserve(slice_args.size());
    uint64_t total_volume = 0;
    for (auto &args : slice_args) {
      sample_sizes.push_back(volume(args.shape));
      total_volume += volume(args.shape);
    }

    if (blocks_per_sm_ == 0) {
      CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm_, slice_impl::SliceKernel<OutputType, InputType, Dims, false>, kBlockDim,
          0));
    }
    unsigned max_active_blocks = blocks_per_sm_ * GetSmCount();
    uint64_t waves = div_ceil(total_volume + 1, kMaxBlockSize * max_active_blocks);
    unsigned block_align = 32 * slice_impl::PackedBuffer<OutputType>::kCapacity;
    block_size_ = align_up(div_ceil(total_volume, max_active_blocks * waves), block_align);
    if (block_size_ < kMinBlockSize) block_size_ = kMinBlockSize;
    if (block_size_ > kMaxBlockSize) block_size_ = kMaxBlockSize;

    block_count_ = 0;
    for (auto sample_size : sample_sizes) {
      block_count_ += div_ceil(sample_size, block_size_);
    }

    req.output_shapes = { GetOutputShapes<Dims>(in.shape, slice_args) };
    return req;
  }

  void Run(KernelContext &context, OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<SliceArgs<OutputType, Dims>> &slice_args) {
    if (block_count_ == 0) {
      return;  // No data to copy
    }
    const auto num_samples = in.size();
    OutputType *fill_values_cpu =
        context.scratchpad->AllocatePinned<OutputType>(num_samples * nfill_values_);
    for (int i = 0; i < in.size(); i++) {
      if (default_fill_values_) {
        assert(nfill_values_ == 1);
        fill_values_cpu[i] = OutputType{};
      } else {
        auto *fill_values = fill_values_cpu + i * nfill_values_;
        for (int d = 0; d < nfill_values_; d++)
          fill_values[d] = slice_args[i].fill_values[d];
      }
    }
    OutputType *fill_values_gpu = context.scratchpad->ToGPU(
        context.gpu.stream, make_span(fill_values_cpu, num_samples * nfill_values_));
    CUDA_CALL(cudaGetLastError());

    // Host memory
    slice_impl::SliceSampleDesc<Dims> *sample_descs_cpu =
        context.scratchpad->AllocatePinned<slice_impl::SliceSampleDesc<Dims>>(num_samples);
    slice_impl::SliceBlockDesc *block_descs_cpu =
        context.scratchpad->AllocatePinned<slice_impl::SliceBlockDesc>(block_count_);

    bool any_padded_sample = false;
    std::vector<int64_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      const auto out_shape = out.tensor_shape(i);
      const auto anchor = slice_args[i].anchor;
      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = GetStrides(in_shape);
      CalcStrides(sample_desc.out_strides, out_shape);
      sample_desc.anchor = anchor;
      sample_desc.in_shape = in_shape;
      sample_desc.out_shape = out_shape;
      sample_desc.step = slice_args[i].step;

      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(out_shape);

      // fill values points to gpu memory
      sample_desc.fill_values = fill_values_gpu + i * nfill_values_;
      sample_desc.channel_dim = nfill_values_ > 1 ? slice_args[i].channel_dim : -1;
      sample_desc.need_pad = NeedPad(Dims, anchor, in_shape, out_shape);

      // pre-anchor and step if there is no padding
      if (!sample_desc.need_pad) {
        const InputType *in_data = in.tensor_data(i);
        for (int d = 0; d < Dims; ++d) {
          in_data += sample_desc.anchor[d] * sample_desc.in_strides[d];
          sample_desc.in_strides[d] *= sample_desc.step[d];
        }
        sample_desc.in = in_data;
      } else {
        sample_desc.in = in.tensor_data(i);
      }

      any_padded_sample |= sample_desc.need_pad;
    }

    int64_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      uint64_t offset = 0;
      uint64_t remaining = sample_sizes[i];
      while (remaining > 0) {
        uint64_t size = remaining < block_size_ ? remaining : block_size_;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    slice_impl::SliceSampleDesc<Dims> *sample_descs;
    slice_impl::SliceBlockDesc *block_descs;
    std::tie(sample_descs, block_descs) =
        context.scratchpad->ToContiguousGPU(context.gpu.stream,
                                            make_cspan(sample_descs_cpu, num_samples),
                                            make_cspan(block_descs_cpu, block_count_));
    CUDA_CALL(cudaGetLastError());

    const auto grid = block_count_;
    BOOL_SWITCH(any_padded_sample, NeedPad, (
      slice_impl::SliceKernel<OutputType, InputType, Dims, NeedPad>
        <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs);
    ));  // NOLINT
    CUDA_CALL(cudaGetLastError());
  }

 private:
  int nfill_values_ = 0;
  bool default_fill_values_ = false;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_CUH_
