// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_GPU_H_

#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_error.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/core/fast_div.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/slice/slice_kernel_utils.h"

__device__ DALI_FORCEINLINE bool __ldg(const bool* ptr) {
  return __ldg(reinterpret_cast<const dali::kernels::type_of_size<sizeof(bool)> *>(ptr));
}

namespace dali {
namespace kernels {

namespace {

DALI_HOST_DEV DALI_FORCEINLINE bool is_out_of_bounds(int64_t idx, int64_t data_extent) {
  // check idx < 0 and idx >= data_extent at once
  return static_cast<uint64_t>(idx) >= static_cast<uint64_t>(data_extent);
}

}  // namespace


namespace detail {

template <int Dims>
struct SliceSampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;

  TensorShape<Dims> out_shape;
  TensorShape<Dims> in_shape;
  TensorShape<Dims> anchor;

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

/**
 * @brief Simplified algorithm when no padding is necessary
 * @remarks `in` already refers to the slice anchor start
 */
template <int Dims, typename OutputType, typename InputType>
__device__ void SliceFuncNoPad(OutputType *__restrict__ out, const InputType *__restrict__ in,
                               const fast_div<uint64_t> *out_strides, const int64_t *in_strides,
                               uint64_t offset, uint64_t block_end) {
  if (Dims > 1 && out_strides[Dims - 1] == static_cast<uint32_t>(in_strides[Dims - 1])) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFuncNoPad<NextDims, OutputType, InputType>(out, in, out_strides, in_strides, offset,
                                                    block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x) {
    uint64_t idx = offset;
    uint64_t out_idx = idx;
    uint64_t in_idx = 0;

    #pragma unroll
    for (int d = 0; d < Dims; d++) {
      int i_d = div_mod(idx, idx, out_strides[d]);
      in_idx += i_d * in_strides[d];
    }
    in_idx += idx;  // remaining dims have equal strides
    out[out_idx] = clamp<OutputType>(in[in_idx]);
  }
}

/**
 * @brief General algorithm that allows for padding in any dimension
 * @remarks `in` refers to the beginning of the input (not the slice anchor)
 * @remarks `AllDims=true` means that Dims refer to the actual number of dimensions,
 *           meaning we haven't skipped last dimensions that have same input and output strides
 */
template <int Dims, typename OutputType, typename InputType, bool AllDims = true>
__device__ void SliceFunc(OutputType *__restrict__ out, const InputType *__restrict__ in,
                          const fast_div<uint64_t> *out_strides, const int64_t *in_strides,
                          const int64_t *out_shape, const int64_t *in_shape, const int64_t *anchor,
                          const OutputType *__restrict__ fill_values, int channel_dim,
                          uint64_t offset, uint64_t block_end) {
  if (Dims > 1 && anchor[Dims - 1] == 0 && in_shape[Dims - 1] == out_shape[Dims - 1] &&
      channel_dim != Dims - 1) {
    const int NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFunc<NextDims, OutputType, InputType, false>(out, in, out_strides, in_strides, out_shape,
                                                      in_shape, anchor, fill_values, channel_dim,
                                                      offset, block_end);
    return;
  }

  constexpr int LastDim = Dims - 1;
  int64_t inner_in_anchor = anchor[LastDim];
  int64_t inner_in_extent = in_shape[LastDim];
  if (!AllDims) {  // if we fused dimensions, adjust inner dimension's anchor and extent
    inner_in_anchor = anchor[LastDim] * in_strides[LastDim];
    inner_in_extent = Dims > 1 ? in_strides[LastDim - 1] : in_shape[LastDim] * in_strides[LastDim];
  }

  for (; offset < block_end; offset += blockDim.x) {
    uint64_t idx = offset;
    uint64_t out_idx = idx;

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
      out_of_bounds |= is_out_of_bounds(anchor[d] + i_d, in_shape[d]);
      if (!out_of_bounds)
        in_idx += i_d * in_strides[d];
    }

    constexpr int d = LastDim;
    i_d = idx;  // out_strides[d] is 1
    if (AllDims && d == channel_dim)
      i_c = i_d;
    out_of_bounds |= is_out_of_bounds(inner_in_anchor + i_d, inner_in_extent);
    if (!out_of_bounds)
      in_idx += i_d;  // in_strides[d] is 1

    // Fill values are reused a lot, so let's make sure they are cached (by using __ldg())
    out[out_idx] = out_of_bounds ? __ldg(&fill_values[i_c]) : clamp<OutputType>(in[in_idx]);
  }
}

template <typename OutputType, typename InputType, int Dims, bool SupportPad>
__global__ void SliceKernel(const SliceSampleDesc<Dims> *samples, const SliceBlockDesc *blocks) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  uint64_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  uint64_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<OutputType*>(sample.out);
  auto *in = static_cast<const InputType*>(sample.in);
  auto *out_strides = sample.out_strides;
  auto *in_strides = sample.in_strides.data();
  if (SupportPad && sample.need_pad) {
    auto *anchor = sample.anchor.data();
    auto *in_shape = sample.in_shape.data();
    auto *out_shape = sample.out_shape.data();
    auto *fill_values = static_cast<const OutputType*>(sample.fill_values);
    auto channel_dim = sample.channel_dim;
    SliceFunc<Dims>(out, in, out_strides, in_strides, out_shape, in_shape, anchor, fill_values,
                    channel_dim, offset, block_end);
  } else {
    SliceFuncNoPad<Dims>(out, in, out_strides, in_strides, offset, block_end);
  }
}

}  // namespace detail

template <typename OutputType, typename InputType, int Dims>
class SliceGPU {
 private:
  static constexpr int64_t kBlockDim = 256;
  static constexpr int64_t kBlockSize = 64 * kBlockDim;
  int64_t block_count_ = 0;

 public:
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<SliceArgs<OutputType, Dims>> &slice_args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    auto num_samples = in.size();

    nfill_values_ = 0;
    for (const auto& args : slice_args) {
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
      for (const auto& args : slice_args) {
        if (args.channel_dim < 0 || args.channel_dim >= Dims)
          throw std::invalid_argument(
              "Channel dim must be valid for multi-channel fill values");
        if (nfill_values_ != args.shape[args.channel_dim])
          throw std::invalid_argument(
              "The number of fill values should match the number of channels in the output slice");
      }
    }

    se.add<mm::memory_kind::host, OutputType>(num_samples * nfill_values_);
    se.add<mm::memory_kind::device, OutputType>(num_samples * nfill_values_);

    se.add<mm::memory_kind::host, detail::SliceSampleDesc<Dims>>(num_samples);
    se.add<mm::memory_kind::device, detail::SliceSampleDesc<Dims>>(num_samples);

    std::vector<int64_t> sample_sizes;
    sample_sizes.reserve(slice_args.size());
    for (auto &args : slice_args) {
      sample_sizes.push_back(volume(args.shape));
    }

    block_count_ = 0;
    for (auto sample_size : sample_sizes) {
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<mm::memory_kind::host, detail::SliceBlockDesc>(block_count_);
    se.add<mm::memory_kind::device, detail::SliceBlockDesc>(block_count_);
    req.scratch_sizes = se.sizes;

    req.output_shapes = { GetOutputShapes<Dims>(in.shape, slice_args) };
    return req;
  }

  void Run(KernelContext &context,
           OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<SliceArgs<OutputType, Dims>> &slice_args) {
    if (block_count_ == 0) {
      return;  // No data to copy
    }
    const auto num_samples = in.size();
    OutputType *fill_values_cpu =
        context.scratchpad->AllocateHost<OutputType>(num_samples * nfill_values_);
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
    detail::SliceSampleDesc<Dims> *sample_descs_cpu =
        context.scratchpad->AllocateHost<detail::SliceSampleDesc<Dims>>(num_samples);
    detail::SliceBlockDesc *block_descs_cpu =
        context.scratchpad->AllocateHost<detail::SliceBlockDesc>(block_count_);

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

      const InputType *in_data = in.tensor_data(i);
      // `sample_desc.in` is expected to point to the slice anchor
      for (int d = 0; d < Dims; d++)
        in_data += anchor[d] * sample_desc.in_strides[d];

      sample_desc.out = out.tensor_data(i);
      sample_desc.in = in_data;
      sample_sizes[i] = volume(out_shape);

      // fill values points to gpu memory
      sample_desc.fill_values = fill_values_gpu + i * nfill_values_;
      sample_desc.channel_dim = nfill_values_ > 1 ? slice_args[i].channel_dim : -1;
      sample_desc.need_pad = NeedPad(Dims, anchor, in_shape, out_shape);
      any_padded_sample |= sample_desc.need_pad;
    }

    int64_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      uint64_t offset = 0;
      uint64_t remaining = sample_sizes[i];
      while (remaining > 0) {
        uint64_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    detail::SliceSampleDesc<Dims> *sample_descs;
    detail::SliceBlockDesc *block_descs;
    std::tie(sample_descs, block_descs) =
        context.scratchpad->ToContiguousGPU(context.gpu.stream,
                                            make_cspan(sample_descs_cpu, num_samples),
                                            make_cspan(block_descs_cpu, block_count_));
    CUDA_CALL(cudaGetLastError());

    const auto grid = block_count_;
    BOOL_SWITCH(any_padded_sample, NeedPad, (
      detail::SliceKernel<OutputType, InputType, Dims, NeedPad>
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

#endif  // DALI_KERNELS_SLICE_SLICE_GPU_H_
