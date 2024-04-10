// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/common/cast_gpu.h"
#include "dali/core/convert.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {
namespace cast {

namespace impl {

struct SampleDesc {
  void *output;
  const void *input;
  uint32_t first_block;
  int64_t sample_size;
};

inline __device__ uint32_t FindSampleIdx(const SampleDesc *samples,
                                         unsigned nsamples) {
  uint32_t i = 0;
  for (uint32_t jump = (1 << (32 - __clz(nsamples) - 1)); jump; jump >>= 1) {
    if (i + jump < nsamples && samples[i + jump].first_block <= blockIdx.x)
      i += jump;
  }
  return i;
}

template <typename Out, typename In>
__global__ void BinSearchCastKernel(const SampleDesc *samples,
                                    unsigned nsamples, int block_sz) {
  int sample_idx = FindSampleIdx(samples, nsamples);
  SampleDesc sample = samples[sample_idx];
  auto *out = static_cast<Out *>(sample.output);
  const auto *in = static_cast<const In *>(sample.input);
  auto size = sample.sample_size;
  auto block_idx = blockIdx.x - sample.first_block;
  auto block_start = block_idx * block_sz;
  auto block_end = cuda_min<int64_t>(block_start + block_sz, size);
  for (unsigned x = threadIdx.x + block_start; x < block_end; x += blockDim.x) {
    out[x] = ConvertSat<Out>(in[x]);
  }
}

}  // namespace impl

template <typename Out, typename In>
void CastGPU<Out, In>::Run(KernelContext &ctx,
                           const TensorListView<StorageGPU, Out, 1> &out,
                           const TensorListView<StorageGPU, const In, 1> &in) {
  if (out.num_elements() != in.num_elements())
    throw std::invalid_argument("Different number of elements in output vs. input");

  int num_samples = in.num_samples();
  impl::SampleDesc *samples = ctx.scratchpad->AllocatePinned<impl::SampleDesc>(num_samples);

  constexpr int kBlockSize = 256;
  constexpr int kLogicalBlockSize = 4 * kBlockSize;
  uint32_t offset_blk = 0;
  int nonempty_nsamples = 0;
  for (int i = 0; i < num_samples; i++) {
    int64_t sample_size = in.shape.tensor_size(i);
    if (out.shape.tensor_size(i) != sample_size)
      throw std::invalid_argument("Different number of elements in output vs. input");
    if (sample_size == 0)
      continue;
    auto &sample = samples[nonempty_nsamples++];
    sample.output = out.tensor_data(i);
    sample.input = in.tensor_data(i);
    sample.first_block = offset_blk;
    sample.sample_size = sample_size;
    offset_blk += div_ceil(sample_size, kLogicalBlockSize);
  }

  if (nonempty_nsamples == 0)
    return;

  auto *samples_dev = ctx.scratchpad->ToGPU(
      ctx.gpu.stream, span<const impl::SampleDesc>(samples, nonempty_nsamples));

  impl::BinSearchCastKernel<Out, In><<<offset_blk, kBlockSize, 0, ctx.gpu.stream>>>(
      samples_dev, nonempty_nsamples, kLogicalBlockSize);
  CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_IMPL(Out, In) template struct DLL_PUBLIC CastGPU<Out, In>;

#define INSTANTIATE_FOREACH_INTYPE(Out) \
  INSTANTIATE_IMPL(Out, bool);          \
  INSTANTIATE_IMPL(Out, uint8_t);       \
  INSTANTIATE_IMPL(Out, uint16_t);      \
  INSTANTIATE_IMPL(Out, uint32_t);      \
  INSTANTIATE_IMPL(Out, uint64_t);      \
  INSTANTIATE_IMPL(Out, int8_t);        \
  INSTANTIATE_IMPL(Out, int16_t);       \
  INSTANTIATE_IMPL(Out, int32_t);       \
  INSTANTIATE_IMPL(Out, int64_t);       \
  INSTANTIATE_IMPL(Out, float);         \
  INSTANTIATE_IMPL(Out, double);        \
  INSTANTIATE_IMPL(Out, dali::float16); \
  INSTANTIATE_IMPL(Out, DALIDataType);  \
  INSTANTIATE_IMPL(Out, DALIImageType); \
  INSTANTIATE_IMPL(Out, DALIInterpType);

INSTANTIATE_FOREACH_INTYPE(bool);
INSTANTIATE_FOREACH_INTYPE(uint8_t);
INSTANTIATE_FOREACH_INTYPE(uint16_t);
INSTANTIATE_FOREACH_INTYPE(uint32_t);
INSTANTIATE_FOREACH_INTYPE(uint64_t);
INSTANTIATE_FOREACH_INTYPE(int8_t);
INSTANTIATE_FOREACH_INTYPE(int16_t);
INSTANTIATE_FOREACH_INTYPE(int32_t);
INSTANTIATE_FOREACH_INTYPE(int64_t);
INSTANTIATE_FOREACH_INTYPE(float);
INSTANTIATE_FOREACH_INTYPE(double);
INSTANTIATE_FOREACH_INTYPE(dali::float16);
INSTANTIATE_FOREACH_INTYPE(DALIDataType);
INSTANTIATE_FOREACH_INTYPE(DALIImageType);
INSTANTIATE_FOREACH_INTYPE(DALIInterpType);

}  // namespace cast
}  // namespace kernels
}  // namespace dali
