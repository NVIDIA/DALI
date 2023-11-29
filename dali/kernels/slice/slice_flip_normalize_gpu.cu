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

#include <cuda_runtime.h>
#include <tuple>
#include "dali/core/float16.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/slice/slice_flip_normalize_gpu.h"
#include "dali/kernels/slice/slice_flip_normalize_gpu_impl.cuh"

namespace dali {
namespace kernels {

namespace slice_flip_normalize {

template <typename Out, typename In, int spatial_ndim, int channel_dim>
int SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>::GetNumChannels(
    const TensorListShape<ndim> &sh) {
  if (sh.num_samples() == 0)
    return 0;
  if (channel_dim < 0)
    return 1;
  const auto first_sh = sh.tensor_shape_span(0);
  if (channel_dim >= first_sh.size())
    throw std::invalid_argument("Not enough dimensions in the shape");
  int nchannels = first_sh[channel_dim];
  for (int i = 1; i < sh.num_samples(); i++) {
    if (nchannels != sh.tensor_shape_span(i)[channel_dim]) {
      throw std::invalid_argument("All samples should have the same number of channels");
    }
  }
  return nchannels;
}

template <typename Out, typename In, int spatial_ndim, int channel_dim>
int SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>::GetOutNumChannels(
    const TensorListShape<ndim> &sh, const Args &args) {
  if (sh.num_samples() != static_cast<int>(args.sample_args.size())) {
    std::invalid_argument(
        "Number of samples in the arguments should match the number of samples in the shape");
  }
  int nchannels = GetNumChannels(sh);
  int out_nchannels = std::max(nchannels, static_cast<int>(args.sample_args[0].fill_values.size()));
  for (int i = 1; i < sh.num_samples(); i++) {
    if (args.sample_args[i].fill_values.size() != args.sample_args[0].fill_values.size())
      throw std::invalid_argument(
          "All sample arguments should have the same number of fill values");
  }
  return out_nchannels;
}

template <typename Out, typename In, int spatial_ndim, int channel_dim>
KernelRequirements SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>::Setup(
    KernelContext &ctx, const TensorListShape<ndim> &sh, const Args &args) {
  (void) ctx;
  int nsamples = sh.num_samples();
  if (nsamples != static_cast<int>(args.sample_args.size()))
    throw std::invalid_argument("Invalid number of samples in kernel args");
  out_shape_ = TensorListShape<ndim>(nsamples, ndim);
  out_shape_orig_ = TensorListShape<ndim>(nsamples, ndim);
  perm_ = args.perm;
  if (perm_ == ivec<ndim>{})
    std::iota(perm_.begin(), perm_.end(), 0);
  inv_perm_ = inverse_permutation(perm_);

  nchannels_ = GetNumChannels(sh);
  out_nchannels_ = GetOutNumChannels(sh, args);

  for (int i = 0; i < nsamples; i++) {
    auto out_sh = sh[i];
    const auto &sample_args = args.sample_args[i];
    Fill(out_sh, sample_args.roi.extent());
    out_sh[channel_dim] = out_nchannels_;
    out_shape_orig_.set_tensor_shape(i, out_sh);
    out_sh = permute(out_sh, perm_);
    out_shape_.set_tensor_shape(i, out_sh);
  }
  KernelRequirements req;
  req.output_shapes = {out_shape_};
  return req;
}

template <typename Out, typename In, int spatial_ndim, int channel_dim>
std::tuple<float *, float *, Out *>
SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>::SetupParams(KernelContext &ctx,
                                                                       const Args &args) {
  int num_samples = args.sample_args.size();
  float *norm_add_cpu = ctx.scratchpad->AllocatePinned<float>(num_samples * nchannels_);
  float *norm_mul_cpu = ctx.scratchpad->AllocatePinned<float>(num_samples * nchannels_);
  Out *fill_values_cpu = ctx.scratchpad->AllocatePinned<Out>(num_samples * out_nchannels_);
  for (int i = 0; i < num_samples; i++) {
    const auto &a = args.sample_args[i];
    auto *norm_add_data = norm_add_cpu + i * nchannels_;
    auto *norm_mul_data = norm_mul_cpu + i * nchannels_;
    int mean_sz = a.mean.size();
    assert(mean_sz == a.inv_stddev.size());
    int c = 0;
    for (; c < mean_sz; c++) {
      norm_add_data[c] = -a.mean[c] * a.inv_stddev[c];
      norm_mul_data[c] = a.inv_stddev[c];
    }
    for (; c < nchannels_; c++) {
      norm_add_data[c] = 0.0f;
      norm_mul_data[c] = 1.0f;
    }
    auto *fill_values_data = fill_values_cpu + i * out_nchannels_;
    int fill_values_sz = a.fill_values.size();
    c = 0;
    for (; c < fill_values_sz; c++)
      fill_values_data[c] = ConvertSat<Out>(a.fill_values[c]);
    for (; c < out_nchannels_; c++)
      fill_values_data[c] = ConvertSat<Out>(0.0f);
  }

  return ctx.scratchpad->ToContiguousGPU(
      ctx.gpu.stream, make_span(norm_add_cpu, num_samples * nchannels_),
      make_span(norm_mul_cpu, num_samples * nchannels_),
      make_span(fill_values_cpu, num_samples * out_nchannels_));
}


template <typename Out, typename In, int spatial_ndim, int channel_dim>
void SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>::Run(
    KernelContext &ctx, const OutListGPU<Out, ndim> &out, const InListGPU<In, ndim> &in,
    const Args &args) {
  using Tile = kernels::BlockDesc<spatial_ndim>;
  using Sample = SampleDesc<Out, In, spatial_ndim>;
  int nsamples = in.num_samples();

  Sample *samples_cpu = ctx.scratchpad->AllocatePinned<Sample>(nsamples);
  auto [norm_add_gpu, norm_mul_gpu, fill_values_gpu] = SetupParams(ctx, args);

  bool need_pad = out_nchannels_ != nchannels_;
  for (int i = 0; i < nsamples; i++) {
    auto &sample_args = args.sample_args[i];
    auto &sample = samples_cpu[i];

    auto in_sh = in.shape[i];
    auto in_strides_sh = GetStrides(in_sh);
    vec<spatial_ndim, int> in_size;
    vec<spatial_ndim, int64_t> in_strides;
    const vec<spatial_ndim, int> zero{0};
    Fill(in_size, in_sh);
    Fill(in_strides, in_strides_sh);

    auto out_data = out.tensor_data(i);
    auto out_size = static_cast<vec<spatial_ndim, int>>(sample_args.roi.extent());
    auto out_strides_sh = GetStrides(out_shape_[i]);
    out_strides_sh = permute(out_strides_sh, inv_perm_);
    vec<spatial_ndim, int64_t> out_strides;
    Fill(out_strides, out_strides_sh);

    auto in_data = in.tensor_data(i);
    auto roi = sample_args.roi;
    Roi<2> bounds = {zero, out_size};
    vec<spatial_ndim, int> pad_begin = max(-roi.lo, zero);
    vec<spatial_ndim, int> pad_end = max(roi.hi - in_size, zero);
    need_pad |= pad_begin != zero || pad_end != zero;

    // We do flipping by adjusting the input data pointer to point to the last
    // element in the dimension, changing the sign of the stride in that dimension
    // and adjusting the ROI accordingly, so that we can index the surface object
    // as in the non-flipped case.
    for (int d = 0; d < spatial_ndim; d++) {
      if (sample_args.flip[d]) {
        in_data += (roi.hi[d] - 1) * in_strides[d];
        in_strides[d] = -in_strides[d];
        bounds.lo[d] = pad_end[d];
        bounds.hi[d] = out_size[d] - pad_begin[d];
      } else {
        in_data += roi.lo[d] * in_strides[d];
        bounds.lo[d] = pad_begin[d];
        bounds.hi[d] = out_size[d] - pad_end[d];
      }
    }

    sample.in = {in_data, in_size, nchannels_, in_strides, in_strides_sh[channel_dim]};
    sample.out = {out_data, out_size, out_nchannels_, out_strides, out_strides_sh[channel_dim]};
    sample.bounds = bounds;
    sample.norm_add = norm_add_gpu + i * nchannels_;
    sample.norm_mul = norm_mul_gpu + i * nchannels_;
    sample.fill_values = fill_values_gpu + i * out_nchannels_;
  }
  int max_blocks = 0;
  if (need_pad) {
    max_blocks = MaxThreadsPerBlock(SliceNormalizeKernel_2D<Out, In>);
  } else {
    max_blocks = MaxThreadsPerBlock(SliceNormalizeKernel_2D_NoPad<Out, In>);
  }
  block_setup_.SetDefaultBlockSize({64, 64});
  block_setup_.SetBlockDim(dim3(32, max_blocks / 32));
  block_setup_.SetupBlocks(out_shape_orig_, true);
  auto tiles_cpu = block_setup_.Blocks();
  auto grid_dim = block_setup_.GridDim();
  auto block_dim = block_setup_.BlockDim();

  Sample *samples_gpu = nullptr;
  Tile *tiles_gpu = nullptr;
  std::tie(samples_gpu, tiles_gpu) =
      ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, make_span(samples_cpu, nsamples), tiles_cpu);

  if (spatial_ndim == 2) {
    if (need_pad) {
      SliceNormalizeKernel_2D<Out, In>
          <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(samples_gpu, tiles_gpu);
    } else {
      SliceNormalizeKernel_2D_NoPad<Out, In>
          <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(samples_gpu, tiles_gpu);
    }
  } else {
    assert(false);  // TODO(janton): implement
  }
  CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_IMPL(Out, In, SpatialDims, ChannelDim)                                    \
  template class DLL_PUBLIC SliceFlipNormalizeGPU<Out, In, SpatialDims, ChannelDim>;

#define INSTANTIATE_FOREACH_INTYPE(Out, spatial_ndim, channel_dim)  \
  INSTANTIATE_IMPL(Out, uint8_t, spatial_ndim, channel_dim);  \
  INSTANTIATE_IMPL(Out, int16_t, spatial_ndim, channel_dim);  \
  INSTANTIATE_IMPL(Out, uint16_t, spatial_ndim, channel_dim); \
  INSTANTIATE_IMPL(Out, int32_t, spatial_ndim, channel_dim);  \
  INSTANTIATE_IMPL(Out, float, spatial_ndim, channel_dim);    \
  INSTANTIATE_IMPL(Out, dali::float16, spatial_ndim, channel_dim);

#define INSTANTIATE_FOREACH_OUTTYPE(spatial_ndim, channel_dim)    \
  INSTANTIATE_FOREACH_INTYPE(float, spatial_ndim, channel_dim);   \
  INSTANTIATE_FOREACH_INTYPE(dali::float16, spatial_ndim, channel_dim); \
  INSTANTIATE_FOREACH_INTYPE(uint8_t, spatial_ndim, channel_dim); \
  INSTANTIATE_FOREACH_INTYPE(int8_t, spatial_ndim, channel_dim);

INSTANTIATE_FOREACH_OUTTYPE(2, 0);
INSTANTIATE_FOREACH_OUTTYPE(2, 2);

}  // namespace slice_flip_normalize

}  // namespace kernels
}  // namespace dali
