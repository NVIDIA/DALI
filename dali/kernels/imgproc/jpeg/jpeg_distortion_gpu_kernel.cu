// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_kernel.h"
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_impl.cuh"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace jpeg {

KernelRequirements JpegDistortionBaseGPU::Setup(KernelContext &ctx,
                                                const TensorListShape<3> &in_shape,
                                                bool horz_subsample, bool vert_subsample) {
  horz_subsample_ = horz_subsample;
  vert_subsample_ = vert_subsample;
  KernelRequirements req;
  int nsamples = in_shape.num_samples();

  chroma_shape_.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    auto chroma_sh = chroma_shape_.tensor_shape_span(i);
    auto sh = in_shape.tensor_shape_span(i);
    // used to generate logical blocks (one thread per chroma pixel)
    chroma_sh[0] = div_ceil(sh[0], 1 + vert_subsample_);
    chroma_sh[1] = div_ceil(sh[1], 1 + horz_subsample_);
  }

  block_setup_.SetBlockDim(dim3(32, 16, 1));
  int xblock = 64 * (2 - horz_subsample_);
  int yblock = 128;
  block_setup_.SetDefaultBlockSize({xblock, yblock});
  block_setup_.SetupBlocks(chroma_shape_, true);
  int nblocks = block_setup_.Blocks().size();
  req.output_shapes = {in_shape};
  return req;
}

void JpegDistortionBaseGPU::SetupSampleDescs(const OutListGPU<uint8_t, 3> &out,
                                             const InListGPU<uint8_t, 3> &in,
                                             span<const int> quality) {
  const auto &in_shape = in.shape;
  int nsamples = in_shape.num_samples();
  sample_descs_.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    auto &sample_desc = sample_descs_[i];
    auto in_sh = in_shape.tensor_shape_span(i);
    auto width = in_sh[1];
    auto height = in_sh[0];
    sample_desc.in = in[i].data;
    sample_desc.out = out[i].data;
    sample_desc.size.x = width;
    sample_desc.size.y = height;
    sample_desc.strides.x = 3;
    sample_desc.strides.y = width * 3;
    int q;
    if (quality.empty()) {
      q = 95;
    } else if (quality.size() == 1) {
      q = quality[0];
    } else {
      q = quality[i];
    }
    sample_desc.luma_Q_table = GetLumaQuantizationTable(q);
    sample_desc.chroma_Q_table = GetChromaQuantizationTable(q);
  }
}

void JpegCompressionDistortionGPU::Run(KernelContext &ctx, const OutListGPU<uint8_t, 3> &out,
                                       const InListGPU<uint8_t, 3> &in, span<const int> quality) {
  const auto &in_shape = in.shape;
  int nsamples = in_shape.num_samples();
  if (quality.size() > 1 && quality.size() != nsamples) {
    throw std::invalid_argument(
      make_string("Unexpected number of elements in ``quality`` argument. "
                  "The argument could contain a single value (used for the whole batch), "
                  "one value per sample, or no values (a default is used). Received ",
                  quality.size(), " values but batch size is ", nsamples, "."));
  }
  SetupSampleDescs(out, in, quality);
  SampleDesc *samples_gpu;
  BlockDesc *blocks_gpu;
  std::tie(samples_gpu, blocks_gpu) = ctx.scratchpad->ToContiguousGPU(
      ctx.gpu.stream, make_cspan(sample_descs_), block_setup_.Blocks());
  dim3 grid_dim = block_setup_.GridDim();
  dim3 block_dim = block_setup_.BlockDim();
  BOOL_SWITCH(horz_subsample_, HorzSubsample, (
    BOOL_SWITCH(vert_subsample_, VertSubsample, (
      JpegCompressionDistortion<HorzSubsample, VertSubsample>
          <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(samples_gpu, blocks_gpu);
    ));  // NOLINT
  ));  // NOLINT
  CUDA_CALL(cudaGetLastError());
}

void ChromaSubsampleDistortionGPU::Run(KernelContext &ctx, const OutListGPU<uint8_t, 3> &out,
                                       const InListGPU<uint8_t, 3> &in) {
  SetupSampleDescs(out, in);
  SampleDesc *samples_gpu;
  BlockDesc *blocks_gpu;
  std::tie(samples_gpu, blocks_gpu) = ctx.scratchpad->ToContiguousGPU(
      ctx.gpu.stream, make_cspan(sample_descs_), block_setup_.Blocks());
  dim3 grid_dim = block_setup_.GridDim();
  dim3 block_dim = block_setup_.BlockDim();
  BOOL_SWITCH(horz_subsample_, HorzSubsample, (
    BOOL_SWITCH(vert_subsample_, VertSubsample, (
      ChromaSubsampleDistortion<HorzSubsample, VertSubsample>
          <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(samples_gpu, blocks_gpu);
    ));  // NOLINT
  ));  // NOLINT
  CUDA_CALL(cudaGetLastError());
}

}  // namespace jpeg
}  // namespace kernels
}  // namespace dali
