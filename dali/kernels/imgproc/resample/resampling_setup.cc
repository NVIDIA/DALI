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

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/resampling_setup.h"

namespace dali {
namespace kernels {

inline ResamplingFilter GetResamplingFilter(
    const ResamplingFilters *filters, const FilterDesc &params) noexcept {
  switch (params.type) {
    case ResamplingFilterType::Gaussian:
      return filters->Gaussian(params.radius*0.5f);
    case ResamplingFilterType::Triangular:
      return filters->Triangular(params.radius);
    case ResamplingFilterType::Lanczos3:
      return filters->Lanczos3();
  default:
    return { nullptr, 0, 0, 0 };
  }
}


void SeparableResamplingSetup::SetupComputation(
    const TensorListShape<3> &in, const Params &params) {
  if (!filters)
    Initialize(0);

  int N = in.num_samples();
  assert(params.size() == N);

  sample_descs.resize(N);
  intermediate_shape.resize(N);
  output_shape.resize(N);
  intermediate_size = 0;

  ptrdiff_t in_offset = 0, out_offset = 0, tmp_offset = 0;

  total_blocks = { 0, 0 };

  for (int i = 0; i < N; i++) {
    SampleDesc &desc = sample_descs[i];
    auto ts_in = in.tensor_shape_span(i);
    int H = ts_in[0];
    int W = ts_in[1];
    int C = ts_in[2];
    int out_H = params[i][0].output_size;
    int out_W = params[i][1].output_size;

    if (out_H == KeepOriginalSize) out_H = H;
    if (out_W == KeepOriginalSize) out_W = W;

    auto ts_out = output_shape.tensor_shape_span(i);
    ts_out[0] = out_H;
    ts_out[1] = out_W;
    ts_out[2] = C;

    desc.shapes[0] = {{ H, W }};
    desc.shapes[2] = {{ out_H, out_W }};

    int size_vert = W * out_H;
    int size_horz = H * out_W;
    int filter_support[2];
    for (int axis = 0; axis < 2; axis++) {
      auto fdesc = ts_out[axis] < ts_in[axis] ? params[i][axis].min_filter
                                              : params[i][axis].mag_filter;
      if (fdesc.radius == 0)
        fdesc.radius = DefaultFilterRadius(fdesc.type, ts_in[axis], ts_out[axis]);
      desc.filter_type[axis] = fdesc.type;
      auto &filter = desc.filter[axis];
      filter = GetResamplingFilter(filters.get(), fdesc);
      filter_support[axis] = filter.num_coeffs > 0 ? filter.support() : 1;
    }
    int compute_vert = size_vert * filter_support[0];
    int compute_horz = size_horz * filter_support[1];

    // ...maybe fine tune the size/compute weights?
    const float size_weight = 3;
    float cost_vert = size_weight*size_vert + compute_vert;
    float cost_horz = size_weight*size_horz + compute_horz;

    const int vblock = 24;
    const int hblock = 32;

    auto ts_tmp = intermediate_shape.tensor_shape_span(i);
    if (cost_vert < cost_horz) {
      sample_descs[i].order = VertHorz;
      ts_tmp[0] = out_H;
      ts_tmp[1] = W;
      desc.block_count.pass[0] = (out_H + vblock - 1) / vblock;
      desc.block_count.pass[1] = (out_W + hblock - 1) / hblock;
    } else {
      sample_descs[i].order = HorzVert;
      ts_tmp[0] = H;
      ts_tmp[1] = out_W;
      desc.block_count.pass[0] = (out_W + hblock - 1) / hblock;
      desc.block_count.pass[1] = (out_H + vblock - 1) / vblock;
    }
    ts_tmp[2] = C;
    desc.shapes[1] = {{ static_cast<int>(ts_tmp[0]), static_cast<int>(ts_tmp[1]) }};

    for (int stage = 0; stage < 3; stage++)
      desc.strides[stage] = desc.shapes[stage][1] * C;
    desc.channels = C;

    desc.offsets = {{ in_offset, tmp_offset, out_offset }};

    in_offset  += volume(ts_in);
    tmp_offset += volume(ts_tmp);
    out_offset += volume(ts_out);

    total_blocks.pass[0]  += desc.block_count.pass[0];
    total_blocks.pass[1] += desc.block_count.pass[1];
  }
  intermediate_size = tmp_offset;
}

void SeparableResamplingSetup::InitializeSampleLookup(
    const OutTensorCPU<SampleBlockInfo, 1> &sample_lookup) {
  assert(sample_lookup.shape[0] >= total_blocks.pass[0] + total_blocks.pass[1]);
  int block = 0;
  int N = sample_descs.size();
  for (int i = 0; i < N; i++) {
    for (int b = 0; b < sample_descs[i].block_count.pass[0]; b++) {
      sample_lookup.data[block++] = { i, b };
    }
  }
  assert(block == total_blocks.pass[0]);
  for (int i = 0; i < N; i++) {
    for (int b = 0; b < sample_descs[i].block_count.pass[1]; b++) {
      sample_lookup.data[block++] = { i, b };
    }
  }
  assert(block == total_blocks.pass[0] + total_blocks.pass[1]);
}

}  // namespace kernels
}  // namespace dali
