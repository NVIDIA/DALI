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

ResamplingFilter GetResamplingFilter(const ResamplingFilters *filters, const FilterDesc &params) {
  switch (params.type) {
    case ResamplingFilterType::Linear:
      return filters->Triangular(1);
    case ResamplingFilterType::Triangular:
      return filters->Triangular(params.radius);
    case ResamplingFilterType::Gaussian:
      return filters->Gaussian(params.radius*0.5f/M_SQRT2);
    case ResamplingFilterType::Cubic:
      return filters->Cubic();
    case ResamplingFilterType::Lanczos3:
      return filters->Lanczos3();
    default:
      return { nullptr, 0, 0, 0 };
  }
}

void SeparableResamplingSetup::SetFilters(SampleDesc &desc, const ResamplingParams2D &params) {
  for (int axis = 0; axis < 2; axis++) {
    float in_size;
    if (params[axis].roi.use_roi) {
      in_size = std::abs(params[axis].roi.end - params[axis].roi.start);
    } else {
      in_size = desc.in_shape()[axis];
    }

    auto fdesc = desc.out_shape()[axis] < in_size ? params[axis].min_filter
                                                  : params[axis].mag_filter;
    if (fdesc.radius == 0)
      fdesc.radius = DefaultFilterRadius(fdesc.type, in_size, desc.out_shape()[axis]);
    desc.filter_type[axis] = fdesc.type;
    auto &filter = desc.filter[axis];
    filter = GetResamplingFilter(filters.get(), fdesc);
  }
}

SeparableResamplingSetup::ROI SeparableResamplingSetup::ComputeScaleAndROI(
    SampleDesc &desc, const ResamplingParams2D &params) {
  ROI roi;

  for (int axis = 0; axis < 2; axis++) {
    float roi_start, roi_end;
    if (params[axis].roi.use_roi) {
      roi_start = params[axis].roi.start;
      roi_end = params[axis].roi.end;
    } else {
      roi_start = 0;
      roi_end = desc.in_shape()[axis];
    }
    desc.origin[axis] = roi_start;
    desc.scale[axis] = (roi_end-roi_start) / desc.out_shape()[axis];

    auto &filter = desc.filter[axis];

    int support = filter.num_coeffs ? filter.support() : 1;

    float lo, hi;
    if (roi_start <= roi_end) {
      lo = roi_start - filter.anchor;
      hi = roi_end - filter.anchor + support;
    } else {  // flipped
      lo = roi_end - filter.anchor;
      hi = roi_start - filter.anchor + support;
    }
    roi.lo[axis] = std::max<int>(0, std::min<int>(desc.in_shape()[axis], std::floor(lo)));
    roi.hi[axis] = std::max<int>(0, std::min<int>(desc.in_shape()[axis], std::floor(hi)));
  }

  return roi;
}

void SeparableResamplingSetup::SetupSample(
    SampleDesc &desc,
    const TensorShape<3> &in_shape,
    const ResamplingParams2D &params) {
  int H = in_shape[0];
  int W = in_shape[1];
  int C = in_shape[2];
  int out_H = params[0].output_size;
  int out_W = params[1].output_size;

  if (out_H == KeepOriginalSize) out_H = H;
  if (out_W == KeepOriginalSize) out_W = W;

  desc.in_shape() = {{ H, W }};
  desc.out_shape() = {{ out_H, out_W }};
  SetFilters(desc, params);
  ROI roi = ComputeScaleAndROI(desc, params);

  int size_vert = roi.size(1) * out_H;
  int size_horz = roi.size(0) * out_W;
  int filter_support[2] = {
    std::max(1, desc.filter[0].support()),
    std::max(1, desc.filter[1].support())
  };

  int compute_vh = size_vert * filter_support[0] + out_W * out_H * filter_support[1];
  int compute_hv = size_horz * filter_support[1] + out_W * out_H * filter_support[0];

  // ...maybe fine tune the size/compute weights?
  const float size_weight = 3;
  float cost_vert = size_weight*size_vert + compute_vh;
  float cost_horz = size_weight*size_horz + compute_hv;

  int tmp_H, tmp_W;
  if (cost_vert < cost_horz) {
    desc.order = VertHorz;
    tmp_H = out_H;
    tmp_W = roi.size(1);
    desc.block_count.pass[0] = (out_H + block_size.y - 1) / block_size.y;
    desc.block_count.pass[1] = (out_W + block_size.x - 1) / block_size.x;
  } else {
    desc.order = HorzVert;
    tmp_H = roi.size(0);
    tmp_W = out_W;
    desc.block_count.pass[0] = (out_W + block_size.x - 1) / block_size.x;
    desc.block_count.pass[1] = (out_H + block_size.y - 1) / block_size.y;
  }
  desc.tmp_shape() = {{ tmp_H, tmp_W }};

  for (int stage = 0; stage < 3; stage++) {
    desc.strides[stage] = desc.shapes[stage][1] * C;
    desc.offsets[stage] = 0;
  }
  desc.channels = C;

  if (desc.order == VertHorz) {
    desc.origin[1] -= roi.lo[1];
    desc.in_offset() += roi.lo[1] * desc.channels;
    desc.in_shape()[1] = roi.size(1);
  } else {
    desc.origin[0] -= roi.lo[0];
    desc.in_offset() += roi.lo[0] * desc.in_stride();
    desc.in_shape()[0] = roi.size(0);
  }
}

void BatchResamplingSetup::SetupBatch(
    const TensorListShape<3> &in, const Params &params) {
  if (!filters)
    Initialize();

  int N = in.num_samples();
  assert(params.size() == static_cast<size_t>(N));

  sample_descs.resize(N);
  intermediate_shape.resize(N);
  output_shape.resize(N);
  intermediate_size = 0;

  ptrdiff_t in_offset = 0, out_offset = 0, tmp_offset = 0;

  total_blocks = { 0, 0 };

  for (int i = 0; i < N; i++) {
    SampleDesc &desc = sample_descs[i];
    auto ts_in = in.tensor_shape(i);
    SetupSample(desc, ts_in, params[i]);

    auto ts_tmp = intermediate_shape.tensor_shape_span(i);
    ts_tmp[0] = desc.tmp_shape()[0];
    ts_tmp[1] = desc.tmp_shape()[1];
    ts_tmp[2] = desc.channels;

    auto ts_out = output_shape.tensor_shape_span(i);
    ts_out[0] = desc.out_shape()[0];
    ts_out[1] = desc.out_shape()[1];
    ts_out[2] = desc.channels;

    desc.in_offset() += in_offset;
    desc.tmp_offset() += tmp_offset;
    desc.out_offset() += out_offset;

    in_offset  += volume(ts_in);
    tmp_offset += volume(ts_tmp);
    out_offset += volume(ts_out);

    total_blocks.pass[0] += desc.block_count.pass[0];
    total_blocks.pass[1] += desc.block_count.pass[1];
  }
  intermediate_size = tmp_offset;
}

void BatchResamplingSetup::InitializeSampleLookup(
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
