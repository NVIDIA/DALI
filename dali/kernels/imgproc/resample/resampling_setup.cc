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
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {
namespace resampling {

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

template <int spatial_ndim>
void SeparableResamplingSetup<spatial_ndim>::SetFilters(
    SampleDesc &desc,
    const ResamplingParamsND<spatial_ndim> &params) {
  for (int dim = 0; dim < spatial_ndim; dim++) {
    int axis = spatial_ndim - 1 - dim;
    float in_size;
    if (params[dim].roi.use_roi) {
      in_size = std::abs(params[dim].roi.end - params[dim].roi.start);
    } else {
      in_size = desc.in_shape()[axis];
    }

    auto fdesc = desc.out_shape()[axis] < in_size ? params[dim].min_filter
                                                  : params[dim].mag_filter;
    if (fdesc.radius == 0)
      fdesc.radius = DefaultFilterRadius(fdesc.type, in_size, desc.out_shape()[axis]);
    desc.filter_type[axis] = fdesc.type;
    auto &filter = desc.filter[axis];
    filter = GetResamplingFilter(filters.get(), fdesc);
  }
}

template <int spatial_ndim>
typename SeparableResamplingSetup<spatial_ndim>::ROI
SeparableResamplingSetup<spatial_ndim>::ComputeScaleAndROI(
    SampleDesc &desc, const ResamplingParamsND<spatial_ndim> &params) {
  ROI roi;

  for (int dim = 0; dim < spatial_ndim; dim++) {
    int axis = spatial_ndim - 1 - dim;
    float roi_start, roi_end;
    if (params[dim].roi.use_roi) {
      roi_start = params[dim].roi.start;
      roi_end = params[dim].roi.end;
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

template <>
void SeparableResamplingSetup<2>::SetupSample(
    SampleDesc &desc,
    const TensorShape<tensor_ndim> &in_shape,
    const ResamplingParams2D &params) {

  int H = in_shape[0];
  int W = in_shape[1];
  int C = in_shape[2];
  int out_H = params[0].output_size;
  int out_W = params[1].output_size;

  if (out_H == KeepOriginalSize) out_H = H;
  if (out_W == KeepOriginalSize) out_W = W;

  desc.in_shape() = { W, H };
  desc.out_shape() = { out_W, out_H };
  SetFilters(desc, params);
  ROI roi = ComputeScaleAndROI(desc, params);

  int64_t size_vert = volume({roi.extent().x, out_H});
  int64_t size_horz = volume({out_W, roi.extent().y});
  ivec2 filter_support = {
    std::max(1, desc.filter[0].support()),
    std::max(1, desc.filter[1].support())
  };

  int64_t out_area = volume(desc.out_shape());
  int64_t compute_vh = size_vert * filter_support.y + out_area * filter_support.x;
  int64_t compute_hv = size_horz * filter_support.x + out_area * filter_support.y;

  // ...maybe fine tune the size/compute weights?
  const float size_weight = 3;
  float cost_vert = size_weight*size_vert + compute_vh;
  float cost_horz = size_weight*size_horz + compute_hv;

  int tmp_H, tmp_W;
  if (cost_vert < cost_horz) {
    desc.order = VertHorz();
    tmp_H = out_H;
    tmp_W = roi.extent().x;
    desc.block_count[0] = div_ceil(out_H, block_size.y);
    desc.block_count[1] = div_ceil(out_W, block_size.x);
  } else {
    desc.order = HorzVert();
    tmp_H = roi.extent().y;
    tmp_W = out_W;
    desc.block_count[0] = div_ceil(out_W, block_size.x);
    desc.block_count[1] = div_ceil(out_H, block_size.y);
  }
  desc.tmp_shape(0) = { tmp_W, tmp_H };

  for (int stage = 0; stage < 3; stage++) {
    desc.strides[stage][0] = desc.shapes[stage].x * C;
    desc.offsets[stage] = 0;
  }
  desc.channels = C;

  if (desc.order == VertHorz()) {
    desc.origin.x -= roi.lo.x;
    desc.in_offset() += roi.lo.x * desc.channels;
    desc.in_shape().x = roi.extent().x;
  } else {
    desc.origin.y -= roi.lo.y;
    desc.in_offset() += roi.lo.y * desc.strides[0][0];
    desc.in_shape().y = roi.extent().y;
  }
}

template <>
void BatchResamplingSetup<2>::SetupBatch(
    const TensorListShape<3> &in, const Params &params) {
  if (!filters)
    Initialize();

  int N = in.num_samples();
  assert(params.size() == static_cast<span_extent_t>(N));

  sample_descs.resize(N);
  for (auto &shape : intermediate_shapes)
    shape.resize(N);

  output_shape.resize(N);
  for (auto &size : intermediate_sizes)
    size = 0;

  total_blocks = { 0, 0 };

  for (int i = 0; i < N; i++) {
    SampleDesc &desc = sample_descs[i];
    auto ts_in = in.tensor_shape(i);
    SetupSample(desc, ts_in, params[i]);

    for (int t = 0; t < num_tmp_buffers; t++) {
      TensorShape<tensor_ndim> ts_tmp = shape_cat(vec2shape(desc.tmp_shape(t)), desc.channels);
      intermediate_shapes[t].set_tensor_shape(i, ts_tmp);
      intermediate_sizes[t] += volume(ts_tmp);
    }

    auto ts_out = output_shape.tensor_shape_span(i);
    static_assert(channel_dim == spatial_ndim, "Shape calculation requires channel-last layout");
    auto sample_shape = shape_cat(vec2shape(desc.out_shape()), desc.channels);
    output_shape.set_tensor_shape(i, sample_shape);

    for (int d = 0; d < spatial_ndim; d++)
      total_blocks[d] += desc.block_count[d];
  }
}

template <int spatial_ndim>
void BatchResamplingSetup<spatial_ndim>::InitializeSampleLookup(
    const OutTensorCPU<SampleBlockInfo, 1> &sample_lookup) {
  int blocks_in_all_passes = 0;
  for (int i = 0; i < spatial_ndim; i++)
    blocks_in_all_passes += total_blocks[i];

  assert(sample_lookup.shape[0] >= blocks_in_all_passes);
  (void)blocks_in_all_passes;  // for non-debug builds

  int block = 0;
  int N = sample_descs.size();
  for (int i = 0; i < N; i++) {
    for (int b = 0; b < sample_descs[i].block_count[0]; b++) {
      sample_lookup.data[block++] = { i, b };
    }
  }
  assert(block == total_blocks[0]);
  for (int i = 0; i < N; i++) {
    for (int b = 0; b < sample_descs[i].block_count[1]; b++) {
      sample_lookup.data[block++] = { i, b };
    }
  }
  assert(block == total_blocks[0] + total_blocks[1]);
}

template class BatchResamplingSetup<2>;
template class BatchResamplingSetup<3>;

}  // namespace resampling
}  // namespace kernels
}  // namespace dali
