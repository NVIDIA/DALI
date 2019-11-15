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
      return filters->Lanczos3(params.radius);
    default:
      return { nullptr, 0, 0, 1 };
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

ProcessingOrder<2> GetProcessingOrder(ivec2 in_size, ivec2 out_size, ivec2 filter_support) {
  int64_t size_vert = volume({in_size.x, out_size.y});
  int64_t size_horz = volume({out_size.x, in_size.y});
  int64_t out_area = volume(out_size);
  int64_t compute_vh = size_vert * filter_support.y + out_area * filter_support.x;
  int64_t compute_hv = size_horz * filter_support.x + out_area * filter_support.y;

  // ...maybe fine tune the size/compute weights?
  const float size_weight = 3;
  float cost_vert = size_weight*size_vert + compute_vh;
  float cost_horz = size_weight*size_horz + compute_hv;
  return cost_vert < cost_horz ? VertHorz() : HorzVert();
}

ProcessingOrder<3> GetProcessingOrder(ivec3 in_size, ivec3 out_size, ivec3 filter_support) {
  for (int a0 = 0; a0 < 3; a0++) {
    for (int a1 = 0; a1 < 3; a1++) {
      if (a1 == a0)
        continue;
      for (int a2 = 0; a2 < 3; a2++)
        if (a2 == a0 || a2 == a1) {
          continue;
        }


    }
  }
  return { 0, 1, 2 };

  /*int64_t size_vert = volume({roi.extent().x, out_H});
  int64_t size_horz = volume({out_W, roi.extent().y});
  int64_t out_area = volume(desc.out_shape());
  int64_t compute_vh = size_vert * filter_support.y + out_area * filter_support.x;
  int64_t compute_hv = size_horz * filter_support.x + out_area * filter_support.y;

  // ...maybe fine tune the size/compute weights?
  const float size_weight = 3;
  float cost_vert = size_weight*size_vert + compute_vh;
  float cost_horz = size_weight*size_horz + compute_hv;
  return cost_vert < cost_horz ? VertHorz() : HorzVert();*/
}

template <>
void SeparableResamplingSetup<2>::ComputeBlockLayout(SampleDesc &sample) {
  for (int pass = 0; pass < 2; pass++) {
    int axis = sample.order[pass];
    sample.block_count[pass] = div_ceil(sample.out_shape()[axis], block_dim[axis]);
  }
}

template <>
void SeparableResamplingSetup<3>::ComputeBlockLayout(SampleDesc &sample) {
  for (int pass = 0; pass < 3; pass++) {
    int axis = sample.order[pass];
    if (axis == 2) {
      sample.block_count[pass] = div_ceil(sample.out_shape()[axis], block_dim[axis]);
    } else {
      sample.block_count[pass] = div_ceil(sample.out_shape()[axis], block_dim[axis]) *
                                 div_ceil(sample.out_shape().z, block_dim.z);
    }
  }
}

template <int spatial_ndim>
void SeparableResamplingSetup<spatial_ndim>::SetupSample(
    SampleDesc &desc,
    const TensorShape<tensor_ndim> &in_shape,
    const ResamplingParamsND<spatial_ndim> &params) {
  int channels = in_shape[channel_dim];
  auto in_size = shape2vec(skip_dim<channel_dim>(in_shape));
  ivec<spatial_ndim> out_size;
  for (int i = 0, d = spatial_ndim - 1; i < spatial_ndim; i++, d--) {
    out_size[i] = params[d].output_size;
    if (out_size[i] == KeepOriginalSize)
      out_size[i] = in_size[i];
  }

  desc.in_shape() = in_size;
  desc.out_shape() = out_size;
  desc.channels = channels;

  SetFilters(desc, params);
  ROI roi = ComputeScaleAndROI(desc, params);

  ivec<spatial_ndim> filter_support;
  for (int i = 0, d = spatial_ndim - 1; i < spatial_ndim; i++, d--) {
    int support = desc.filter[d].support();
    // NN filter has support -1, so we need the max() below
    filter_support[i] = std::max(1, support);
  }

  desc.order = GetProcessingOrder(roi.extent(), out_size, filter_support);;

  {
    ivec<spatial_ndim> pass_size = roi.extent();
    for (int pass = 0; pass < spatial_ndim; pass++) {
      int axis = desc.order[pass];

      if (pass < spatial_ndim - 1) {
        // for the last pass, the shape is out_size and is already set
        pass_size[axis] = out_size[axis];
        desc.tmp_shape(pass) = pass_size;
      }
    }
  }

  ComputeBlockLayout(desc);

  // this sets strides and offsets for all stages, including input and output
  for (int stage = 0; stage <= spatial_ndim; stage++) {
    ptrdiff_t stride = desc.channels;
    // There're fewer strides than dimensions.
    for (int d = 0; d < spatial_ndim - 1; d++) {
      stride *= desc.shapes[stage][d];
      desc.strides[stage][d] = stride;
    }
    desc.offsets[stage] = 0;
  }

  {
    int first_pass_axis = desc.order[0];
    auto strides = cat<ptrdiff_t>(channels, desc.strides[0]);
    for (int a = 0; a < spatial_ndim; a++) {
      if (a != first_pass_axis) {
        desc.origin[a] -= roi.lo[a];
        desc.in_offset() += roi.lo[a] * strides[a];
        desc.in_shape()[a] = roi.extent()[a];
      }
    }
  }

  /*if (desc.order == VertHorz()) {
    desc.origin.x -= roi.lo.x;
    desc.in_offset() += roi.lo.x * desc.channels;
    desc.in_shape().x = roi.extent().x;
  } else {
    desc.origin.y -= roi.lo.y;
    desc.in_offset() += roi.lo.y * desc.strides[0][0];
    desc.in_shape().y = roi.extent().y;
  }*/
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
