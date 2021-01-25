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
#include <algorithm>
#include "dali/core/util.h"
#include "dali/kernels/imgproc/resample/resampling_setup.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {
namespace resampling {

constexpr int kMaxGPUFilterSupport = 8192;

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
    const ResamplingParamsND<spatial_ndim> &params) const {
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

/**
 * @brief Caclulates the footprints of the processing stages.
 *
 * Note: The "ROI" calculated here is the relevant part of the source, not the user-specified box.
 * It will be extended to also contain filter footprint, which allows for pefect stitching.
 */
template <int spatial_ndim>
typename SeparableResamplingSetup<spatial_ndim>::ROI
SeparableResamplingSetup<spatial_ndim>::ComputeScaleAndROI(
    SampleDesc &desc, const ResamplingParamsND<spatial_ndim> &params) const {
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
    if (support > kMaxGPUFilterSupport)
      filter.rescale(kMaxGPUFilterSupport);

    float lo, hi;
    if (roi_start <= roi_end) {
      lo = roi_start - filter.anchor;
      hi = roi_end - filter.anchor + support;
    } else {  // flipped
      lo = roi_end - filter.anchor;
      hi = roi_start - filter.anchor + support;
    }
    roi.lo[axis] = std::max<int>(0, std::min<int>(desc.in_shape()[axis], std::floor(lo)));
    roi.hi[axis] = std::max<int>(0, std::min<int>(desc.in_shape()[axis], std::ceil(hi)));
  }

  return roi;
}

/**
 * @brief Calculates optimum processing order based on input/output sizes and filter support.
 *
 * The sizes of intermediate storage and time taken to compute the intermediate images
 * may depend on the order - i.e. if downscaling only one axis, it's beneficial to resample that
 * axis first, so that intermediate image is smaller.
 */
template <int ndim>
struct ProcessingOrderCalculator {
  ProcessingOrderCalculator(ivec<ndim> in_size, ivec<ndim> out_size, ivec<ndim> filter_support)
  : in_size(in_size), out_size(out_size), filter_support(filter_support) {
  }

  ProcessingOrder<ndim> operator()() {
    for (int i = 0; i < ndim; i++)
      best_order[i] = i;
    min_cost = 1e+30f;
    axis_mask = {};
    curr_size = in_size;
    run(0);
    return best_order;
  }

 private:
  float compute_cost_mul(int axis) {
    // y-axis is likely to be the cheapest
    return axis == 0 ? 1.4f : axis > 1 ? 1.2f : 1.0f;
  }

  static constexpr float size_bias = 3;

  float calculate_cost(int pass, int axis) {
    auto vol = volume(curr_size);
    float base_compute_cost = filter_support[axis] * vol;
    float compute_cost = compute_cost_mul(axis) * base_compute_cost;
    return compute_cost + vol * size_bias;
  }

  // recursively check every possible order in DFS fashion
  void run(int pass, float total_cost = 0) {
    if (total_cost >= min_cost)
      return;  // this branch of recursion will not yield a better result - abandon it

    if (pass == ndim) {
      min_cost = total_cost;
      best_order = curr_order;
    } else {
      for (int a = 0; a < ndim; a++) {
        if (axis_mask[a])
          continue;
        axis_mask[a] = true;
        auto prev_size = curr_size[a];
        curr_size[a] = out_size[a];
        curr_order[pass] = a;

        float pass_cost = calculate_cost(pass, a);
        run(pass + 1, total_cost + pass_cost);

        curr_size[a] = prev_size;
        axis_mask[a] = false;
      }
    }
  }

  float min_cost = 1e+30f;
  ivec<ndim> in_size, out_size, curr_size, filter_support;
  ProcessingOrder<ndim> best_order, curr_order;
  bvec<ndim> axis_mask;
};

template <int ndim>
ProcessingOrder<ndim> GetProcessingOrder(
    ivec<ndim> in_size,
    ivec<ndim> out_size,
    ivec<ndim> filter_support) {
  ProcessingOrderCalculator<ndim> poc(in_size, out_size, filter_support);
  return poc();
}

/**
 * @brief Calculates block layout for a 2D sample
 */
template <>
void SeparableResamplingSetup<2>::ComputeBlockLayout(SampleDesc &sample) const {
  for (int pass = 0; pass < 2; pass++) {
    int axis = sample.order[pass];
    // Horizontal pass (axis == 0) is processed in vertical slices
    // the width of block_dim and extending down the entire image.
    // Vertical pass is processed in horizontal slices spanning
    // the whole width of the image.

    ivec2 blk = sample.shapes[pass+1];
    blk[axis] = block_dim[axis];
    sample.logical_block_shape[pass] = blk;
  }
}

/**
 * @brief Calculates block layout for a 3D sample
 */
template <>
void SeparableResamplingSetup<3>::ComputeBlockLayout(SampleDesc &sample) const {
  if (volume(sample.out_shape()) == 0) {
    for (int pass = 0; pass < 3; pass++)
      sample.logical_block_shape[pass] = 0;
    return;
  }
  for (int pass = 0; pass < 3; pass++) {
    int axis = sample.order[pass];

    const int MaxElementsPerBlock = 1<<18;  // 256k elements, incl. channels

    // Horizontal pass (axis == 0) is processed in vertical slices
    // the width of block_dim and extending down the entire image.
    // Depth of the slice is calculated so that it contains no more than
    // MaxElementsPerBlock items.
    // Vertical pass is processed in horizontal slices spanning
    // the whole width of the image. Depth is handled like for horizontal pass.
    // Depth pass is calculated by taking up to 32 Z slices

    ivec3 pass_output_shape = sample.shapes[pass+1];
    ivec3 blk = pass_output_shape;
    if (axis < 2) {
      blk[axis] = block_dim[axis];
      blk.z = MaxElementsPerBlock / (blk[0] * blk[1] * sample.channels);
    } else {
      blk.z = block_dim.y;  // (yes, .y)
      blk.y = MaxElementsPerBlock / (blk[0] * blk[2] * sample.channels);
    }
    blk = clamp(blk, ivec3(1, 1, 1), pass_output_shape);
    sample.logical_block_shape[pass] = blk;
  }
}

/**
 * @brief Preprares a sample descriptor based on input shape and resampling parameters
 *
 * The sample descriptor contains input, output and intermediate buffer shapes, strides, etc.
 * The order in which resampling takes places is selected so that the cost
 * (weighted sum of memory consumption and compute time) is minimized.
 */
template <int spatial_ndim>
void SeparableResamplingSetup<spatial_ndim>::SetupSample(
    SampleDesc &desc,
    const TensorShape<tensor_ndim> &in_shape,
    const ResamplingParamsND<spatial_ndim> &params) const {
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

  desc.order = GetProcessingOrder(roi.extent(), out_size, filter_support);

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
}

/**
 * @brief Prepares sample descriptors and block info for entire batch
 *
 * This function populates the sample_descs, intermediate_shapes, intermediate_sizes etc.
 * It does not calculate block descriptors directly, but it calculates the number of blocks
 * required to calculate each stage of each sample using SampleDesc::logical_block_shape.
 */
template <int spatial_ndim>
void BatchResamplingSetup<spatial_ndim>::SetupBatch(
    const TensorListShape<tensor_ndim> &in, const Params &params) {
  constexpr int channel_dim =  BatchResamplingSetup<spatial_ndim>::channel_dim;
  if (!this->filters)
    this->Initialize();

  int N = in.num_samples();
  assert(params.size() == static_cast<span_extent_t>(N));

  sample_descs.resize(N);
  for (auto &shape : intermediate_shapes)
    shape.resize(N);

  output_shape.resize(N);
  for (auto &size : intermediate_sizes)
    size = 0;

  total_blocks = 0;

  for (int i = 0; i < N; i++) {
    SampleDesc &desc = sample_descs[i];
    auto ts_in = in.tensor_shape(i);
    this->SetupSample(desc, ts_in, params[i]);

    for (int t = 0; t < num_tmp_buffers; t++) {
      TensorShape<tensor_ndim> ts_tmp = shape_cat(vec2shape(desc.tmp_shape(t)), desc.channels);
      intermediate_shapes[t].set_tensor_shape(i, ts_tmp);
      intermediate_sizes[t] += volume(ts_tmp);
    }

    auto ts_out = output_shape.tensor_shape_span(i);
    static_assert(channel_dim == spatial_ndim, "Shape calculation requires channel-last layout");
    auto sample_shape = shape_cat(vec2shape(desc.out_shape()), desc.channels);

    output_shape.set_tensor_shape(i, sample_shape);
    if (volume(desc.out_shape()) == 0)
      continue;  // this sample does not generate any blocks

    for (int pass = 0; pass < spatial_ndim; pass++) {
      ivec<spatial_ndim> blocks;
      for (int d = 0; d < spatial_ndim; d++) {
        blocks[d] = div_ceil(desc.shapes[pass+1][d], desc.logical_block_shape[pass][d]);
      }
      total_blocks[pass] += volume(blocks);
    }
  }
}

template <int n>
int AddBlocks(BlockDesc<n> *blocks, int sample_idx,
              ivec<n> block_size, ivec<n> extent,
              ivec<n> current_block = {}, int current_dim = 0) {
  if (current_dim == n) {
    blocks[0].sample_idx = sample_idx;
    blocks[0].start = current_block * block_size;
    blocks[0].end = dali::min((current_block + 1) * block_size, extent);
    return 1;
  } else {
    int nblocks = 0;
    for (int pos = 0; pos < extent[current_dim]; pos += block_size[current_dim]) {
      nblocks += AddBlocks(blocks + nblocks, sample_idx,
                           block_size, extent, current_block, current_dim + 1);
      current_block[current_dim]++;
    }
    return nblocks;
  }
}

/**
 * @brief Calculates the mapping from grid block indices to samples and regions within samples
 *
 * Each block descriptor contains the sample index and the range of output coordinates processed
 * by this block (lo, hi).
 */
template <int spatial_ndim>
void BatchResamplingSetup<spatial_ndim>::InitializeSampleLookup(
    const OutTensorCPU<BlockDesc, 1> &sample_lookup) {
  int blocks_in_all_passes = 0;
  for (int i = 0; i < spatial_ndim; i++)
    blocks_in_all_passes += total_blocks[i];

  assert(sample_lookup.shape[0] >= blocks_in_all_passes);
  (void)blocks_in_all_passes;  // for non-debug builds

  int block = 0;
  int N = sample_descs.size();
  for (int pass = 0; pass < spatial_ndim; pass++) {
    for (int i = 0; i < N; i++) {
      auto &desc = sample_descs[i];
      if (volume(desc.out_shape()) > 0) {
        int sample_block_count = AddBlocks(sample_lookup.data + block, i,
                                          desc.logical_block_shape[pass], desc.shapes[pass+1]);
        block += sample_block_count;
      }
    }
  }
  assert(block == blocks_in_all_passes);
}

template class BatchResamplingSetup<2>;
template class BatchResamplingSetup<3>;

}  // namespace resampling
}  // namespace kernels
}  // namespace dali
