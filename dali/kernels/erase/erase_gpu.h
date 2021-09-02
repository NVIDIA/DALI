// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_ERASE_ERASE_GPU_H_
#define DALI_KERNELS_ERASE_ERASE_GPU_H_

#include <vector>

#include "dali/core/dev_array.h"
#include "dali/core/format.h"
#include "dali/core/geom/box.h"
#include "dali/core/geom/vec.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/erase/erase_args.h"

namespace dali {
namespace kernels {

/**
 * We divide the work in following way:
 *
 * We cover every n-dimensional sample by n-dimensional cubes (in fact a hyperrectangle)
 * of one selected size. We call such a hyperrectangle a region.
 *
 * The size of such region is suitable for traversal of block of threads of dimension (32, 32, 1)
 *
 *
 * Grid: (number of cubes, number of samples, 1)
 * - the number of cubes is the max over the numbers of cubes to cover each sample
 *
 * Block: (32, 32, 1) - the dimensions of thread block are fixed. We use such "physical block"
 * to traverse every region - the hyperrectangle.
 *
 * Steps:
 * 1. In every block we first look for intersections with erased regions, to filter out only
 * the ones that we actually need.
 * 2. We check if the hyperrectangle is fully covered by erase region or not covered at all
 * to delegate to special cases - fill only or memcopy
 * 3. If we have mixed cover, we check each coordinate and either copy the input or the fill value
 *
 * The processing of hyperrectangle of n-dims is done by n-nested for loop from outer to innermost
 * dimension.
 * The two innermost dimensions (omitting the channel) are traversed by the block of parallel
 * threads.
 * The loop is implemented as template based recursion.
 * Possible optimization/obfuscation - factor out checks over dimension to specific level of
 * the loop nest (for the mixed cover case).
 *
 */

template <typename T, int ndim>
using box = Box<ndim, T>;

template <int ndim>
using ibox = box<int32_t, ndim>;

template <typename T, int ndim>
struct erase_sample_desc {
  const T * __restrict__ in = nullptr;
  T * __restrict__ out = nullptr;
  span<const ibox<ndim>> erase_regions = {};
  ivec<ndim> sample_shape;
  ivec<ndim> sample_stride;
};

template <int ndim>
DALI_HOST_DEV ivec<ndim> get_region_start(int region_idx, ivec<ndim> region_shape,
                                          ivec<ndim> sample_shape) {
  // let's calculate how many regions cover given axes:
  auto region_cover_shape = div_ceil(sample_shape, region_shape);
  // this introduces the same ordering of regions as data layout
  auto region_cover_strides = GetStrides(region_cover_shape);
  ivec<ndim> region_position{};  // n-dim index of region among other regions
  region_position[0] = region_idx / region_cover_strides[0];
  for (int i = 1; i < ndim; i++) {
    region_idx -= region_position[i - 1] * region_cover_strides[i - 1];
    region_position[i] = region_idx / region_cover_strides[i];
  }
  return region_position * region_shape;
}

template <int ndim>
__device__ ibox<ndim> get_region(int region_idx, ivec<ndim> region_shape, ivec<ndim> sample_shape) {
  auto anchor = get_region_start(region_idx, region_shape, sample_shape);
  return {anchor, anchor + region_shape};
}

constexpr unsigned FULL_MASK = 0xffffffffu;

/**
 * @brief Returns true if that parameter configuration is just outer layer of recursive loop call
 */
constexpr bool is_outer_loops(int ndim, int current_dim, int channel_dim) {
  // channel_dim is not one of the 2 last dimensions
  if (channel_dim < ndim - 2) {
    return ndim - current_dim > 2;  // true if we are not in the last two dimensions
  } else {
    return ndim - current_dim > 3;  // we have to select a variant that accommodates channel dim
  }
}


/**
 * @brief Outer layer of recursive loop over current region
 */
template <typename Worker, int channel_dim, int current_dim = 0, typename T, int ndim>
__device__ std::enable_if_t<is_outer_loops(ndim, current_dim, channel_dim)>
erase_generic(erase_sample_desc<T, ndim> sample, ibox<ndim> region, span<T> fill_values,
              span<ibox<ndim>> erase_regions = {}, ivec<ndim> coordinate = {},
              int64_t offset_base = 0) {
  constexpr int d = current_dim;
  int boundary = ::min(region.hi[d], sample.sample_shape[d]);
  for (int i = region.lo[d]; i < boundary; i++) {
    coordinate[d] = i;
    int64_t offset = offset_base + i * sample.sample_stride[d];
    erase_generic<Worker, channel_dim, d + 1>(sample, region, fill_values, erase_regions,
                                              coordinate, offset);
  }
}

/**
 * We have 2 full most inner loops, channel_dim is somewhere on the way here, or doesn't exist
 *
 * We handle this by covering the innermost dimensions by:
 * for (y : blockDim.y)
 *   for (x : blockDim.x)
 *     process {coordinate[0], coordinate[1], ..., coordinate[current_dim - 1], y, x}
 */
template <typename Worker, int channel_dim, int current_dim = 0, typename T, int ndim>
__device__ std::enable_if_t<ndim - current_dim == 2 && channel_dim < ndim - 2>
erase_generic(erase_sample_desc<T, ndim> sample, ibox<ndim> region, span<T> fill_values,
              span<ibox<ndim>> erase_regions = {}, ivec<ndim> coordinate = {},
              int64_t offset_base = 0) {
  constexpr int d = current_dim;
  constexpr int dY = d;
  constexpr int dX = d + 1;
  int boundary_y = ::min(region.hi[dY], sample.sample_shape[dY]);
  int boundary_x = ::min(region.hi[dX], sample.sample_shape[dX]);

  for (int y = region.lo[dY] + threadIdx.y; y < boundary_y; y += blockDim.y) {
    coordinate[dY] = y;
    int64_t offset_y = offset_base + y * sample.sample_stride[dY];
    for (int x = region.lo[dX] + threadIdx.x; x < boundary_x; x += blockDim.x) {
      coordinate[dX] = x;
      int64_t offset_x = offset_y + x * sample.sample_stride[dX];
      Worker::template copy_or_erase<channel_dim>(sample, erase_regions, coordinate, offset_x,
                                                  fill_values);
    }
  }
}

/**
 * Channel dim is the one of the last two dimensions
 *
 * We loop with thread block over ndim-3 and fused (ndim-2, ndim-1) dims
 * for (y : blockDim.y)
 *   for (idx : blockDim.x * numChannels) // fused loop for last two dimensions
 *     x = idx / stride_x
 *     c = idx % stride_x
 *       process {coordinate[0], coordinate[1], ..., coordinate[current_dim - 1], y, x, c}
 *
 * Note: naming follows {..., y, x, c}, but it can also be {..., y, c, x}
 */
template <typename Worker, int channel_dim, int current_dim = 0, typename T, int ndim>
__device__ std::enable_if_t<ndim - current_dim == 3 && channel_dim >= ndim - 2>
erase_generic(erase_sample_desc<T, ndim> sample, ibox<ndim> region, span<T> fill_values,
              span<ibox<ndim>> erase_regions = {}, ivec<ndim> coordinate = {},
              int64_t offset_base = 0) {
  constexpr int d = current_dim;
  constexpr int dY = d;
  constexpr int dX = d + 1;
  constexpr int dC = d + 2;
  int boundary_y = ::min(region.hi[dY], sample.sample_shape[dY]);
  int boundary_x = ::min(region.hi[dX], sample.sample_shape[dX]);
  int boundary_c = ::min(region.hi[dC], sample.sample_shape[dC]);

  int flattened_start = region.lo[dX] * sample.sample_stride[dX] + region.lo[dC] + threadIdx.x;
  int flattened_end = region.hi[dX] * sample.sample_stride[dX] + region.hi[dC];
  int start_x = flattened_start / sample.sample_stride[dX];
  int start_c = flattened_start % sample.sample_stride[dX];
  int step_x = blockDim.x / sample.sample_stride[dX];
  int step_c = blockDim.x % sample.sample_stride[dX];
  for (int y = region.lo[dY] + threadIdx.y; y < boundary_y; y += blockDim.y) {
    coordinate[dY] = y;
    int64_t offset_y = offset_base + y * sample.sample_stride[dY];
    for (int x = start_x, c = start_c, flat = flattened_start; flat < flattened_end;
         x += step_x, c += step_c, flat += blockDim.x) {
      if (c >= sample.sample_stride[dX]) {
        x += 1;
        c -= sample.sample_stride[dX];
      }
      coordinate[dX] = x;
      coordinate[dC] = c;
      if (region.lo[dX] <= coordinate[dX] && coordinate[dX] < boundary_x &&
          region.lo[dC] <= coordinate[dC] && coordinate[dC] < boundary_c) {
        int64_t offset_x = offset_y + flat;
        Worker::template copy_or_erase<channel_dim>(sample, erase_regions, coordinate,
                                                             offset_x, fill_values);
      }
    }
  }
}

/**
 * @brief Edge case kernel, 1D case
 */
template <typename Worker, int channel_dim, int current_dim = 0, typename T, int ndim>
__device__ std::enable_if_t<ndim == 1> erase_generic(erase_sample_desc<T, ndim> sample,
    ibox<ndim> region, span<T> fill_values, span<ibox<ndim>> erase_regions = {},
    ivec<ndim> coordinate = {}, int64_t offset_base = 0) {
  constexpr int d = current_dim;
  int boundary = ::min(region.hi[d], sample.sample_shape[d]);

  for (int idx = region.lo[d] + threadIdx.y * blockDim.x + threadIdx.x; idx < boundary;
       idx += blockDim.y * blockDim.x) {
    coordinate[d] = idx;
    Worker::template copy_or_erase<channel_dim>(sample, erase_regions, coordinate,
                                                offset_base + idx, fill_values);
  }
}

/**
 * @brief Edge case kernel, 2D case with channels
 *
 * TODO(klecki): fuse the loops
 */
template <typename Worker, int channel_dim, int current_dim = 0, typename T, int ndim>
__device__ std::enable_if_t<(ndim == 2 && current_dim == 0 && channel_dim >= 0)> erase_generic(
    erase_sample_desc<T, ndim> sample, ibox<ndim> region, span<T> fill_values,
    span<ibox<ndim>> erase_regions = {}, ivec<ndim> coordinate = {}, int64_t offset_base = 0) {
  constexpr int d = current_dim;
  constexpr int dC = d + 1;
  int boundary = ::min(region.hi[d], sample.sample_shape[d]);
  int boundary_c = ::min(region.hi[dC], sample.sample_shape[dC]);

  for (int idx = region.lo[d] + threadIdx.y * blockDim.x + threadIdx.x; idx < boundary;
       idx += blockDim.y * blockDim.x) {
    coordinate[d] = idx;
    int64_t offset = offset_base + idx * sample.sample_stride[d];
    for (int c = region.lo[dC]; c < boundary_c; c++) {
      coordinate[dC] = c;
      Worker::template copy_or_erase<channel_dim>(sample, erase_regions, coordinate, offset + c,
                                                  fill_values);
    }
  }
}


struct do_copy {
  template <int channel_dim, typename T, int ndim>
  __device__ static void copy_or_erase(erase_sample_desc<T, ndim> sample,
                                       span<ibox<ndim>> erase_regions, ivec<ndim> coordinate,
                                       int64_t offset, span<T> fill_values) {
    (void)erase_regions;
    (void)coordinate;
    (void)fill_values;
    sample.out[offset] = sample.in[offset];
  }
};

struct do_erase {
  template <int channel_dim, typename T, int ndim>
  __device__ static void copy_or_erase(erase_sample_desc<T, ndim> sample,
                                       span<ibox<ndim>> erase_regions, ivec<ndim> coordinate,
                                       int64_t offset, span<T> fill_values) {
    (void)erase_regions;
    if (channel_dim == -1) {
      sample.out[offset] = fill_values[0];
    } else {
      sample.out[offset] = fill_values[coordinate[channel_dim]];
    }
  }
};

struct do_copy_or_erase {
  template <int channel_dim, typename T, int ndim>
  __device__ static void copy_or_erase(erase_sample_desc<T, ndim> sample,
                                       span<ibox<ndim>> erase_regions, ivec<ndim> coordinate,
                                       int64_t offset, span<T> fill_values) {
    auto fill_value = channel_dim == -1 ? fill_values[0] : fill_values[coordinate[channel_dim]];
    copy_or_erase<channel_dim>(sample, erase_regions, coordinate, offset, fill_value);
  }

  template <int channel_dim, typename T, int ndim>
  __device__ static void copy_or_erase(erase_sample_desc<T, ndim> sample,
                                       span<ibox<ndim>> erase_regions, ivec<ndim> coordinate,
                                       int64_t offset, T fill_value) {
    bool copy = true;
    for (auto &region : erase_regions) {
      if (region.contains(coordinate)) {
        copy = false;
      }
    }
    sample.out[offset] = copy ? sample.in[offset] : fill_value;
  }
};

template <int channel_dim = -1, typename T, int ndim = 2>
__global__ void erase_gpu_impl(const erase_sample_desc<T, ndim> *samples,
                               ivec<ndim> region_shape, span<T> fill_values) {
  const auto region_idx = blockIdx.x;
  const auto sample_idx = blockIdx.y;

  const auto &sample = samples[sample_idx];
  ivec<ndim> sample_shape = sample.sample_shape;

  auto region_box = get_region(region_idx, region_shape, sample_shape);

  const auto *erase_regions_ptrs = sample.erase_regions.data();
  const auto erase_regions_count = sample.erase_regions.size();
  constexpr int max_regions = 40 * 1024 / sizeof(ibox<ndim>);
  __shared__ ibox<ndim> erase_regions[max_regions];
  __shared__ int filtered_region_idx;

  filtered_region_idx = 0;
  __syncthreads();
  int full_cover = false;

  // check which erase_regions overlap with current region_box
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < erase_regions_count;
       i += blockDim.y * blockDim.x) {
    if (erase_regions_ptrs[i].overlaps(region_box)) {
      erase_regions[atomicAdd(&filtered_region_idx, 1)] = erase_regions_ptrs[i];
      if (erase_regions_ptrs[i].contains(region_box)) {
        full_cover = true;
        break;
      }
    }
  }
  __syncthreads();

  //  Check if we have full or no cover at all
  int total_erase_regions = filtered_region_idx;
  int is_fully_erased = __any_sync(FULL_MASK, full_cover);

  if (total_erase_regions == 0) {
    // do a full copy
    erase_generic<do_copy, channel_dim>(sample, region_box, fill_values);
    return;
  } else if (is_fully_erased) {
    // do a total erase
    erase_generic<do_erase, channel_dim>(sample, region_box, fill_values,
                                         make_span(erase_regions, total_erase_regions));
    return;
  } else {
    erase_generic<do_copy_or_erase, channel_dim>(sample, region_box, fill_values,
                                                 make_span(erase_regions, total_erase_regions));
  }
}

template <int ndim>
ivec<ndim> to_ivec(TensorShape<ndim> shape) {
  ivec<ndim> result;
  for (int d = 0; d < ndim; d++) {
    result[d] = shape[d];
  }
  return result;
}

template <typename T, int ndim, int channel_dim = -1>
struct EraseGpu {
  static_assert(
      -1 <= channel_dim && channel_dim < ndim,
      "Channel dim can either be ignored (for channel_dim = -1) or must be in [0, ndim) interval.");
  using sample_t = erase_sample_desc<T, ndim>;

  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<T, ndim> &in,
                           const InListGPU<ibox<ndim>, 1> &erased_regions,
                           span<const T> fill_values = {}) {
    const int num_samples = in.num_samples();

    if (channel_dim == -1) {
      DALI_ENFORCE(fill_values.size() == 1 || fill_values.size() == 0,
                   make_string("Kernel is unaware of channel dimension, exactly 0 or 1 fill value "
                               "is expected, got: ",
                               fill_values.size(), "."));
    } else {
      // ensure that dim `channel_dim` in every sample matches fill_values.size();
      for (int i = 0; i < num_samples; i++) {
        DALI_ENFORCE(in.shape.tensor_shape_span(i)[channel_dim] == fill_values.size() ||
                         fill_values.size() == 1 || fill_values.size() == 0,
                     make_string("Channel dimension in all samples must match the number "
                                 "of elements provided in `fill_values` or there need to be "
                                 "exactly 0 or 1 fill values provied. Sample ",
                                 i, " has ", in.shape.tensor_shape_span(i)[channel_dim],
                                 " elements, but there are ", fill_values.size(), " fill values."));
      }
    }

    KernelRequirements req;

    ScratchpadEstimator se;

    se.add<mm::memory_kind::host, sample_t>(num_samples);
    se.add<mm::memory_kind::device, sample_t>(num_samples);
    if (channel_dim >= 0) {
      int num_channels = in.shape.tensor_shape_span(0)[channel_dim];
      se.add<mm::memory_kind::host, T>(num_channels);
      se.add<mm::memory_kind::device, T>(num_channels);
    } else {
      se.add<mm::memory_kind::host, T>(1);
      se.add<mm::memory_kind::device, T>(1);
    }

    req.output_shapes = {in.shape};
    req.scratch_sizes = se.sizes;
    return req;
  }

  /**
   * @brief Erase regions from coresponding samples.
   *
   * fill_values can be either:
   * * empty - every region is filled with a default 0 value.
   * * contain 1 element - the single value is used to fill the erased regions
   * * if channel_dim >= 0, it can contain as many elements as the specified channel_dimension,
   *   one value per channel is used when filling the erased regions
   */
  void Run(KernelContext &ctx,
           OutListGPU<T, ndim> &out,
           const InListGPU<T, ndim> &in,
           const InListGPU<ibox<ndim>, 1> &erased_regions,
           span<const T> fill_values = {}) {
    auto stream = ctx.gpu.stream;  // MAYBE some runtime check if it's not used?
    const int num_samples = in.num_samples();

    DeviceArray<T, 1> default_fill = {0};

    if (fill_values.empty()) {
      fill_values = make_span(default_fill);
    }

    const int num_fill_values = channel_dim >= 0 ? in.shape.tensor_shape_span(0)[channel_dim] : 1;

    ivec<ndim> region_dim;

    // Prepare Dim {2, 2, ..., 2, 64, 64}
    for (int d = 0; d < ndim - 2; d++) {
      region_dim[d] = 2;
    }
    if (ndim > 2) {
      region_dim[ndim - 2] = 64;
      region_dim[ndim - 1] = 128;
      if (channel_dim >= 0) {
        const int channels = in.shape[0][channel_dim];
        if (channel_dim == ndim - 1) {
          region_dim[ndim - 3] = region_dim[ndim - 2];
          region_dim[ndim - 2] = region_dim[ndim - 1];
          region_dim[ndim - 1] = channels;
        } else if (channel_dim == ndim - 2) {
          region_dim[ndim - 3] = region_dim[ndim - 2];
          region_dim[ndim - 2] = channels;
        } else {
          region_dim[channel_dim] = channels;
        }
      }
    } else if (ndim == 2) {
      region_dim[0] = 64;
      region_dim[1] = 128;
      if (channel_dim >= 0) {
        const int channels = in.shape[0][channel_dim];
        region_dim[channels] = channels;
      }
    } else {
      region_dim[0] = 32 * 32;
    }


    int max_regions = 1;
    for (int i = 0; i < in.num_samples(); i++) {
      auto region_cover = div_ceil(to_ivec(in.shape[i]), region_dim);
      auto total_regions = volume(region_cover);
      // std::max was complaining
      max_regions = max_regions >= total_regions ? max_regions : total_regions;
    }
    dim3 grid_dim = {(uint32_t)max_regions, (uint32_t)num_samples, 1};
    dim3 block_dim = {32, 32, 1};  // fixed block dim

    auto* sample_desc_cpu = ctx.scratchpad->AllocateHost<sample_t>(num_samples);
    auto* fill_values_cpu = ctx.scratchpad->AllocateHost<T>(num_fill_values);

    for (int i = 0; i < num_fill_values; i++) {
      fill_values_cpu[i] = fill_values.size() == 1 ? fill_values[0] : fill_values[i];
    }

    for (int i = 0; i < num_samples; i++) {
      auto &sample = sample_desc_cpu[i];
      sample.in = in.data[i];
      sample.out = out.data[i];
      sample.erase_regions = make_cspan(erased_regions.tensor_data(i),
                                        erased_regions.tensor_shape(i).num_elements());
      for (int dim = 0; dim < ndim; dim++) {
        sample.sample_shape[dim] = in.tensor_shape_span(i)[dim];
      }
      sample.sample_stride = GetStrides(sample.sample_shape);
    }

    sample_t *sample_desc_gpu;
    T *fill_values_gpu;
    std::tie(sample_desc_gpu, fill_values_gpu) =
      ctx.scratchpad->ToContiguousGPU(stream, make_cspan(sample_desc_cpu, num_samples),
                                      make_cspan(fill_values_cpu, num_fill_values));
    auto fill_values_span = make_span(fill_values_gpu, num_fill_values);

    erase_gpu_impl<channel_dim><<<grid_dim, block_dim, 0, stream>>>(
        sample_desc_gpu, region_dim, fill_values_span);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ERASE_ERASE_GPU_H_
