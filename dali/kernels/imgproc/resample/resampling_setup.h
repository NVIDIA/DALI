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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_SETUP_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_SETUP_H_

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/core/dev_array.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {
namespace resampling {

template <int spatial_ndim>
using ProcessingOrder = i8vec<spatial_ndim>;

constexpr ProcessingOrder<2> VertHorz() { return { 1, 0 }; }
constexpr ProcessingOrder<2> HorzVert() { return { 0, 1 }; }

template <int spatial_ndim>
struct SampleDesc {
  using Shape = ivec<spatial_ndim>;
  using Strides = vec<spatial_ndim - 1, ptrdiff_t>;
  static constexpr int num_tmp_buffers = spatial_ndim - 1;
  static constexpr int num_buffers = num_tmp_buffers + 2;

  DeviceArray<uintptr_t, num_buffers> pointers;
  DeviceArray<ptrdiff_t, num_buffers> offsets;
  DeviceArray<Strides, num_buffers>   strides;
  DeviceArray<Shape, num_buffers>  shapes;

  template <typename Input, typename Tmp, typename Output, int D = spatial_ndim>
  void set_base_pointers(Input *in, Tmp *tmp, Output *out) {
    static_assert(D == 2 && D == spatial_ndim, "This overload is only usable for 2D resampling");
    std::array<Tmp *, 1> tmp_arr = {{ tmp }};
    set_base_pointers(in, tmp_arr, out);
  }

  template <typename Input, typename Tmp, typename Output>
  void set_base_pointers(Input *in, std::array<Tmp *, num_tmp_buffers> tmp, Output *out) {
    int i = 0;
    pointers[i] = reinterpret_cast<uintptr_t>(in  + offsets[i]);
    i++;
    for (i = 1; i < num_buffers - 1; i++)
      pointers[i] = reinterpret_cast<uintptr_t>(tmp[i - 1] + offsets[i]);
    pointers[i] = reinterpret_cast<uintptr_t>(out + offsets[i]);
  }

  DALI_HOST_DEV Shape &in_shape()                   { return shapes[0]; }
  DALI_HOST_DEV const Shape &in_shape() const       { return shapes[0]; }
  DALI_HOST_DEV Shape &tmp_shape(int i)             { return shapes[i+1]; }
  DALI_HOST_DEV const Shape &tmp_shape(int i) const { return shapes[i+1]; }
  DALI_HOST_DEV Shape &out_shape()                  { return shapes[spatial_ndim]; }
  DALI_HOST_DEV const Shape &out_shape() const      { return shapes[spatial_ndim]; }

  DALI_HOST_DEV ptrdiff_t  &in_offset()            { return offsets[0]; }
  DALI_HOST_DEV ptrdiff_t   in_offset() const      { return offsets[0]; }
  DALI_HOST_DEV ptrdiff_t &tmp_offset(int i)       { return offsets[i+1]; }
  DALI_HOST_DEV ptrdiff_t  tmp_offset(int i) const { return offsets[i+1]; }
  DALI_HOST_DEV ptrdiff_t &out_offset()            { return offsets[spatial_ndim]; }
  DALI_HOST_DEV ptrdiff_t  out_offset() const      { return offsets[spatial_ndim]; }

  template <typename T>
  DALI_HOST_DEV const T *in_ptr() const { return reinterpret_cast<const T*>(pointers[0]); }
  template <typename T>
  DALI_HOST_DEV T *tmp_ptr(int i) const { return reinterpret_cast<T*>(pointers[i+1]); }
  template <typename T>
  DALI_HOST_DEV T *out_ptr() const { return reinterpret_cast<T*>(pointers[spatial_ndim]); }

  vec<spatial_ndim> origin, scale;

  ProcessingOrder<spatial_ndim> order;
  int channels;
  ResamplingFilterType filter_type[spatial_ndim];  // NOLINT
  ResamplingFilter filter[spatial_ndim];           // NOLINT

  /**
   * @brief Number of blocks per pass
   *
   * This is kept as an array (instead of vector) to avoid indexing confusion;
   * block_count[0] refers to number of blocks in first pass, regarless of actual processing
   * order
   */
  DeviceArray<int, spatial_ndim> block_count;
};

/**
 * @brief Maps a block (by blockIdx) to a sample.
 */
struct SampleBlockInfo {
  int sample, block_in_sample;
};

ResamplingFilter GetResamplingFilter(const ResamplingFilters *filters, const FilterDesc &params);

/**
 * @brief Builds and maintains resampling setup
 */
template <int _spatial_ndim>
class SeparableResamplingSetup {
 public:
  static constexpr int channel_dim = _spatial_ndim;  // assumes interleaved channel data
  static constexpr int spatial_ndim = _spatial_ndim;
  static constexpr int tensor_ndim = spatial_ndim + (channel_dim >= 0 ? 1 : 0);
  using SampleDesc = resampling::SampleDesc<spatial_ndim>;

  /**
   * @brief Number of logical temporary buffers
   *
   * Last pass writes directly to output.
   * Physical buffers may overlap for more than 3 dimensions.
   */
  static constexpr int num_tmp_buffers = spatial_ndim - 1;
  /** @brief Number of buffers: temporaries + input + output */
  static constexpr int num_buffers = num_tmp_buffers + 2;

  DLL_PUBLIC void SetupSample(SampleDesc &desc,
                              const TensorShape<tensor_ndim> &in_shape,
                              const ResamplingParamsND<spatial_ndim> &params);

  void Initialize() {
    filters = GetResamplingFilters();
  }
  void InitializeCPU() {
    filters = GetResamplingFiltersCPU();
  }

  int2 block_size = { 32, 24 };

 protected:
  using ROI = Roi<spatial_ndim>;

  void SetFilters(SampleDesc &desc, const ResamplingParamsND<spatial_ndim> &params);
  ROI ComputeScaleAndROI(SampleDesc &desc, const ResamplingParamsND<spatial_ndim> &params);

  std::shared_ptr<ResamplingFilters> filters;

  static_assert(std::is_pod<SampleDesc>::value,
    "Internal error! Something crept into SampleDesc and made it non-POD");
};

template <int _spatial_ndim>
class BatchResamplingSetup : public SeparableResamplingSetup<_spatial_ndim> {
 public:
  using Base = SeparableResamplingSetup<_spatial_ndim>;
  using Base::spatial_ndim;
  using Base::tensor_ndim;
  using Base::num_tmp_buffers;
  using Params = span<ResamplingParamsND<spatial_ndim>>;
  using SampleDesc = resampling::SampleDesc<spatial_ndim>;

  std::vector<SampleDesc> sample_descs;
  TensorListShape<3> output_shape, intermediate_shapes[num_tmp_buffers]; // NOLINT
  size_t intermediate_sizes[num_tmp_buffers];  // NOLINT
  ivec<spatial_ndim> total_blocks;

  DLL_PUBLIC void SetupBatch(const TensorListShape<tensor_ndim> &in, const Params &params);

  template <typename Collection>
  void SetupBatch(const TensorListShape<tensor_ndim> &in, const Collection &params) {
    SetupBatch(in, make_span(params));
  }
  DLL_PUBLIC void InitializeSampleLookup(const OutTensorCPU<SampleBlockInfo, 1> &sample_lookup);
};

}  // namespace resampling
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_SETUP_H_
