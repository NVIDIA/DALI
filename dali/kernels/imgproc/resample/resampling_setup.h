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
#include "dali/kernels/dev_array.h"

namespace dali {
namespace kernels {

/// @brief Maps a block (by blockIdx) to a sample.
struct SampleBlockInfo {
  int sample, block_in_sample;
};

/// @brief Builds and maintains resampling setup
struct SeparableResamplingSetup {
  using Params = std::vector<ResamplingParams2D>;

  enum ProcessingOrder : int {
    HorzVert,
    VertHorz
  };

  enum BufferIdx : int {
    IdxIn = 0,
    IdxTmp = 1,
    IdxOut = 2,
  };

  /// Number of blocks per pass may differ depending on
  /// the image aspect ratio and block aspect ratio.
  struct BlockCount {
    int pass[2];
  };

  struct SampleDesc {
    using DevShape = DeviceArray<int, 2>;

    DeviceArray<ptrdiff_t, 3> offsets;
    DeviceArray<int, 3>       strides;
    DeviceArray<DevShape, 3>  shapes;

    DeviceArray<float, 2> origin, scale;

    ProcessingOrder order;
    int channels;
    ResamplingFilterType filter_type[2];
    ResamplingFilter filter[2];

    BlockCount block_count;
  };

  static_assert(std::is_pod<SampleDesc>::value,
    "Internal error! Something crept into SampleDesc and made it non-POD");

  std::vector<SampleDesc> sample_descs;
  TensorListShape<3> output_shape, intermediate_shape;
  size_t intermediate_size;
  BlockCount total_blocks;
  int2 block_size = { 32, 24 };

  std::shared_ptr<ResamplingFilters> filters;
  void Initialize(cudaStream_t stream) {
    filters = GetResamplingFilters(stream);
  }

  void SetupComputation(const TensorListShape<3> &in, const Params &params);
  void InitializeSampleLookup(const OutTensorCPU<SampleBlockInfo, 1> &sample_lookup);

  struct ROI {
    int lo[2], hi[2];
    int size(int dim) const { return hi[dim] - lo[dim]; }
  };

  void SetFilters(SampleDesc &desc, const ResamplingParams2D &params);
  ROI ComputeScaleAndROI(SampleDesc &desc, const ResamplingParams2D &params);
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_SETUP_H_
