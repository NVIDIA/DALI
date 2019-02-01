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

#ifndef DALI_AUX_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_H_
#define DALI_AUX_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_H_

#include <dali/pipeline/data/backend.h>
#include <dali/kernels/backend_tags.h>
#include <dali/kernels/tensor_view.h>

namespace dali {
namespace optical_flow {

namespace detail {

template<typename ComputeBackend>
struct Compute2Storage {
  using type = kernels::StorageCPU;
};

template<>
struct Compute2Storage<kernels::ComputeGPU> {
  using type = kernels::StorageGPU;
};

}  // namespace detail

enum VectorGridSize {
  UNDEF,
  SIZE_4,  /// 4x4 grid
  MAX,
};

struct OpticalFlowParams {
  float perf_quality_factor = .0f;  /// 0..1, where 0 is best quality, lowest performance
  VectorGridSize grid_size = UNDEF;
  bool enable_hints = false;
};

using dali::kernels::TensorView;

template<typename ComputeBackend>
class DLL_PUBLIC OpticalFlowAdapter {
 protected:
  using StorageBackend = typename detail::Compute2Storage<ComputeBackend>::type;
 public:
  explicit OpticalFlowAdapter(OpticalFlowParams params) {}


  virtual void CalcOpticalFlow(TensorView<StorageBackend, const uint8_t, 3> reference_image,
                               TensorView<StorageBackend, const uint8_t, 3> input_image,
                               TensorView<StorageBackend, float, 3> output_image,
                               TensorView<StorageBackend, const float, 3> external_hints = TensorView<StorageBackend, const float, 3>()) = 0;  // NOLINT


  virtual ~OpticalFlowAdapter() = default;
};

}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_AUX_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_H_
