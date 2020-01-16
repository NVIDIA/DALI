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

#ifndef DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_OPTICAL_FLOW_ADAPTER_H_
#define DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_OPTICAL_FLOW_ADAPTER_H_

#include <string>

#include "dali/core/backend_tags.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/backend.h"

namespace dali {
namespace optical_flow {

enum struct VectorGridSize {
  UNDEF,
  SIZE_4 = 4,  /// 4x4 grid
  MAX,
};

struct OpticalFlowParams {
  float perf_quality_factor;  /// 0..1, where 0 is best quality, lowest performance
  VectorGridSize grid_size;
  bool enable_temporal_hints;
  bool enable_external_hints;
};

using dali::TensorView;

template<typename ComputeBackend>
class DLL_PUBLIC OpticalFlowAdapter {
 protected:
  using StorageBackend = typename compute_to_storage<ComputeBackend>::type;

 public:
  explicit OpticalFlowAdapter(OpticalFlowParams params) : of_params_(params) {}


  /**
   * Return shape of output tensor for given OpticalFlow class
   */
  virtual TensorShape<DynamicDimensions> GetOutputShape() = 0;


  /**
   * Perform OpticalFlow calculation.
   */
  virtual void CalcOpticalFlow(TensorView<StorageBackend, const uint8_t, 3> reference_image,
                               TensorView<StorageBackend, const uint8_t, 3> input_image,
                               TensorView<StorageBackend, float, 3> output_image,
                               TensorView<StorageBackend, const float, 3> external_hints = TensorView<StorageBackend, const float, 3>()) = 0;  // NOLINT


  virtual ~OpticalFlowAdapter() = default;

 protected:
  const OpticalFlowParams of_params_;
};

}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_OPTICAL_FLOW_ADAPTER_H_
