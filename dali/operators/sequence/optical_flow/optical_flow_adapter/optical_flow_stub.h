// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_OPTICAL_FLOW_STUB_H_
#define DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_OPTICAL_FLOW_STUB_H_

#include <vector>

#include "dali/core/tensor_view.h"
#include "dali/operators/sequence/optical_flow/optical_flow_adapter/optical_flow_adapter.h"

namespace dali {
namespace optical_flow {

/**
 * Stub implementation for OpticalFlow.
 * All it does is assign two values: 666.f and 333.f to output_image
 */
template<typename ComputeBackend>
class DLL_PUBLIC OpticalFlowStub : public OpticalFlowAdapter<ComputeBackend> {
  using StorageBackend = typename OpticalFlowAdapter<ComputeBackend>::StorageBackend;
 public:
  explicit OpticalFlowStub(const OpticalFlowParams &params) :
          OpticalFlowAdapter<ComputeBackend>(params) {}

  void Init(OpticalFlowParams &/*params*/) override {}

  void Prepare(size_t /*width*/, size_t /*height*/) override {}

  TensorShape<DynamicDimensions> CalcOutputShape(int /*height*/, int /*width*/) override {
    return {2, 3, 4};
  }


  void CalcOpticalFlow(TensorView<StorageBackend, const uint8_t, 3> reference_image,
                       TensorView<StorageBackend, const uint8_t, 3> input_image,
                       TensorView<StorageBackend, float, 3> output_image,
                       TensorView<StorageBackend, const float, 3> external_hints) override;


  static constexpr float kStubValue = 666.f;  /// Stub output value for Optical Flow
};


template<>
inline void OpticalFlowStub<ComputeCPU>::CalcOpticalFlow(
        dali::TensorView<dali::StorageCPU, const uint8_t, 3> reference_image,
        dali::TensorView<dali::StorageCPU, const uint8_t, 3> input_image,
        dali::TensorView<dali::StorageCPU, float, 3> output_image,
        dali::TensorView<dali::StorageCPU, const float, 3> external_hints) {
  auto ptr = output_image.data;
  ptr[0] = kStubValue;
  ptr[1] = kStubValue / 2;
}


template<>
inline void OpticalFlowStub<ComputeGPU>::CalcOpticalFlow(
        dali::TensorView<dali::StorageGPU, const uint8_t, 3> reference_image,
        dali::TensorView<dali::StorageGPU, const uint8_t, 3> input_image,
        dali::TensorView<dali::StorageGPU, float, 3> output_image,
        dali::TensorView<dali::StorageGPU, const float, 3> external_hints) {
  auto ptr = output_image.data;
  std::vector<float> data = {kStubValue, kStubValue / 2};
  CUDA_CALL(cudaMemcpy(ptr, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
}


}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_OPERATORS_SEQUENCE_OPTICAL_FLOW_OPTICAL_FLOW_ADAPTER_OPTICAL_FLOW_STUB_H_
