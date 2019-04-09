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

#ifndef DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_TURING_H_
#define DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_TURING_H_

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include "dali/util/cucontext.h"
#include "dali/util/custream.h"
#include "nvOpticalFlowCuda.h"
#include "nvOpticalFlowCommon.h"
#include "dali/common.h"
#include "dali/pipeline/operators/optical_flow/optical_flow_adapter/optical_flow_adapter.h"
#include "dali/pipeline/operators/optical_flow/turing_of/optical_flow_buffer.h"

namespace dali {
namespace optical_flow {
namespace kernel {

/**
 * Converts RGB image to RGBA and puts it in strided memory
 * @param input
 * @param output User is responsible for allocation of output
 * @param pitch Stride within output memory layout. In bytes.
 * @param width_px In pixels.
 * @param height
 * @param stream Stream, in which kernel is called
 */
DLL_PUBLIC void
RgbToRgba(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
          cudaStream_t stream = 0);

/**
 * Convert BGR image to RGBA and puts it in strided memory
 * @param input
 * @param output User is responsible for allocation of output
 * @param pitch Stride within output memory layout. In bytes.
 * @param width_px In pixels.
 * @param height
 * @param stream Stream, in which kernel is called
 */
DLL_PUBLIC void
BgrToRgba(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
          cudaStream_t stream = 0);

/**
 * Puts grayscale data in strided memory
 * @param input
 * @param output User is responsible for allocation of output
 * @param pitch Stride within output memory layout. In bytes.
 * @param width_px In pixels.
 * @param height
 * @param stream Stream, in which kernel is called
 */
DLL_PUBLIC void
Gray(const uint8_t *input, uint8_t *output, size_t pitch, size_t width_px, size_t height,
     cudaStream_t stream = 0);

/**
 * Decodes components of flow vector and unstrides memory
 * @param input
 * @param output User is responsible for allocation of output
 * @param pitch Stride within input memory layout. In bytes.
 * @param width_px In pixels.
 * @param height
 * @param stream Stream, in which kernel is called
 */
DLL_PUBLIC void
DecodeFlowComponents(const int16_t *input, float *output, size_t pitch, size_t width_px,
                     size_t height, cudaStream_t stream = 0);

/**
 * Encode flow components and put in strided memory (for external hints)
 * @param input
 * @param output User is responsible for allocation of output
 * @param pitch Stride within input memory layout. In bytes.
 * @param width_px In pixels.
 * @param height
 * @param stream Stream, in which kernel is called
 */
DLL_PUBLIC void
EncodeFlowComponents(const float *input, int16_t *output, size_t pitch, size_t width_px,
                     size_t height, cudaStream_t stream = 0);


inline __host__ __device__ float decode_flow_component(int16_t value) {
  return value * (1 / 32.f);
}


inline __host__ __device__ int16_t encode_flow_component(float value) {
  return static_cast<int16_t>(value * 32.f);
}

}  // namespace kernel

class DLL_PUBLIC OpticalFlowTuring : public OpticalFlowAdapter<kernels::ComputeGPU> {
 public:
  OpticalFlowTuring(OpticalFlowParams params, size_t width, size_t height, size_t channels,
                    DALIImageType image_type, cudaStream_t stream = 0);


  virtual ~OpticalFlowTuring();


  kernels::TensorShape<kernels::DynamicDimensions> GetOutputShape() override {
    auto sz = init_params_.outGridSize;
    // There are 2 flow vector components: (x, y)
    return {static_cast<int>(height_ + sz - 1) / sz, static_cast<int>(width_ + sz - 1) / sz, 2};
  }


  void CalcOpticalFlow(TensorView<StorageBackend, const uint8_t, 3> reference_image,
                       TensorView<StorageBackend, const uint8_t, 3> input_image,
                       TensorView<StorageBackend, float, 3> output_image,
                       TensorView<StorageBackend, const float, 3> external_hints =
                       TensorView<StorageBackend, const float, 3>()) override;


 private:
  void SetInitParams(OpticalFlowParams api_params);


  NV_OF_EXECUTE_INPUT_PARAMS
  GenerateExecuteInParams(NvOFGPUBufferHandle in_handle, NvOFGPUBufferHandle ref_handle,
                          NvOFGPUBufferHandle hints_handle = nullptr);


  NV_OF_EXECUTE_OUTPUT_PARAMS GenerateExecuteOutParams(NvOFGPUBufferHandle out_handle);


  void LoadTuringOpticalFlow(const std::string &library_path);


  const std::string kInitSymbol = "NvOFAPICreateInstanceCuda";

  const size_t width_, height_, channels_;
  CUdevice device_;
  dali::CUContext context_;
  cudaStream_t stream_;
  NvOFHandle of_handle_;
  NV_OF_CUDA_API_FUNCTION_LIST turing_of_;
  NV_OF_INIT_PARAMS init_params_;
  std::unique_ptr<OpticalFlowBuffer> inbuf_, refbuf_, outbuf_, hintsbuf_;
  DALIImageType image_type_;
};

}  // namespace optical_flow
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_TURING_H_

