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

#include "dali/aux/optical_flow/turing_of/optical_flow_turing.h"
#include <dlfcn.h>


namespace dali {
namespace optical_flow {

namespace {

void VerifySupport(NV_OF_STATUS status) {
  switch (status) {
    case NV_OF_SUCCESS:
      return;
    case NV_OF_ERR_OF_NOT_AVAILABLE:
    case NV_OF_ERR_UNSUPPORTED_DEVICE:
      throw unsupported_exception();
    default:
      DALI_FAIL("Optical flow failed, code: " + std::to_string(status));
  }
}

}  // namespace

OpticalFlowTuring::OpticalFlowTuring(dali::optical_flow::OpticalFlowParams params, size_t width,
                                     size_t height, size_t channels, cudaStream_t stream) :
        OpticalFlowAdapter<kernels::ComputeGPU>(params), width_(width), height_(height),
        channels_(channels), device_(), context_(), stream_(stream) {
  DALI_ENFORCE(channels_ == 1 || channels_ == 3 || channels_ == 4);
  DALI_ENFORCE(cuInitChecked(), "Failed to initialize driver");

  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));
  CUDA_CALL(cuDeviceGet(&device_, device_id));
  LoadTuringOpticalFlow("libnvidia-opticalflow.so");

  SetInitParams(params);

  context_ = CUContext(device_);
  CUcontext ctx = context_;

  TURING_OF_API_CALL(turing_of_.nvCreateOpticalFlowCuda(ctx, &of_handle_));
  TURING_OF_API_CALL(turing_of_.nvOFSetIOCudaStreams(of_handle_, stream_, stream_));
  VerifySupport(turing_of_.nvOFInit(of_handle_, &of_params_));

  inbuf_.reset(
          new OpticalFlowBuffer(of_handle_, width_, height_, turing_of_, NV_OF_BUFFER_USAGE_INPUT,
                                NV_OF_BUFFER_FORMAT_ABGR8));
  refbuf_.reset(
          new OpticalFlowBuffer(of_handle_, width_, height_, turing_of_, NV_OF_BUFFER_USAGE_INPUT,
                                NV_OF_BUFFER_FORMAT_ABGR8));
  outbuf_.reset(
          new OpticalFlowBuffer(of_handle_, (width_ + 3) / 4, (height_ + 3) / 4, turing_of_,
                                NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2));
}


OpticalFlowTuring::~OpticalFlowTuring() {
  inbuf_.reset(nullptr);
  refbuf_.reset(nullptr);
  outbuf_.reset(nullptr);
  auto err = turing_of_.nvOFDestroy(of_handle_);
  if (err != NV_OF_SUCCESS) {
    // Failing to destroy OF leads to significant GPU resource leak,
    // so we'd rather terminate than live with this
    std::terminate();
  }
}


void OpticalFlowTuring::CalcOpticalFlow(
        dali::kernels::TensorView<dali::kernels::StorageGPU, const uint8_t, 3> reference_image,
        dali::kernels::TensorView<dali::kernels::StorageGPU, const uint8_t, 3> input_image,
        dali::kernels::TensorView<dali::kernels::StorageGPU, float, 3> output_image,
        dali::kernels::TensorView<dali::kernels::StorageGPU, const float, 3> external_hints) {
  kernel::RgbToRgba(input_image.data, reinterpret_cast<uint8_t *>(inbuf_->GetPtr()),
                    inbuf_->GetStride().x, width_, height_, stream_);
  kernel::RgbToRgba(reference_image.data, reinterpret_cast<uint8_t *>(refbuf_->GetPtr()),
                    refbuf_->GetStride().x, width_, height_, stream_);

  auto in_params = GenerateExecuteInParams(inbuf_->GetHandle(), refbuf_->GetHandle());
  auto out_params = GenerateExecuteOutParams(outbuf_->GetHandle());
  TURING_OF_API_CALL(turing_of_.nvOFExecute(of_handle_, &in_params, &out_params));

  kernel::DecodeFlowComponents(reinterpret_cast<int16_t *>(outbuf_->GetPtr()), output_image.data,
                               outbuf_->GetStride().x, outbuf_->GetDescriptor().width,
                               outbuf_->GetDescriptor().height, stream_);
}


void OpticalFlowTuring::SetInitParams(dali::optical_flow::OpticalFlowParams api_params) {
  of_params_.width = static_cast<uint32_t>(width_);
  of_params_.height = static_cast<uint32_t>(height_);

  if (api_params.grid_size == VectorGridSize::SIZE_4) {
    of_params_.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    of_params_.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_4;
  } else {
    of_params_.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_UNDEFINED;
    of_params_.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
  }

  of_params_.mode = NV_OF_MODE_OPTICALFLOW;

  if (api_params.perf_quality_factor >= 0.0 && api_params.perf_quality_factor < 0.375f) {
    of_params_.perfLevel = NV_OF_PERF_LEVEL_SLOW;
  } else if (api_params.perf_quality_factor < 0.75f) {
    of_params_.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
  } else if (api_params.perf_quality_factor <= 1.0f) {
    of_params_.perfLevel = NV_OF_PERF_LEVEL_FAST;
  } else {
    of_params_.perfLevel = NV_OF_PERF_LEVEL_UNDEFINED;
  }

  of_params_.enableExternalHints = api_params.enable_hints ? NV_OF_TRUE : NV_OF_FALSE;
  of_params_.enableOutputCost = NV_OF_FALSE;
  of_params_.hPrivData = NULL;
}


NV_OF_EXECUTE_INPUT_PARAMS OpticalFlowTuring::GenerateExecuteInParams
        (NvOFGPUBufferHandle in_handle, NvOFGPUBufferHandle ref_handle) {
  NV_OF_EXECUTE_INPUT_PARAMS params;
  params.inputFrame = in_handle;
  params.referenceFrame = ref_handle;
  params.disableTemporalHints = NV_OF_TRUE;
  params.externalHints = nullptr;
  params.padding = 0;
  params.hPrivData = NULL;
  return params;
}


NV_OF_EXECUTE_OUTPUT_PARAMS OpticalFlowTuring::GenerateExecuteOutParams
        (NvOFGPUBufferHandle out_handle) {
  NV_OF_EXECUTE_OUTPUT_PARAMS params;
  params.outputBuffer = out_handle;
  params.outputCostBuffer = nullptr;
  params.hPrivData = NULL;
  return params;
}


void OpticalFlowTuring::LoadTuringOpticalFlow(const std::string &library_path) {
  auto handle = dlopen(library_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
  DALI_ENFORCE(handle, "Failed to load TuringOF library: " + std::string(dlerror()));

  // Pointer to initialization function
  NV_OF_STATUS (*init)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *);

  init = (decltype(init)) dlsym(handle, kInitSymbol.c_str());
  DALI_ENFORCE(init, "Failed to find symbol " + kInitSymbol + ": " + std::string(dlerror()));

  TURING_OF_API_CALL((*init)(NV_OF_API_VERSION, &turing_of_));
}

}  // namespace optical_flow
}  // namespace dali
