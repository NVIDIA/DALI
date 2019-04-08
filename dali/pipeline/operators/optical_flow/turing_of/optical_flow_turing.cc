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

#include <dlfcn.h>

#include "dali/pipeline/operators/optical_flow/turing_of/optical_flow_turing.h"

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
                                     size_t height, size_t channels, DALIImageType image_type,
                                     cudaStream_t stream) :
        OpticalFlowAdapter<kernels::ComputeGPU>(params), width_(width), height_(height),
        channels_(channels), device_(), context_(), stream_(stream), image_type_(image_type) {
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
  VerifySupport(turing_of_.nvOFInit(of_handle_, &init_params_));

  inbuf_.reset(
          new OpticalFlowBuffer(of_handle_, width_, height_, turing_of_, NV_OF_BUFFER_USAGE_INPUT,
                                NV_OF_BUFFER_FORMAT_ABGR8));
  refbuf_.reset(
          new OpticalFlowBuffer(of_handle_, width_, height_, turing_of_, NV_OF_BUFFER_USAGE_INPUT,
                                NV_OF_BUFFER_FORMAT_ABGR8));
  outbuf_.reset(
          new OpticalFlowBuffer(of_handle_, (width_ + 3) / 4, (height_ + 3) / 4, turing_of_,
                                NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2));
  if (of_params_.enable_external_hints) {
    hintsbuf_.reset(
            new OpticalFlowBuffer(of_handle_, (width_ + 3) / 4, (height_ + 3) / 4, turing_of_,
                                  NV_OF_BUFFER_USAGE_HINT, NV_OF_BUFFER_FORMAT_SHORT2));
  }
}


OpticalFlowTuring::~OpticalFlowTuring() {
  inbuf_.reset(nullptr);
  refbuf_.reset(nullptr);
  outbuf_.reset(nullptr);
  if (of_params_.enable_external_hints) {
    hintsbuf_.reset(nullptr);
  }
  auto err = turing_of_.nvOFDestroy(of_handle_);
  if (err != NV_OF_SUCCESS) {
    // Failing to destroy OF leads to significant GPU resource leak,
    // so we'd rather terminate than live with this
    std::terminate();
  }
}


using dali::kernels::TensorView;
using dali::kernels::StorageGPU;


void OpticalFlowTuring::CalcOpticalFlow(
        TensorView<StorageGPU, const uint8_t, 3> reference_image,
        TensorView<StorageGPU, const uint8_t, 3> input_image,
        TensorView<StorageGPU, float, 3> output_image,
        TensorView<StorageGPU, const float, 3> external_hints) {
  if (of_params_.enable_external_hints) {
    DALI_ENFORCE(external_hints.shape == output_image.shape,
                 "If external hint are used, shape must match against output_image");
  } else {
    DALI_ENFORCE(external_hints.shape == kernels::TensorShape<3>(),
            "If external hints aren't used, shape must be empty");
  }
  switch (image_type_) {
    case DALI_BGR:
      kernel::BgrToRgba(input_image.data, reinterpret_cast<uint8_t *>(inbuf_->GetPtr()),
                        inbuf_->GetStride().x, width_, height_, stream_);
      kernel::BgrToRgba(reference_image.data, reinterpret_cast<uint8_t *>(refbuf_->GetPtr()),
                        refbuf_->GetStride().x, width_, height_, stream_);
      break;
    case DALI_RGB:
      kernel::RgbToRgba(input_image.data, reinterpret_cast<uint8_t *>(inbuf_->GetPtr()),
                        inbuf_->GetStride().x, width_, height_, stream_);
      kernel::RgbToRgba(reference_image.data, reinterpret_cast<uint8_t *>(refbuf_->GetPtr()),
                        refbuf_->GetStride().x, width_, height_, stream_);
      break;
    case DALI_GRAY:
      kernel::Gray(input_image.data, reinterpret_cast<uint8_t *>(inbuf_->GetPtr()),
                   inbuf_->GetStride().x, width_, height_, stream_);
      kernel::Gray(reference_image.data, reinterpret_cast<uint8_t *>(refbuf_->GetPtr()),
                   refbuf_->GetStride().x, width_, height_, stream_);
      break;
    default:
      DALI_FAIL("Provided image type not supported");
  }

  auto in_params = GenerateExecuteInParams(inbuf_->GetHandle(), refbuf_->GetHandle(),
                                           of_params_.enable_external_hints ? hintsbuf_->GetHandle()
                                                                            : nullptr);
  auto out_params = GenerateExecuteOutParams(outbuf_->GetHandle());
  TURING_OF_API_CALL(turing_of_.nvOFExecute(of_handle_, &in_params, &out_params));

  kernel::DecodeFlowComponents(reinterpret_cast<int16_t *>(outbuf_->GetPtr()), output_image.data,
                               outbuf_->GetStride().x, outbuf_->GetDescriptor().width,
                               outbuf_->GetDescriptor().height, stream_);
}


void OpticalFlowTuring::SetInitParams(dali::optical_flow::OpticalFlowParams api_params) {
  init_params_.width = static_cast<uint32_t>(width_);
  init_params_.height = static_cast<uint32_t>(height_);

  if (api_params.grid_size == VectorGridSize::SIZE_4) {
    init_params_.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init_params_.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_4;
  } else {
    init_params_.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_UNDEFINED;
    init_params_.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
  }

  init_params_.mode = NV_OF_MODE_OPTICALFLOW;

  if (api_params.perf_quality_factor >= 0.0 && api_params.perf_quality_factor < 0.375f) {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_SLOW;
  } else if (api_params.perf_quality_factor < 0.75f) {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
  } else if (api_params.perf_quality_factor <= 1.0f) {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_FAST;
  } else {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_UNDEFINED;
  }

  init_params_.enableExternalHints = of_params_.enable_external_hints ? NV_OF_TRUE : NV_OF_FALSE;
  init_params_.enableOutputCost = NV_OF_FALSE;
  init_params_.hPrivData = NULL;
}


NV_OF_EXECUTE_INPUT_PARAMS OpticalFlowTuring::GenerateExecuteInParams
        (NvOFGPUBufferHandle in_handle, NvOFGPUBufferHandle ref_handle,
         NvOFGPUBufferHandle hints_handle) {
  NV_OF_EXECUTE_INPUT_PARAMS params;
  params.inputFrame = in_handle;
  params.referenceFrame = ref_handle;
  params.disableTemporalHints = of_params_.enable_temporal_hints ? NV_OF_FALSE : NV_OF_TRUE;
  params.externalHints = hints_handle;
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
  if (!handle) {
    throw unsupported_exception("Failed to load TuringOF library: " + std::string(dlerror()));
  }

  // Pointer to initialization function
  NV_OF_STATUS (*init)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *);

  init = (decltype(init)) dlsym(handle, kInitSymbol.c_str());
  DALI_ENFORCE(init, "Failed to find symbol " + kInitSymbol + ": " + std::string(dlerror()));

  TURING_OF_API_CALL((*init)(NV_OF_API_VERSION, &turing_of_));
}

}  // namespace optical_flow
}  // namespace dali
