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

#include <dlfcn.h>
#include <vector>

#include "dali/operators/sequence/optical_flow/optical_flow_impl/optical_flow_impl.h"
#include "dali/core/device_guard.h"

namespace dali {
namespace optical_flow {

namespace {

NV_OF_STATUS VerifySupport(NV_OF_STATUS status) {
  switch (status) {
    case NV_OF_ERR_OF_NOT_AVAILABLE:
    case NV_OF_ERR_UNSUPPORTED_DEVICE:
      throw unsupported_exception("Feature unsupported");
    default:
      return status;
  }
}

}  // namespace

std::vector<uint32_t> GetCapabilities(PFNNVOFGETCAPS get_cap, NvOFHandle handle, NV_OF_CAPS cap) {
  uint32_t elm_num = 0;
  CUDA_CALL(get_cap(handle, cap, nullptr, &elm_num));
  std::vector<uint32_t> ret;
  ret.resize(elm_num);
  CUDA_CALL(get_cap(handle, cap, ret.data(), &elm_num));
  return ret;
}

void IsGridSupported(const std::vector<uint32_t> &caps, int grid_val, const std::string grid_kind) {
  bool is_grid_supported = false;
  for (auto v : caps) {
    if (v == static_cast<uint32_t>(grid_val)) {
      is_grid_supported = true;
      break;
    }
  }
  DALI_ENFORCE(is_grid_supported, make_string(grid_kind, " grid size: ", grid_val,
               " is not supported, supported are: ", caps));
}

void IsSizeSupported(std::vector<uint32_t> &caps_min, std::vector<uint32_t> &caps_max, int size,
                     std::string size_kind) {
  DALI_ENFORCE(caps_min[0] <= static_cast<uint32_t>(size) &&
               caps_max[0] >= static_cast<uint32_t>(size), make_string(size_kind, " size ", size,
               " should be between ", caps_min[0], " and ", caps_max[0]));
}

NV_OF_OUTPUT_VECTOR_GRID_SIZE OutGridSizeToEnum(int out_grid) {
  switch (out_grid) {
    case 1:
      return NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;
    case 2:
      return NV_OF_OUTPUT_VECTOR_GRID_SIZE_2;
    case 4:
      return NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    default:
      return NV_OF_OUTPUT_VECTOR_GRID_SIZE_UNDEFINED;
  }
}

NV_OF_HINT_VECTOR_GRID_SIZE HintGridSizeToEnum(int grid_hint) {
  switch (grid_hint) {
    case 1:
      return NV_OF_HINT_VECTOR_GRID_SIZE_1;
    case 2:
      return NV_OF_HINT_VECTOR_GRID_SIZE_2;
    case 4:
      return NV_OF_HINT_VECTOR_GRID_SIZE_4;
    case 8:
      return NV_OF_HINT_VECTOR_GRID_SIZE_8;
    default:
      return NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
  }
}

OpticalFlowImpl::OpticalFlowImpl(const dali::optical_flow::OpticalFlowParams &params, size_t width,
                                 size_t height, size_t channels, DALIImageType image_type,
                                 int device_id, cudaStream_t stream) :
        OpticalFlowAdapter<ComputeGPU>(params), width_(width), height_(height),
        channels_(channels), out_grid_size_(params.out_grid_size),
        hint_grid_size_(params.hint_grid_size), device_id_(device_id),
        context_(), stream_(stream), image_type_(image_type) {
}

void OpticalFlowImpl::Init(dali::optical_flow::OpticalFlowParams &params) {
  DALI_ENFORCE(channels_ == 1 || channels_ == 3 || channels_ == 4);
  DALI_ENFORCE(cuInitChecked(), "Failed to initialize driver");

  lib_handle_ = ofDriverHandle(LoadOpticalFlow("libnvidia-opticalflow.so"),
                               [](DLLDRIVER lib_handle) {
                                 dlclose(lib_handle);
                               });

  SetInitParams(params);
  DeviceGuard g(device_id_);
  CUDA_CALL(cuCtxGetCurrent(&context_));
  CreateOf();

  auto grid_sizes = GetCapabilities(of_inst_.nvOFGetCaps, of_handle_,
                                    NV_OF_CAPS_SUPPORTED_HINT_GRID_SIZES);

  auto hint_sizes = GetCapabilities(of_inst_.nvOFGetCaps, of_handle_,
                                    NV_OF_CAPS_SUPPORTED_HINT_GRID_SIZES);

  width_min_ = GetCapabilities(of_inst_.nvOFGetCaps, of_handle_, NV_OF_CAPS_WIDTH_MIN);
  height_min_ = GetCapabilities(of_inst_.nvOFGetCaps, of_handle_, NV_OF_CAPS_HEIGHT_MIN);
  width_max_ = GetCapabilities(of_inst_.nvOFGetCaps, of_handle_, NV_OF_CAPS_WIDTH_MAX);
  height_max_ = GetCapabilities(of_inst_.nvOFGetCaps, of_handle_, NV_OF_CAPS_HEIGHT_MAX);

  if (width_min_.size()) default_init_params_.width = width_min_[0];
  if (height_min_.size()) default_init_params_.height = height_min_[0];
  if (grid_sizes.size()) default_init_params_.outGridSize = OutGridSizeToEnum(grid_sizes[0]);

  IsGridSupported(grid_sizes, out_grid_size_, "Output");
  IsGridSupported(hint_sizes, hint_grid_size_, "Hint");
  IsSizeSupported(width_min_, width_max_, width_, "Width");
  IsSizeSupported(height_min_, height_max_, height_, "Height");

  CUDA_CALL(of_inst_.nvOFSetIOCudaStreams(of_handle_, stream_, stream_));
  auto status = VerifySupport(of_inst_.nvOFInit(of_handle_, &init_params_));
  CUDA_CALL(status);
  is_initialized_ = true;

  CreateBuffers();
}

void OpticalFlowImpl::CreateBuffers() {
  inbuf_.reset(
          new OpticalFlowBuffer(of_handle_, width_, height_, of_inst_, NV_OF_BUFFER_USAGE_INPUT,
                                NV_OF_BUFFER_FORMAT_ABGR8));
  refbuf_.reset(
          new OpticalFlowBuffer(of_handle_, width_, height_, of_inst_, NV_OF_BUFFER_USAGE_INPUT,
                                NV_OF_BUFFER_FORMAT_ABGR8));
  outbuf_.reset(
          new OpticalFlowBuffer(of_handle_, div_ceil(width_, out_grid_size_),
                                div_ceil(height_, out_grid_size_), of_inst_,
                                NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2));
  if (of_params_.enable_external_hints) {
    hintsbuf_.reset(
            new OpticalFlowBuffer(of_handle_, div_ceil(width_, hint_grid_size_),
                                  div_ceil(height_, hint_grid_size_), of_inst_,
                                  NV_OF_BUFFER_USAGE_HINT, NV_OF_BUFFER_FORMAT_SHORT2));
  }
}

void OpticalFlowImpl::DestroyBuffers() {
  inbuf_.reset(nullptr);
  refbuf_.reset(nullptr);
  outbuf_.reset(nullptr);
  if (of_params_.enable_external_hints) {
    hintsbuf_.reset(nullptr);
  }
}

void OpticalFlowImpl::CreateOf() {
  DeviceGuard g(device_id_);
  auto ret = of_inst_.nvCreateOpticalFlowCuda(context_, &of_handle_);
  if (ret != NV_OF_SUCCESS) {
    throw unsupported_exception(
        "Failed to create Optical Flow context: Verify that your device supports Optical Flow.");
  }
}

void OpticalFlowImpl::DestroyOf() {
  if (of_handle_) {
    // it is not possible to release all driver allocated data structures by calling nvOFDestroy
    // if nvOFInit was not successfully called before, so run init with the default args just
    // to successfully call nvOFDestroy
    if (!is_initialized_) {
      of_inst_.nvOFInit(of_handle_, &default_init_params_);
    }
    auto err = of_inst_.nvOFDestroy(of_handle_);
    // unload lib no matter if it was successful
    if (err != NV_OF_SUCCESS) {
      // Failing to destroy OF leads to significant GPU resource leak,
      // so we'd rather terminate than live with this
      std::cerr << "Fatal error: failed to destroy optical flow" << std::endl;
      std::terminate();
    }
  }
  of_handle_ = nullptr;
  is_initialized_ = false;
}

void OpticalFlowImpl::Prepare(size_t width, size_t height) {
  if (width == width_ && height == height_)
    return;
  if (width != width_) {
    width_ = width;
    IsSizeSupported(width_min_, width_max_, width_, "Width");
  }
  if (height != height_) {
    height_ = height;
    IsSizeSupported(height_min_, height_max_, height_, "Height");
  }

  init_params_.width = static_cast<uint32_t>(width_);
  init_params_.height = static_cast<uint32_t>(height_);

  DestroyBuffers();
  DestroyOf();

  CreateOf();

  CUDA_CALL(of_inst_.nvOFSetIOCudaStreams(of_handle_, stream_, stream_));
  auto status = VerifySupport(of_inst_.nvOFInit(of_handle_, &init_params_));
  CUDA_CALL(status);
  is_initialized_ = true;

  CreateBuffers();
}

OpticalFlowImpl::~OpticalFlowImpl() {
  DestroyBuffers();
  DestroyOf();
}


using dali::TensorView;
using dali::StorageGPU;


void OpticalFlowImpl::CalcOpticalFlow(
        TensorView<StorageGPU, const uint8_t, 3> reference_image,
        TensorView<StorageGPU, const uint8_t, 3> input_image,
        TensorView<StorageGPU, float, 3> output_image,
        TensorView<StorageGPU, const float, 3> external_hints) {
  if (of_params_.enable_external_hints) {
    DALI_ENFORCE(external_hints.shape == output_image.shape,
                 "If external hint are used, shape must match against output_image");
  } else {
    DALI_ENFORCE(external_hints.shape == TensorShape<3>(),
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
                                           of_params_.enable_external_hints
                                           ? hintsbuf_->GetHandle()
                                           : nullptr);
  auto out_params = GenerateExecuteOutParams(outbuf_->GetHandle());
  CUDA_CALL(of_inst_.nvOFExecute(of_handle_, &in_params, &out_params));


  kernel::DecodeFlowComponents(reinterpret_cast<int16_t *>(outbuf_->GetPtr()), output_image.data,
                               outbuf_->GetStride().x, outbuf_->GetDescriptor().width,
                               outbuf_->GetDescriptor().height, stream_);
}

void OpticalFlowImpl::SetInitParams(dali::optical_flow::OpticalFlowParams api_params) {
  init_params_ = {};
  default_init_params_ = {};
  init_params_.width = static_cast<uint32_t>(width_);
  init_params_.height = static_cast<uint32_t>(height_);

  init_params_.outGridSize = OutGridSizeToEnum(api_params.out_grid_size);
  init_params_.enableExternalHints = of_params_.enable_external_hints ? NV_OF_TRUE : NV_OF_FALSE;
  // in driver 520.x.y only NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED can be provided if the hint
  // is not used
  if (of_params_.enable_external_hints) {
    init_params_.hintGridSize = HintGridSizeToEnum(api_params.hint_grid_size);
  } else {
    init_params_.hintGridSize = HintGridSizeToEnum(-1);
  }

  default_init_params_.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;

  init_params_.mode = NV_OF_MODE_OPTICALFLOW;
  default_init_params_.mode = NV_OF_MODE_OPTICALFLOW;

  if (api_params.perf_quality_factor >= 0.0 && api_params.perf_quality_factor < 0.375f) {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_SLOW;
  } else if (api_params.perf_quality_factor < 0.75f) {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
  } else if (api_params.perf_quality_factor <= 1.0f) {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_FAST;
  } else {
    init_params_.perfLevel = NV_OF_PERF_LEVEL_UNDEFINED;
  }
  default_init_params_.perfLevel = NV_OF_PERF_LEVEL_SLOW;

  init_params_.enableOutputCost = NV_OF_FALSE;
  init_params_.hPrivData = NULL;
  init_params_.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;

  default_init_params_.enableExternalHints = NV_OF_FALSE;
  default_init_params_.enableOutputCost = NV_OF_FALSE;
  default_init_params_.hPrivData = NULL;
  default_init_params_.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
}


NV_OF_EXECUTE_INPUT_PARAMS OpticalFlowImpl::GenerateExecuteInParams
        (NvOFGPUBufferHandle in_handle, NvOFGPUBufferHandle ref_handle,
         NvOFGPUBufferHandle hints_handle) {
  // zeroing required for padding, padding2 and hPrivData
  NV_OF_EXECUTE_INPUT_PARAMS params = {};
  params.inputFrame = in_handle;
  params.referenceFrame = ref_handle;
  params.disableTemporalHints = of_params_.enable_temporal_hints ? NV_OF_FALSE : NV_OF_TRUE;
  params.externalHints = hints_handle;
  return params;
}


NV_OF_EXECUTE_OUTPUT_PARAMS OpticalFlowImpl::GenerateExecuteOutParams
        (NvOFGPUBufferHandle out_handle) {
  NV_OF_EXECUTE_OUTPUT_PARAMS params;
  params.outputBuffer = out_handle;
  params.outputCostBuffer = nullptr;
  params.hPrivData = NULL;
  return params;
}


OpticalFlowImpl::DLLDRIVER OpticalFlowImpl::LoadOpticalFlow
        (const std::string &library_path) {
  const std::string library_path_1 = library_path + ".1";
  DLLDRIVER lib_handle = dlopen(library_path_1.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (!lib_handle) {
    lib_handle = dlopen(library_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
    if (!lib_handle) {
      throw unsupported_exception("Failed to load OF library: " + std::string(dlerror()));
    }
  }

  // Pointer to initialization function
  NV_OF_STATUS (*init)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *);

  init = (decltype(init)) dlsym(lib_handle, kInitSymbol.c_str());
  DALI_ENFORCE(init, "Failed to find symbol " + kInitSymbol + ": " + std::string(dlerror()));

  CUDA_CALL((*init)(NV_OF_API_VERSION, &of_inst_));
  return lib_handle;
}

}  // namespace optical_flow
}  // namespace dali
