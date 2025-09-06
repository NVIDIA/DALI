// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/imgcodec/util/nvimagecodec_types.h"

namespace dali {
namespace imgcodec {

NvImageCodecInstance NvImageCodecInstance::Create(
    const nvimgcodecInstanceCreateInfo_t* instance_create_info) {
  NvImageCodecInstance ret;
  CHECK_NVIMGCODEC(nvimgcodecInstanceCreate(&ret.handle_, instance_create_info));
  return ret;
}

void NvImageCodecInstance::DestroyHandle(nvimgcodecInstance_t handle) {
  nvimgcodecInstanceDestroy(handle);
}

NvImageCodecDecoder NvImageCodecDecoder::Create(nvimgcodecInstance_t instance,
                                                const nvimgcodecExecutionParams_t* exec_params,
                                                const std::string& opts) {
  NvImageCodecDecoder ret;
  CHECK_NVIMGCODEC(nvimgcodecDecoderCreate(instance, &ret.handle_, exec_params, opts.c_str()));
  return ret;
}

void NvImageCodecDecoder::DestroyHandle(nvimgcodecDecoder_t handle) {
  nvimgcodecDecoderDestroy(handle);
}

NvImageCodecCodeStream NvImageCodecCodeStream::FromHostMem(nvimgcodecInstance_t instance,
                                                           const void *data, size_t length) {
  NvImageCodecCodeStream ret;
  CHECK_NVIMGCODEC(nvimgcodecCodeStreamCreateFromHostMem(
      instance, &ret.handle_, static_cast<const unsigned char*>(data), length));
  return ret;
}

void NvImageCodecCodeStream::DestroyHandle(nvimgcodecCodeStream_t handle) {
  nvimgcodecCodeStreamDestroy(handle);
}

NvImageCodecImage NvImageCodecImage::Create(nvimgcodecInstance_t instance,
                                            const nvimgcodecImageInfo_t *image_info) {
  NvImageCodecImage ret;
  CHECK_NVIMGCODEC(nvimgcodecImageCreate(instance, &ret.handle_, image_info));
  return ret;
}


void NvImageCodecImage::DestroyHandle(nvimgcodecImage_t handle) {
  nvimgcodecImageDestroy(handle);
}


}  // namespace imgcodec
}  // namespace dali
