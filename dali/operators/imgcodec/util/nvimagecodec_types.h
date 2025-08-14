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

#ifndef DALI_OPERATORS_IMGCODEC_UTIL_NVIMAGECODEC_TYPES_H_
#define DALI_OPERATORS_IMGCODEC_UTIL_NVIMAGECODEC_TYPES_H_

#include <nvimgcodec.h>
#include <sstream>
#include <string>
#include "dali/core/error_handling.h"
#include "dali/core/nvtx.h"
#include "dali/core/unique_handle.h"
#include "dali/operators/imgcodec/imgcodec.h"
#include "dali/pipeline/data/types.h"

#define CHECK_NVIMGCODEC(call)                             \
  {                                                       \
    nvimgcodecStatus_t _e = (call);                        \
    if (_e != NVIMGCODEC_STATUS_SUCCESS) {                 \
      std::stringstream _error;                           \
      _error << "nvImageCodec failure: '#" << _e << "'"; \
      throw std::runtime_error(_error.str());             \
    }                                                     \
  }

namespace dali {
namespace imgcodec {

static DALIDataType to_dali_dtype(nvimgcodecSampleDataType_t dtype) {
  switch (dtype) {
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
      return DALI_UINT8;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT8:
      return DALI_INT8;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
      return DALI_UINT16;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
      return DALI_INT16;
    case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
      return DALI_FLOAT;
    default:
      return DALI_NO_TYPE;
  }
}

static ImageInfo to_dali_img_info(const nvimgcodecImageInfo_t& info) {
  ImageInfo ret;
  ret.orientation = info.orientation;

  int num_channels = 0;
  int num_planes = info.num_planes;
  for (int p = 0; p < num_planes; p++)
    num_channels += info.plane_info[p].num_channels;
  DALI_ENFORCE(num_planes == 1 || num_channels == num_planes,
               "Only full planar or full interleaved layouts are supported");
  ret.shape = TensorShape{info.plane_info[0].height, info.plane_info[0].width, num_channels};
  return ret;
}

struct DLL_PUBLIC NvImageCodecInstance
    : public UniqueHandle<nvimgcodecInstance_t, NvImageCodecInstance> {
  DALI_INHERIT_UNIQUE_HANDLE(nvimgcodecInstance_t, NvImageCodecInstance);

  NvImageCodecInstance() = default;

  static NvImageCodecInstance Create(const nvimgcodecInstanceCreateInfo_t* instance_create_info);

  static constexpr nvimgcodecInstance_t null_handle() {
    return nullptr;
  }

  static void DestroyHandle(nvimgcodecInstance_t handle);
};

struct DLL_PUBLIC NvImageCodecDecoder
    : public UniqueHandle<nvimgcodecDecoder_t, NvImageCodecDecoder> {
  DALI_INHERIT_UNIQUE_HANDLE(nvimgcodecDecoder_t, NvImageCodecDecoder);

  NvImageCodecDecoder() = default;

  static NvImageCodecDecoder Create(nvimgcodecInstance_t instance,
                                    const nvimgcodecExecutionParams_t* exec_params,
                                    const std::string& opts);

  static constexpr nvimgcodecDecoder_t null_handle() {
    return nullptr;
  }

  static void DestroyHandle(nvimgcodecDecoder_t handle);
};

struct DLL_PUBLIC NvImageCodecCodeStream
    : public UniqueHandle<nvimgcodecCodeStream_t, NvImageCodecCodeStream> {
  DALI_INHERIT_UNIQUE_HANDLE(nvimgcodecCodeStream_t, NvImageCodecCodeStream);

  NvImageCodecCodeStream() = default;

  static NvImageCodecCodeStream FromHostMem(nvimgcodecInstance_t instance, const void* data,
                                            size_t length);

  static NvImageCodecCodeStream FromSubCodeStream(nvimgcodecCodeStream_t code_stream,
                                                  const nvimgcodecCodeStreamView_t* cs_view);

  static constexpr nvimgcodecCodeStream_t null_handle() {
    return nullptr;
  }

  static void DestroyHandle(nvimgcodecCodeStream_t handle);
};

struct DLL_PUBLIC NvImageCodecImage : public UniqueHandle<nvimgcodecImage_t, NvImageCodecImage> {
  DALI_INHERIT_UNIQUE_HANDLE(nvimgcodecImage_t, NvImageCodecImage);

  NvImageCodecImage() = default;

  static NvImageCodecImage Create(nvimgcodecInstance_t instance,
                                  const nvimgcodecImageInfo_t* image_info);

  static constexpr nvimgcodecImage_t null_handle() {
    return nullptr;
  }

  static void DestroyHandle(nvimgcodecImage_t handle);
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_OPERATORS_IMGCODEC_UTIL_NVIMAGECODEC_TYPES_H_
