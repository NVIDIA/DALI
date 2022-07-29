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

#include <string>
#include <vector>
#include "dali/imgcodec/decoders/nvjpeg2k.h"
#include "dali/core/static_switch.h"
#include "dali/operators/decoder/nvjpeg/permute_layout.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/dev_buffer.h"

namespace dali {
namespace imgcodec {

bool NvJpeg2kDecoderInstance::ParseJpeg2000Info(int id,
                                                ProcessingInfo *info,
                                                ImageSource *in,
                                                DecodeParams opts) {
  const auto &jpeg2k_stream = nvjpeg2k_streams_[id];
  if (nvjpeg2kStreamParse(nvjpeg2k_handle_, in->RawData<uint8_t>(), in->Size(),
                          0, 0, jpeg2k_stream) != NVJPEG2K_STATUS_SUCCESS)
    return false;

  nvjpeg2kImageInfo_t image_info;
  CUDA_CALL(nvjpeg2kStreamGetImageInfo(jpeg2k_stream, &image_info));

  nvjpeg2kImageComponentInfo_t comp;
  CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp, 0));
  const auto height = comp.component_height;
  const auto width = comp.component_width;
  info->bpp = comp.precision;

  for (uint32_t c = 1; c < image_info.num_components; c++) {
    CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp, c));
    if (height != comp.component_height ||
        width != comp.component_width ||
        info->bpp != comp.precision)
      return false;
  }

  info->shape = {height, width, image_info.num_components};
  info->req_nchannels = NumberOfChannels(opts.format, info->shape[2]);

  DALI_ENFORCE(info->bpp == 8 || info->bpp == 16,
               make_string("Invalid bits per pixel value: ", info->bpp));
  info->needs_processing = info->shape[2] > 1 || info->bpp == 16;
  info->pixel_size = info->bpp == 16 ? sizeof(uint16_t) : sizeof(uint8_t);
  info->pixels_count = info->shape[0] * info->shape[1];
  info->comp_size = info->pixels_count * info->pixel_size;

  return true;
}

bool NvJpeg2kDecoderInstance::DecodeImpl(int id,
                                         ProcessingInfo *info,
                                         ImageSource *in,
                                         uint8_t *out,
                                         DecodeParams opts) {
  const auto &jpeg2k_stream = nvjpeg2k_streams_[id];

  void *pixel_data[NVJPEG_MAX_COMPONENT] = {};
  size_t pitch_in_bytes[NVJPEG_MAX_COMPONENT] = {};

  for (uint32_t c = 0; c < info->shape[2]; c++) {
    pixel_data[c] = out + c * info->comp_size;
    pitch_in_bytes[c] = info->shape[1] * info->pixel_size;
  }

  nvjpeg2kImage_t output_image;
  output_image.pixel_data = pixel_data;
  output_image.pitch_in_bytes = pitch_in_bytes;
  if (info->bpp == 16) {
    info->pixel_type = DALI_UINT16;
    output_image.pixel_type = NVJPEG2K_UINT16;
  } else {  // info->bpp == 8
    info->pixel_type = DALI_UINT8;
    output_image.pixel_type = NVJPEG2K_UINT8;
  }
  output_image.num_components = info->shape[2];

  auto ret = nvjpeg2kDecode(nvjpeg2k_handle_, nvjpeg2k_decoder_,
                            jpeg2k_stream, &output_image, nvjpeg2k_cu_stream_);

  if (ret == NVJPEG2K_STATUS_SUCCESS) {
    return true;
  } else if (ret == NVJPEG2K_STATUS_BAD_JPEG || ret == NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED) {
    return false;
  } else {
    CUDA_CALL_EX(ret, in->SourceInfo());
    assert(false);
  }
}

bool NvJpeg2kDecoderInstance::ConvertData(int id,
                                          ProcessingInfo *info,
                                          uint8_t *in,
                                          uint8_t *out,
                                          DecodeParams opts) {
  if (opts.format == DALI_GRAY) {
    // Converting to Gray, dropping alpha channels if needed
    assert(info->shape[2] >= 3);
    TYPE_SWITCH(info->pixel_type, type2id, Input, (uint8_t, uint16_t), (
      PlanarRGBToGray<uint8_t, Input>(
        out, reinterpret_cast<Input*>(in), info->pixels_count,
        info->pixel_type, nvjpeg2k_cu_stream_);
    ), DALI_FAIL(make_string("Unsupported input type: ", info->pixel_type)));  // NOLINT
  } else {
    // Converting to interleaved, dropping alpha channels if needed
    assert(info->shape[2] >= 3);
    TYPE_SWITCH(info->pixel_type, type2id, Input, (uint8_t, uint16_t), (
      PlanarToInterleaved<uint8_t, Input>(
        out, reinterpret_cast<Input*>(in), info->pixels_count,
        info->req_nchannels, opts.format, info->pixel_type, nvjpeg2k_cu_stream_);
    ), DALI_FAIL(make_string("Unsupported input type: ", info->pixel_type)));  // NOLINT
  }
}

std::vector<DecodeResult> NvJpeg2kDecoderInstance::Decode(
                                  span<SampleView<GPUBackend>> out,
                                  cspan<ImageSource *> in,
                                  DecodeParams opts,
                                  cspan<ROI> rois) {
  std::vector<DecodeResult> result(in.size());
  for (int i = 0; i < in.size(); i++) {
    ProcessingInfo info;
    if (!ParseJpeg2000Info(i, &info, in[i], opts))
      continue;

    CUDA_CALL(cudaEventSynchronize(nvjpeg2k_decode_event_));

    if (!info.needs_processing) {
      result[i].success = DecodeImpl(i, &info, in[i], out[i].mutable_data<uint8_t>(), opts);
    } else {
      auto &buffer = nvjpeg2k_intermediate_buffer_;
      buffer.clear();
      buffer.resize(volume(info.shape) * info.pixel_size);
      try {
        result[i].success = DecodeImpl(i, &info, in[i], buffer.data(), opts) &&
                            ConvertData(i, &info, buffer.data(),
                                        out[i].mutable_data<uint8_t>(), opts);
      } catch (...) {
        result[i].success = false;
        result[i].exception = std::current_exception();
      }
    }

    if (result[i].success)
      CUDA_CALL(cudaEventRecord(nvjpeg2k_decode_event_, nvjpeg2k_cu_stream_));
  }
  return result;
}

}  // namespace imgcodec
}  // namespace dali
