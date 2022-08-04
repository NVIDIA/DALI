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
#include "dali/imgcodec/decoders/nvjpeg/permute_layout.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace imgcodec {

NvJpeg2000DecoderInstance::NvJpeg2000DecoderInstance(int device_id, ThreadPool *tp)
: BatchParallelDecoderImpl(device_id, tp)
, nvjpeg2k_dev_alloc_(nvjpeg_memory::GetDeviceAllocatorNvJpeg2k())
, nvjpeg2k_pin_alloc_(nvjpeg_memory::GetPinnedAllocatorNvJpeg2k())
, nvjpeg2k_decode_states_(tp->NumThreads())
, intermediate_buffers_(tp->NumThreads())
, nvjpeg2k_streams_(tp->NumThreads())
, decode_events_(tp->NumThreads()) {
  size_t device_memory_padding = any_cast<size_t>(GetParam("device_memory_padding"));
  size_t host_memory_padding = any_cast<size_t>(GetParam("host_memory_padding"));

  nvjpeg2k_handle_ = NvJpeg2kHandle(&nvjpeg2k_dev_alloc_, &nvjpeg2k_pin_alloc_);
  DALI_ENFORCE(nvjpeg2k_handle_, "NvJpeg2kHandle initalization failed");

  for (int i = 0; i < tp_->NumThreads(); i++) {
    nvjpeg2k_decode_states_[i] = NvJpeg2kDecodeState(nvjpeg2k_handle_);
    intermediate_buffers_[i].resize(device_memory_padding / 8);
    nvjpeg2k_streams_[i] = NvJpeg2kStream::Create();
    decode_events_[i] = CUDAEvent::Create(device_id);
  }

  for (auto &thread_id : tp_->GetThreadIds()) {
    if (device_memory_padding > 0) {
      nvjpeg_memory::AddBuffer<mm::memory_kind::device>(thread_id, 1024);
      nvjpeg_memory::AddBuffer<mm::memory_kind::device>(thread_id, 4 * 1024);
      nvjpeg_memory::AddBuffer<mm::memory_kind::device>(thread_id, 16 * 1024);
      nvjpeg_memory::AddBuffer<mm::memory_kind::device>(thread_id, device_memory_padding);
      nvjpeg_memory::AddBuffer<mm::memory_kind::device>(thread_id, device_memory_padding); 
    }
    if (host_memory_padding > 0) {
      nvjpeg_memory::AddBuffer<mm::memory_kind::pinned>(thread_id, host_memory_padding);
    }
  }

  // unable to call this here: (from old implementation)
  // CUDA_CALL(cudaEventRecord(nvjpeg2k_decode_event_, nvjpeg2k_cu_stream_));
}

bool NvJpeg2000DecoderInstance::ParseJpeg2000Info(ImageSource *in,
                                                  DecodeParams opts,
                                                  Context *ctx) {
  CUDA_CALL(nvjpeg2kStreamParse(nvjpeg2k_handle_, in->RawData<uint8_t>(), in->Size(),
                                0, 0, *ctx->nvjpeg2k_stream));

  nvjpeg2kImageInfo_t image_info;
  CUDA_CALL(nvjpeg2kStreamGetImageInfo(*ctx->nvjpeg2k_stream, &image_info));

  nvjpeg2kImageComponentInfo_t comp;
  CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(*ctx->nvjpeg2k_stream, &comp, 0));
  const auto height = comp.component_height;
  const auto width = comp.component_width;
  ctx->bpp = comp.precision;

  for (uint32_t c = 1; c < image_info.num_components; c++) {
    CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(*ctx->nvjpeg2k_stream, &comp, c));
    if (height != comp.component_height ||
        width != comp.component_width ||
        ctx->bpp != comp.precision)
      return false;
  }

  ctx->shape = {height, width, image_info.num_components};
  ctx->req_nchannels = NumberOfChannels(opts.format, ctx->shape[2]);

  DALI_ENFORCE(ctx->bpp == 8 || ctx->bpp == 16,
               make_string("Invalid bits per pixel value: ", ctx->bpp));
  ctx->needs_processing = ctx->shape[2] > 1 || ctx->bpp == 16;
  ctx->pixel_size = ctx->bpp == 16 ? sizeof(uint16_t) : sizeof(uint8_t);
  ctx->pixels_count = ctx->shape[0] * ctx->shape[1];
  ctx->comp_size = ctx->pixels_count * ctx->pixel_size;

  return true;
}

bool NvJpeg2000DecoderInstance::DecodeJpeg2000(ImageSource *in,
                                               uint8_t *out,
                                               DecodeParams opts,
                                               Context *ctx) {
  void *pixel_data[NVJPEG_MAX_COMPONENT] = {};
  size_t pitch_in_bytes[NVJPEG_MAX_COMPONENT] = {};
  for (uint32_t c = 0; c < ctx->shape[2]; c++) {
    pixel_data[c] = out + c * ctx->comp_size;
    pitch_in_bytes[c] = ctx->shape[1] * ctx->pixel_size;
  }

  nvjpeg2kImage_t output_image;
  output_image.pixel_data = pixel_data;
  output_image.pitch_in_bytes = pitch_in_bytes;
  if (ctx->bpp == 8) {
    ctx->pixel_type = DALI_UINT8;
    output_image.pixel_type = NVJPEG2K_UINT8;
  } else {  // ctx->bpp == 16
    ctx->pixel_type = DALI_UINT16;
    output_image.pixel_type = NVJPEG2K_UINT16;
  }
  output_image.num_components = ctx->shape[2];

  auto ret = nvjpeg2kDecode(nvjpeg2k_handle_, *ctx->nvjpeg2k_decode_state,
                            *ctx->nvjpeg2k_stream, &output_image, *ctx->cuda_stream);

  if (ret == NVJPEG2K_STATUS_SUCCESS) {
    return true;
  } else if (ret == NVJPEG2K_STATUS_BAD_JPEG || ret == NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED) {
    return false;
  } else {
    CUDA_CALL_EX(ret, in->SourceInfo());
    return false;  // unreachable
  }
}

bool NvJpeg2000DecoderInstance::ConvertData(void *in,
                                            uint8_t *out,
                                            DecodeParams opts,
                                            Context *ctx) {
  if (opts.format == DALI_GRAY) {
    // Converting to Gray, dropping alpha channels if needed
    assert(ctx->shape[2] >= 3);
    TYPE_SWITCH(ctx->pixel_type, type2id, Input, (uint8_t, uint16_t), (
      PlanarRGBToGray<uint8_t, Input>(
        out, reinterpret_cast<Input*>(in), ctx->pixels_count,
        ctx->pixel_type, *ctx->cuda_stream);
    ), DALI_FAIL(make_string("Unsupported input type: ", ctx->pixel_type)));  // NOLINT
  } else {
    // Converting to interleaved, dropping alpha channels if needed
    assert(ctx->shape[2] >= 3);
    TYPE_SWITCH(ctx->pixel_type, type2id, Input, (uint8_t, uint16_t), (
      PlanarToInterleaved<uint8_t, Input>(
        out, reinterpret_cast<Input*>(in), ctx->pixels_count,
        ctx->req_nchannels, opts.format, ctx->pixel_type, *ctx->cuda_stream);
    ), DALI_FAIL(make_string("Unsupported input type: ", ctx->pixel_type)));  // NOLINT
  }
  return true;
}

DecodeResult NvJpeg2000DecoderInstance::DecodeImplTask(int thread_idx,
                                                       cudaStream_t stream,
                                                       SampleView<GPUBackend> out,
                                                       ImageSource *in,
                                                       DecodeParams opts,
                                                       const ROI &roi) {
  Context ctx = {};
  ctx.nvjpeg2k_decode_state = &nvjpeg2k_decode_states_[thread_idx];
  ctx.nvjpeg2k_stream = &nvjpeg2k_streams_[thread_idx];
  ctx.decode_event = &decode_events_[thread_idx];
  ctx.cuda_stream = &stream;
  DecodeResult result = {false, nullptr};

  if (!ParseJpeg2000Info(in, opts, &ctx)) 
    return result;

  CUDA_CALL(cudaEventSynchronize(*ctx.decode_event));

  if (!ctx.needs_processing) {
    result.success = DecodeJpeg2000(in, out.mutable_data<uint8_t>(), opts, &ctx);
  } else {
    auto &buffer = intermediate_buffers_[thread_idx];
    buffer.clear();
    buffer.resize(volume(ctx.shape) * ctx.pixel_size);
    try {
      result.success = 
        DecodeJpeg2000(in, buffer.data(), opts, &ctx) &&
        ConvertData(buffer.data(), out.mutable_data<uint8_t>(), opts, &ctx);
    } catch (...) {
      result.success = false;
      result.exception = std::current_exception();
    }
  }

  if (result.success)
    CUDA_CALL(cudaEventRecord(*ctx.decode_event, *ctx.cuda_stream));
  return result;
}

}  // namespace imgcodec
}  // namespace dali
