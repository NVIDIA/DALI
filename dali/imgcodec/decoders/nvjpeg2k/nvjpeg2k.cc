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
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k.h"
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k_memory.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/imgcodec/decoders/nvjpeg/permute_layout.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace imgcodec {

namespace {
bool check_status(nvjpeg2kStatus_t status, ImageSource *in) {
  if (status == NVJPEG2K_STATUS_SUCCESS) {
    return true;
  } else if (status == NVJPEG2K_STATUS_BAD_JPEG || status == NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED) {
    return false;
  } else {
    CUDA_CALL_EX(status, in->SourceInfo());
    DALI_FAIL("Unreachable");  // silence a warning
  }
}
}  // namespace

NvJpeg2000DecoderInstance::NvJpeg2000DecoderInstance(int device_id)
: BatchParallelDecoderImpl(device_id)
, nvjpeg2k_dev_alloc_(nvjpeg_memory::GetDeviceAllocatorNvJpeg2k())
, nvjpeg2k_pin_alloc_(nvjpeg_memory::GetPinnedAllocatorNvJpeg2k()) {
  // TODO(staniewzki): pass params at construction
  any num_threads_any = GetParam("nvjpeg2k_num_threads");
  int num_threads = num_threads_any.has_value() ? any_cast<int>(num_threads_any) : 4;
  tp_ = std::make_unique<ThreadPool>(num_threads, device_id, true, "NvJpeg2000DecoderInstance");
  per_thread_resources_ = vector<PerThreadResources>(num_threads);
  size_t device_memory_padding = any_cast<size_t>(GetParam("nvjpeg2k_device_memory_padding"));
  size_t host_memory_padding = any_cast<size_t>(GetParam("nvjpeg2k_host_memory_padding"));

  nvjpeg2k_handle_ = NvJpeg2kHandle(&nvjpeg2k_dev_alloc_, &nvjpeg2k_pin_alloc_);
  DALI_ENFORCE(nvjpeg2k_handle_, "NvJpeg2kHandle initalization failed");

  for (auto &res : per_thread_resources_)
    res = {nvjpeg2k_handle_, device_memory_padding, device_id_};

  for (const auto &thread_id : tp_->GetThreadIds()) {
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
}

NvJpeg2000DecoderInstance::~NvJpeg2000DecoderInstance() {
  for (const auto &res : per_thread_resources_)
    CUDA_CALL(cudaStreamSynchronize(res.cuda_stream));
  for (const auto &thread_id : tp_->GetThreadIds())
    nvjpeg_memory::DeleteAllBuffers(thread_id);
}

bool NvJpeg2000DecoderInstance::ParseJpeg2000Info(ImageSource *in, Context &ctx) {
  CUDA_CALL(nvjpeg2kStreamParse(nvjpeg2k_handle_, in->RawData<uint8_t>(), in->Size(),
                                0, 0, ctx.nvjpeg2k_stream));

  CUDA_CALL(nvjpeg2kStreamGetImageInfo(ctx.nvjpeg2k_stream, &ctx.image_info));

  nvjpeg2kImageComponentInfo_t comp;
  CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(ctx.nvjpeg2k_stream, &comp, 0));
  auto height = comp.component_height;
  auto width = comp.component_width;
  ctx.bpp = comp.precision;
  DALI_ENFORCE(ctx.bpp == 8 || ctx.bpp == 16,
               make_string("Unsupported bits per pixel value: ", ctx.bpp));
  ctx.pixel_type = ctx.bpp == 8 ? DALI_UINT8 : DALI_UINT16;

  for (uint32_t c = 1; c < ctx.image_info.num_components; c++) {
    CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(ctx.nvjpeg2k_stream, &comp, c));
    if (height != comp.component_height ||
        width != comp.component_width ||
        ctx.bpp != comp.precision)
      return false;
  }

  if (ctx.roi) {
    auto shape = ctx.roi.shape();
    height = shape[0];
    width = shape[1];
  }
  ctx.shape = {height, width, ctx.image_info.num_components};

  return true;
}

nvjpeg2kImage_t NvJpeg2000DecoderInstance::PrepareOutputArea(uint8_t *out,
                                                             void **pixel_data,
                                                             size_t *pitch_in_bytes,
                                                             int64_t output_offset_x,
                                                             int64_t output_offset_y,
                                                             const Context &ctx) {
  const int64_t pixel_size = dali::TypeTable::GetTypeInfo(ctx.pixel_type).size();
  const int64_t component_byte_size = ctx.shape[0] * ctx.shape[1] * pixel_size;
  const int64_t offset_byte_size = (ctx.shape[1] * output_offset_y + output_offset_x) * pixel_size;
  for (uint32_t c = 0; c < ctx.shape[2]; c++) {
    pixel_data[c] = out + c * component_byte_size + offset_byte_size;
    pitch_in_bytes[c] = ctx.shape[1] * pixel_size;
  }

  nvjpeg2kImage_t image;
  image.pixel_data = pixel_data;
  image.pitch_in_bytes = pitch_in_bytes;
  image.num_components = ctx.shape[2];
  image.pixel_type = ctx.pixel_type == DALI_UINT8 ? NVJPEG2K_UINT8 : NVJPEG2K_UINT16;

  return image;
}

bool NvJpeg2000DecoderInstance::DecodeJpeg2000(ImageSource *in, uint8_t *out, const Context &ctx) {
  // allocating buffers
  void *pixel_data[NVJPEG_MAX_COMPONENT] = {};
  size_t pitch_in_bytes[NVJPEG_MAX_COMPONENT] = {};

  if (!ctx.roi) {
    auto output_image = PrepareOutputArea(out, pixel_data, pitch_in_bytes, 0, 0, ctx);
    auto ret = nvjpeg2kDecode(nvjpeg2k_handle_, ctx.nvjpeg2k_decode_state,
                              ctx.nvjpeg2k_stream, &output_image, ctx.cuda_stream);
    return check_status(ret, in);
  } else {
    // Decode tile by tile: nvjpeg2kDecodeImage seems to be bugged
    auto &image_info = ctx.image_info;
    auto &roi = ctx.roi;
    std::array tile_shape = {image_info.tile_height, image_info.tile_width};

    int state_idx = 0;
    for (uint32_t tile_y = 0; tile_y < image_info.num_tiles_y; tile_y++) {
     for (uint32_t tile_x = 0; tile_x < image_info.num_tiles_x; tile_x++) {
        auto calc_one_dimension = [&](int dim) {
          uint32_t tile_nr = (dim == 1 ? tile_x : tile_y);
          uint32_t tile_begin = tile_nr * tile_shape[dim];
          uint32_t tile_end = tile_begin + tile_shape[dim];
          uint32_t roi_begin = roi.begin[dim];
          uint32_t roi_end = roi.end[dim];

          // Intersection of roi and tile
          uint32_t decode_begin = std::max(roi_begin, tile_begin);
          uint32_t decode_end = std::min(roi_end, tile_end);
          uint32_t output_offset = tile_begin > roi_begin ? tile_begin - roi_begin : 0;

          return std::tuple{decode_begin, decode_end, output_offset};
        };

        auto [begin_x, end_x, output_offset_x] = calc_one_dimension(1);
        auto [begin_y, end_y, output_offset_y] = calc_one_dimension(0);

        if (begin_x < end_x && begin_y < end_y) {
          const TileDecodingResources &per_tile_ctx = ctx.tile_dec_res[state_idx];
          state_idx = (state_idx + 1) % ctx.tile_dec_res.size();

          CUDA_CALL(cudaEventSynchronize(per_tile_ctx.decode_event));

          NvJpeg2kDecodeParams params;
          CUDA_CALL(nvjpeg2kDecodeParamsSetDecodeArea(params, begin_x, end_x, begin_y, end_y));

          auto output_image = PrepareOutputArea(out, pixel_data, pitch_in_bytes, output_offset_x,
                                                output_offset_y, ctx);

          auto ret = nvjpeg2kDecodeTile(nvjpeg2k_handle_,
                                        per_tile_ctx.state,
                                        ctx.nvjpeg2k_stream,
                                        params,
                                        tile_x + tile_y * image_info.num_tiles_x,
                                        0,
                                        &output_image,
                                        ctx.cuda_stream);

          if (ret != NVJPEG2K_STATUS_SUCCESS)
            return check_status(ret, in);

          CUDA_CALL(cudaEventRecord(per_tile_ctx.decode_event, ctx.cuda_stream));
        }
      }
    }
    return true;
  }
}

bool NvJpeg2000DecoderInstance::ConvertData(void *in, uint8_t *out, const Context &ctx) {
  int64_t pixels_count = ctx.shape[0] * ctx.shape[1];
  if (ctx.opts.format == DALI_GRAY) {
    // Converting to Gray, dropping alpha channels if needed
    assert(ctx.shape[2] >= 3);
    TYPE_SWITCH(ctx.pixel_type, type2id, Input, (uint8_t, uint16_t), (
      PlanarRGBToGray<uint8_t, Input>(
        out, reinterpret_cast<Input*>(in), pixels_count,
        ctx.pixel_type, ctx.cuda_stream);
    ), DALI_FAIL(make_string("Unsupported input type: ", ctx.pixel_type)));  // NOLINT
  } else {
    // Converting to interleaved, dropping alpha channels if needed
    assert(ctx.shape[2] >= 3);
    int req_nchannels = NumberOfChannels(ctx.opts.format, ctx.shape[2]);
    TYPE_SWITCH(ctx.pixel_type, type2id, Input, (uint8_t, uint16_t), (
      PlanarToInterleaved<uint8_t, Input>(
        out, reinterpret_cast<Input*>(in), pixels_count,
        req_nchannels, ctx.opts.format, ctx.pixel_type, ctx.cuda_stream);
    ), DALI_FAIL(make_string("Unsupported input type: ", ctx.pixel_type)));  // NOLINT
  }
  return true;
}

DecodeResult NvJpeg2000DecoderInstance::DecodeImplTask(int thread_idx,
                                                       cudaStream_t stream,
                                                       SampleView<GPUBackend> out,
                                                       ImageSource *in,
                                                       DecodeParams opts,
                                                       const ROI &roi) {
  auto &res = per_thread_resources_[thread_idx];
  Context ctx(opts, roi, res);
  DecodeResult result = {false, nullptr};

  if (!ParseJpeg2000Info(in, ctx))
    return result;

  CUDA_CALL(cudaEventSynchronize(ctx.decode_event));

  // nvJPEG2000 decodes into planar layout
  bool is_conversion_needed = ctx.shape[2] > 1 || ctx.bpp == 16;
  try {
    if (!is_conversion_needed) {
      result.success = DecodeJpeg2000(in, out.mutable_data<uint8_t>(), ctx);
    } else {
      auto &buffer = res.intermediate_buffer;
      buffer.clear();
      int64_t pixel_size = dali::TypeTable::GetTypeInfo(ctx.pixel_type).size();
      buffer.resize(volume(ctx.shape) * pixel_size);
      result.success = DecodeJpeg2000(in, buffer.data(), ctx) &&
                       ConvertData(buffer.data(), out.mutable_data<uint8_t>(), ctx);
    }
  } catch (...) {
    result.success = false;
    result.exception = std::current_exception();
  }

  if (result.success) {
    CUDA_CALL(cudaEventRecord(ctx.decode_event, ctx.cuda_stream));
    CUDA_CALL(cudaStreamWaitEvent(stream, ctx.decode_event, 0));
  }

  return result;
}

}  // namespace imgcodec
}  // namespace dali
