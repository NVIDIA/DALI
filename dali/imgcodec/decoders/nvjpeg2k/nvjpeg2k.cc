// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <string>
#include <vector>
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k.h"
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k_memory.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/imgcodec/decoders/nvjpeg/permute_layout.h"
#include "dali/imgcodec/util/convert_gpu.h"
#include "dali/imgcodec/util/convert_utils.h"
#include "dali/core/static_switch.h"
#include "dali/imgcodec/registry.h"
#include "dali/pipeline/util/for_each_thread.h"
#include "dali/kernels/dynamic_scratchpad.h"

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

NvJpeg2000DecoderInstance::NvJpeg2000DecoderInstance(
    int device_id, const std::map<std::string, std::any> &params)
: BatchParallelDecoderImpl(device_id, params)
, nvjpeg2k_dev_alloc_(nvjpeg_memory::GetDeviceAllocatorNvJpeg2k())
, nvjpeg2k_pin_alloc_(nvjpeg_memory::GetPinnedAllocatorNvJpeg2k()) {
  SetParams(params);
  std::any num_threads_any = GetParam("nvjpeg2k_num_threads");
  int num_threads = num_threads_any.has_value() ? std::any_cast<int>(num_threads_any) : 4;
  tp_ = std::make_unique<ThreadPool>(num_threads, device_id, true, "NvJpeg2000DecoderInstance");
  per_thread_resources_ = vector<PerThreadResources>(num_threads);
  size_t device_memory_padding = std::any_cast<size_t>(GetParam("nvjpeg2k_device_memory_padding"));
  size_t host_memory_padding = std::any_cast<size_t>(GetParam("nvjpeg2k_host_memory_padding"));

  nvjpeg2k_handle_ = NvJpeg2kHandle(&nvjpeg2k_dev_alloc_, &nvjpeg2k_pin_alloc_);
  DALI_ENFORCE(nvjpeg2k_handle_, "NvJpeg2kHandle initialization failed");

  ForEachThread(*tp_, [&](int tid) {
    CUDA_CALL(cudaSetDevice(device_id));
    per_thread_resources_[tid] = {nvjpeg2k_handle_, device_memory_padding, device_id_};
  });

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
  tp_->WaitForWork();
  for (const auto &res : per_thread_resources_)
    CUDA_CALL(cudaStreamSynchronize(res.cuda_stream));

  ForEachThread(*tp_, [&](int tid) {
      auto &res = per_thread_resources_[tid];
      res.nvjpeg2k_decode_state.reset();
    });

  for (auto thread_id : tp_->GetThreadIds())
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
  DALI_ENFORCE(ctx.bpp <= 16, make_string("Unsupported bits per pixel value: ", ctx.bpp));
  ctx.pixel_type = ctx.bpp <= 8 ? DALI_UINT8 : DALI_UINT16;

  for (uint32_t c = 1; c < ctx.image_info.num_components; c++) {
    CUDA_CALL(nvjpeg2kStreamGetImageComponentInfo(ctx.nvjpeg2k_stream, &comp, c));
    if (height != comp.component_height ||
        width != comp.component_width ||
        ctx.bpp != comp.precision)
      return false;
  }

  // nvJPEG2000 decodes into the planar layout
  ctx.orig_shape = {ctx.image_info.num_components, height, width};
  ctx.shape = ctx.orig_shape;
  if (ctx.roi) {
    auto roi_shape = ctx.roi.shape();
    ctx.shape = {ctx.image_info.num_components, roi_shape[0], roi_shape[1]};
  }
  return true;
}

bool NvJpeg2000DecoderInstance::DecodeJpeg2000(ImageSource *in, void *out,
                                               const TensorShape<> &shape, const Context &ctx,
                                               const ROI &roi) {
  // allocating buffers
  void *pixel_data[NVJPEG_MAX_COMPONENT] = {};
  size_t pitch_in_bytes[NVJPEG_MAX_COMPONENT] = {};
  uint8_t *out_as_bytes = static_cast<uint8_t*>(out);
  const int64_t channels = shape[0], height = shape[1], width = shape[2];
  const int64_t type_size = dali::TypeTable::GetTypeInfo(ctx.pixel_type).size();
  const int64_t component_byte_size = height * width * type_size;
  for (uint32_t c = 0; c < channels; c++) {
    pixel_data[c] = out_as_bytes + c * component_byte_size;
    pitch_in_bytes[c] = width * type_size;
  }
  nvjpeg2kImage_t image;
  image.pixel_data = pixel_data;
  image.pitch_in_bytes = pitch_in_bytes;
  image.num_components = channels;
  image.pixel_type = ctx.pixel_type == DALI_UINT8 ? NVJPEG2K_UINT8 : NVJPEG2K_UINT16;

  nvjpeg2kStatus_t ret;
  if (roi) {
    CUDA_CALL(nvjpeg2kDecodeParamsSetDecodeArea(
      ctx.params, roi.begin[1], roi.end[1], roi.begin[0], roi.end[0]));
    ret = nvjpeg2kDecodeImage(
      nvjpeg2k_handle_, ctx.nvjpeg2k_decode_state,
      ctx.nvjpeg2k_stream, ctx.params, &image, ctx.cuda_stream);
  } else {
    ret = nvjpeg2kDecode(nvjpeg2k_handle_, ctx.nvjpeg2k_decode_state,
        ctx.nvjpeg2k_stream, &image, ctx.cuda_stream);
  }
  return check_status(ret, in);
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

  CUDA_CALL(cudaEventSynchronize(ctx.decode_event));

  if (!ParseJpeg2000Info(in, ctx))
    return result;

  const int64_t channels = ctx.image_info.num_components;
  bool is_roi_oob = ctx.roi &&
    (ctx.roi.begin[0] < 0 || ctx.roi.end[0] > ctx.image_info.image_width ||
     ctx.roi.begin[1] < 0 || ctx.roi.end[1] > ctx.image_info.image_height);

  DALIImageType format = channels == 1 ? DALI_GRAY : DALI_RGB;
  bool is_processing_needed =
    channels > 1 ||  // nvJPEG2000 decodes into planar layout
    ctx.pixel_type != opts.dtype ||
    format != opts.format ||
    (ctx.bpp != 8 && ctx.bpp != 16) ||
    is_roi_oob;

  auto decode_out = out;
  TensorShape<> decode_sh;
  ROI roi_decode;  // ROI passed to decode API
  ROI roi_convert;  // ROI passed to convert API
  if (ctx.roi && !is_roi_oob) {
    // if ROI decoding and ROI within bounds
    // then use ROI decoding API directly
    decode_sh = ctx.shape;
    roi_decode = ctx.roi;
  } else {
    // Otherwise, we decode full image, then slice/pad via convert
    decode_sh = ctx.orig_shape;
    roi_convert =  ctx.roi;
  }

  kernels::DynamicScratchpad scratchpad({}, ctx.cuda_stream);
  if (is_processing_needed) {
    int64_t type_size = dali::TypeTable::GetTypeInfo(ctx.pixel_type).size();
    size_t nbytes = volume(decode_sh) * type_size;
    void *tmp_buff = scratchpad.AllocateGPU<uint8_t>(nbytes, type_size);
    decode_out = {tmp_buff, decode_sh, ctx.pixel_type};
  }
  result.success = DecodeJpeg2000(in, decode_out.raw_mutable_data(), decode_sh, ctx, roi_decode);

  if (is_processing_needed) {
    auto multiplier = DynamicRangeMultiplier(ctx.bpp, ctx.pixel_type);
    Convert(out, "HWC", opts.format, decode_out, "CHW", format, ctx.cuda_stream,
            roi_convert, {}, multiplier);
  }

  if (result.success) {
    CUDA_CALL(cudaEventRecord(ctx.decode_event, ctx.cuda_stream));
    CUDA_CALL(cudaStreamWaitEvent(stream, ctx.decode_event, 0));
  }

  return result;
}

REGISTER_DECODER("JPEG2000", NvJpeg2000DecoderFactory, CUDADecoderPriority);

}  // namespace imgcodec
}  // namespace dali
