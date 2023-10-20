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
#include <utility>
#include "dali/core/device_guard.h"
#include "dali/core/version_util.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_helper.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/imgcodec/decoders/nvjpeg/permute_layout.h"
#include "dali/imgcodec/registry.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/util/convert_gpu.h"
#include "dali/core/mm/memory.h"

namespace dali {
namespace imgcodec {

namespace {

int nvjpegGetVersion() {
  int major = -1;
  int minor = -1;
  int patch = -1;
  GetVersionProperty(nvjpegGetProperty, &major, MAJOR_VERSION, NVJPEG_STATUS_SUCCESS);
  GetVersionProperty(nvjpegGetProperty, &minor, MINOR_VERSION, NVJPEG_STATUS_SUCCESS);
  GetVersionProperty(nvjpegGetProperty, &patch, PATCH_LEVEL, NVJPEG_STATUS_SUCCESS);
  return MakeVersionNumber(major, minor, patch);
}

}  // namespace

NvJpegDecoderInstance::
NvJpegDecoderInstance(int device_id, const std::map<std::string, std::any> &params)
: BatchParallelDecoderImpl(device_id, params)
, device_allocator_(nvjpeg_memory::GetDeviceAllocator())
, pinned_allocator_(nvjpeg_memory::GetPinnedAllocator()) {
  SetParams(params);

  unsigned int nvjpeg_flags = 0;
#ifdef NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION
  if (use_jpeg_fancy_upsampling_ && nvjpegGetVersion() >= 12001) {
    nvjpeg_flags |= NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION;
  }
#endif

  DeviceGuard dg(device_id_);

  CUDA_CALL(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, nvjpeg_flags, &nvjpeg_handle_));

  tp_ = std::make_unique<ThreadPool>(num_threads_, device_id, true, "NvJpegDecoderInstance");
  resources_.reserve(tp_->NumThreads());

  if (host_memory_padding_ > 0) {
    CUDA_CALL(nvjpegSetPinnedMemoryPadding(host_memory_padding_, nvjpeg_handle_));
    for (auto thread_id : tp_->GetThreadIds()) {
      nvjpeg_memory::AddHostBuffer(thread_id, host_memory_padding_);
      nvjpeg_memory::AddHostBuffer(thread_id, host_memory_padding_);
    }
  }

  if (device_memory_padding_ > 0) {
    CUDA_CALL(nvjpegSetDeviceMemoryPadding(device_memory_padding_, nvjpeg_handle_));
    for (auto thread_id : tp_->GetThreadIds()) {
      nvjpeg_memory::AddBuffer<mm::memory_kind::device>(thread_id, device_memory_padding_);
    }
  }

  for (int i = 0; i < tp_->NumThreads(); i++) {
    resources_.emplace_back(nvjpeg_handle_, &device_allocator_, &pinned_allocator_, device_id_);
  }
}

NvJpegDecoderInstance::
PerThreadResources::PerThreadResources(nvjpegHandle_t nvjpeg_handle,
                                       nvjpegDevAllocator_t *device_allocator,
                                       nvjpegPinnedAllocator_t *pinned_allocator,
                                       int device_id)
  : stream(CUDAStreamPool::instance().Get(device_id)) {
  CUDA_CALL(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_stream));
  for (auto &buffer : pinned_buffers) {
    CUDA_CALL(nvjpegBufferPinnedCreate(nvjpeg_handle, pinned_allocator,
                                      &buffer));
  }
  CUDA_CALL(nvjpegBufferDeviceCreate(nvjpeg_handle, device_allocator,
                                     &device_buffer));

  for (auto &event : decode_events)
    event = CUDAEvent::Create(device_id);

  auto backend = NVJPEG_BACKEND_HYBRID;  // TODO(msala) allow other backens
  CUDA_CALL(nvjpegDecoderCreate(nvjpeg_handle, backend, &decoder_data.decoder));
  CUDA_CALL(nvjpegDecoderStateCreate(nvjpeg_handle, decoder_data.decoder, &decoder_data.state));

  CUDA_CALL(nvjpegDecodeParamsCreate(nvjpeg_handle, &params));
}

NvJpegDecoderInstance::PerThreadResources::PerThreadResources(PerThreadResources&& other)
: decoder_data(other.decoder_data)
, device_buffer(other.device_buffer)
, jpeg_stream(other.jpeg_stream)
, stream(std::move(other.stream))
, decode_events(std::move(other.decode_events))
, params(std::move(other.params)) {
  other.decoder_data = {};
  other.device_buffer = nullptr;
  for (int i = 0; i < 2; i++) {
    pinned_buffers[i] = other.pinned_buffers[i];
    other.pinned_buffers[i] = nullptr;
  }
  other.jpeg_stream = nullptr;
  other.params = nullptr;
}

NvJpegDecoderInstance::~NvJpegDecoderInstance() {
  DeviceGuard dg(device_id_);

  for (auto &thread_id : tp_->GetThreadIds()) {
    nvjpeg_memory::DeleteAllBuffers(thread_id);
  }

  // Call destructors of all thread resources.
  resources_.clear();

  CUDA_CALL(nvjpegDestroy(nvjpeg_handle_));
}

NvJpegDecoderInstance::PerThreadResources::~PerThreadResources() {
  // This check should probably be enough
  if (stream) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }

  if (params) {
    CUDA_CALL(nvjpegDecodeParamsDestroy(params));
  }
  if (jpeg_stream) {
    CUDA_CALL(nvjpegJpegStreamDestroy(jpeg_stream));
  }

  for (auto &buffer : pinned_buffers) {
    if (buffer) {
      CUDA_CALL(nvjpegBufferPinnedDestroy(buffer));
    }
  }

  if (device_buffer) {
    CUDA_CALL(nvjpegBufferDeviceDestroy(device_buffer));
  }

  if (decoder_data.decoder && decoder_data.state) {
    CUDA_CALL(nvjpegDecoderDestroy(decoder_data.decoder));
    CUDA_CALL(nvjpegJpegStateDestroy(decoder_data.state));
  }
}

bool NvJpegDecoderInstance::CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts,
                                      const ROI &roi) {
  JpegParser jpeg_parser{};
  if (!jpeg_parser.CanParse(in))
    return false;

  // This decoder does not support SOF-3 (JPEG lossless) samples
  auto ext_info = jpeg_parser.GetExtendedInfo(in);
  std::array<uint8_t, 2> sof3_marker = {0xff, 0xc3};
  bool is_lossless_jpeg = ext_info.sof_marker == sof3_marker;
  return !is_lossless_jpeg;
}

bool NvJpegDecoderInstance::SetParam(const char *name, const std::any &value) {
  if (strcmp(name, "device_memory_padding") == 0) {
    device_memory_padding_ = std::any_cast<size_t>(value);
    return true;
  } else if (strcmp(name, "host_memory_padding") == 0) {
    host_memory_padding_ = std::any_cast<size_t>(value);
    return true;
  } else if (strcmp(name, "nvjpeg_num_threads") == 0) {
    num_threads_ = std::any_cast<int>(value);
    return true;
  } else if (strcmp(name, "jpeg_fancy_upsampling") == 0) {
    use_jpeg_fancy_upsampling_ = std::any_cast<bool>(value);
    return true;
  }

  return false;
}

std::any NvJpegDecoderInstance::GetParam(const char *name) const {
  if (strcmp(name, "device_memory_padding") == 0) {
    return device_memory_padding_;
  } else if (strcmp(name, "host_memory_padding") == 0) {
    return host_memory_padding_;
  } else if (strcmp(name, "nvjpeg_num_threads") == 0) {
    return num_threads_;
  } else if (strcmp(name, "jpeg_fancy_upsampling") == 0) {
    return use_jpeg_fancy_upsampling_;
  } else {
    return {};
  }
}

DecodeResult NvJpegDecoderInstance::DecodeImplTask(int thread_idx,
                                                   cudaStream_t stream,
                                                   SampleView<GPUBackend> out,
                                                   ImageSource *in,
                                                   DecodeParams opts,
                                                   const ROI &roi) {
  DecodingContext ctx = DecodingContext{ resources_[thread_idx] };
  CUDA_CALL(nvjpegDecodeParamsSetOutputFormat(ctx.resources.params, GetFormat(opts.format)));
  CUDA_CALL(nvjpegDecodeParamsSetAllowCMYK(ctx.resources.params, true));

  Orientation orientation = {};
  auto adjusted_roi = roi;
  if (opts.use_orientation) {
    auto info = JpegParser().GetInfo(in);
    adjusted_roi = PreOrientationRoi(info, roi);
    orientation = info.orientation;
  }
  bool is_orientation_adjusted = orientation.rotate || orientation.flip_x || orientation.flip_y;

  auto roi_shape = adjusted_roi.shape();
  if (roi.use_roi()) {
    CUDA_CALL(nvjpegDecodeParamsSetROI(ctx.resources.params,
                                       adjusted_roi.begin[1], adjusted_roi.begin[0],
                                       roi_shape[1], roi_shape[0]));
  } else {
    CUDA_CALL(nvjpegDecodeParamsSetROI(ctx.resources.params, 0, 0, -1, -1));
  }

  // We don't decode directly to YCbCr, since we want to control the YCbCr definition,
  // which is different between general color conversion libraries (OpenCV) and
  // what JPEG uses.
  // JPEG files are always using bitdepth 8.
  bool needs_processing = opts.format == DALI_YCbCr ||
                          opts.dtype != DALI_UINT8 ||
                          is_orientation_adjusted;

  ctx.resources.swap_buffers();
  auto &decode_event = ctx.resources.decode_event();
  try {
    ParseJpegSample(*in, opts, ctx);

    if (roi.use_roi()) {
      ctx.shape[0] = roi_shape[0];
      ctx.shape[1] = roi_shape[1];
    }

    auto& intermediate_buffer = ctx.resources.intermediate_buffer;

    uint8_t* decode_out;
    if (needs_processing) {
      intermediate_buffer.set_order(cudaStream_t(ctx.resources.stream));
      intermediate_buffer.resize(volume(ctx.shape), DALI_UINT8);
      decode_out = intermediate_buffer.mutable_data<uint8_t>();
    } else {
      decode_out = out.mutable_data<uint8_t>();
    }

    DecodeJpegSample(*in, decode_out, opts, ctx);

    if (needs_processing) {
      SampleView<GPUBackend> decoded_view(decode_out, ctx.shape, DALI_UINT8);
      DALIImageType decoded_format = ctx.shape[2] == 1 ? DALI_GRAY : DALI_RGB;
      Convert(out, "HWC", opts.format, decoded_view, "HWC", decoded_format,
              ctx.resources.stream, {}, orientation);
    }
  } catch (...) {
    return {false, std::current_exception()};
  }

  CUDA_CALL(cudaEventRecord(decode_event, ctx.resources.stream));
  CUDA_CALL(cudaStreamWaitEvent(stream, decode_event, 0));
  return {true, nullptr};
}

void NvJpegDecoderInstance::ParseJpegSample(ImageSource& in, DecodeParams opts,
                                            DecodingContext& ctx) {
  int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT], c;
  nvjpegChromaSubsampling_t subsampling;
  CUDA_CALL(nvjpegGetImageInfo(nvjpeg_handle_, in.RawData<unsigned char>(), in.Size(), &c,
                               &subsampling, widths, heights));
  ctx.shape = {heights[0], widths[0], NumberOfChannels(opts.format, c)};
}

void NvJpegDecoderInstance::DecodeJpegSample(ImageSource& in, uint8_t *out, DecodeParams opts,
                                             DecodingContext &ctx) {
  auto& decoder = ctx.resources.decoder_data.decoder;
  auto& state = ctx.resources.decoder_data.state;
  auto& jpeg_stream = ctx.resources.jpeg_stream;
  auto& stream = ctx.resources.stream;
  auto& device_buffer = ctx.resources.device_buffer;

  CUDA_CALL(cudaEventSynchronize(ctx.resources.decode_event()));
  CUDA_CALL(nvjpegStateAttachPinnedBuffer(state, ctx.resources.pinned_buffer()));
  CUDA_CALL(nvjpegJpegStreamParse(nvjpeg_handle_, in.RawData<unsigned char>(), in.Size(),
                                  false, false, ctx.resources.jpeg_stream));
  CUDA_CALL(nvjpegDecodeJpegHost(nvjpeg_handle_, decoder, state, ctx.resources.params,
                                 jpeg_stream));

  nvjpegImage_t nvjpeg_image;
  // For interleaved, nvjpeg expects a single channel but 3x bigger
  nvjpeg_image.channel[0] = out;
  nvjpeg_image.pitch[0] = ctx.shape[1] * ctx.shape[2];

  CUDA_CALL(nvjpegStateAttachDeviceBuffer(state, device_buffer));
  CUDA_CALL(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, decoder, state, jpeg_stream,
                                             stream));
  CUDA_CALL(nvjpegDecodeJpegDevice(nvjpeg_handle_, decoder, state, &nvjpeg_image, stream));
}

REGISTER_DECODER("JPEG", NvJpegDecoderFactory, CUDADecoderPriority);

}  // namespace imgcodec
}  // namespace dali
