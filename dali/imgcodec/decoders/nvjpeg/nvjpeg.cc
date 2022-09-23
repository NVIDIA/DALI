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

#include <map>
#include <string>
#include <utility>
#include "dali/core/device_guard.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_helper.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/imgcodec/decoders/nvjpeg/permute_layout.h"

namespace dali {
namespace imgcodec {

NvJpegDecoderInstance::
NvJpegDecoderInstance(int device_id, const std::map<std::string, any> &params)
: BatchParallelDecoderImpl(device_id, params)
, device_allocator_(nvjpeg_memory::GetDeviceAllocator())
, pinned_allocator_(nvjpeg_memory::GetPinnedAllocator()) {
  SetParams(params);

  DeviceGuard dg(device_id_);
  CUDA_CALL(nvjpegCreateSimple(&nvjpeg_handle_));

  tp_ = std::make_unique<ThreadPool>(num_threads_, device_id, true, "NvJpegDecoderInstance");
  resources_.reserve(tp_->NumThreads());

  if (host_memory_padding_ > 0) {
    for (auto thread_id : tp_->GetThreadIds()) {
      nvjpeg_memory::AddHostBuffer(thread_id, host_memory_padding_);
      nvjpeg_memory::AddHostBuffer(thread_id, host_memory_padding_);
    }
  }

  if (device_memory_padding_ > 0) {
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
  CUDA_CALL(nvjpegBufferPinnedCreate(nvjpeg_handle, pinned_allocator,
                                     &pinned_buffer));
  CUDA_CALL(nvjpegBufferDeviceCreate(nvjpeg_handle, device_allocator,
                                     &device_buffer));

  decode_event = CUDAEvent::Create(device_id);

  auto backend = NVJPEG_BACKEND_HYBRID;  // TODO(msala) allow other backens
  CUDA_CALL(nvjpegDecoderCreate(nvjpeg_handle, backend, &decoder_data.decoder));
  CUDA_CALL(nvjpegDecoderStateCreate(nvjpeg_handle, decoder_data.decoder, &decoder_data.state));

  CUDA_CALL(nvjpegDecodeParamsCreate(nvjpeg_handle, &params));
}

NvJpegDecoderInstance::PerThreadResources::PerThreadResources(PerThreadResources&& other)
: decoder_data(other.decoder_data)
, device_buffer(other.device_buffer)
, pinned_buffer(other.pinned_buffer)
, jpeg_stream(other.jpeg_stream)
, stream(std::move(other.stream))
, decode_event(std::move(other.decode_event))
, params(std::move(other.params)) {
  other.decoder_data = {};
  other.device_buffer = nullptr;
  other.pinned_buffer = nullptr;
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
  if (pinned_buffer) {
    CUDA_CALL(nvjpegBufferPinnedDestroy(pinned_buffer));
  }
  if (device_buffer) {
    CUDA_CALL(nvjpegBufferDeviceDestroy(device_buffer));
  }

  if (decoder_data.decoder && decoder_data.state) {
    CUDA_CALL(nvjpegDecoderDestroy(decoder_data.decoder));
    CUDA_CALL(nvjpegJpegStateDestroy(decoder_data.state));
  }
}

bool NvJpegDecoderInstance::SetParam(const char *name, const any &value) {
  if (strcmp(name, "device_memory_padding") == 0) {
    device_memory_padding_ = any_cast<size_t>(value);
    return true;
  } else if (strcmp(name, "host_memory_padding") == 0) {
    host_memory_padding_ = any_cast<size_t>(value);
    return true;
  } else if (strcmp(name, "nvjpeg_num_threads") == 0) {
    num_threads_ = any_cast<size_t>(value);
    return true;
  }

  return false;
}

any NvJpegDecoderInstance::GetParam(const char *name) const {
  if (strcmp(name, "device_memory_padding") == 0) {
    return device_memory_padding_;
  } else if (strcmp(name, "host_memory_padding") == 0) {
    return host_memory_padding_;
  } else if (strcmp(name, "nvjpeg_num_threads") == 0) {
    return num_threads_;
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
  if (roi.use_roi()) {
    CUDA_CALL(nvjpegDecodeParamsSetROI(ctx.resources.params, roi.begin[1], roi.begin[0],
                                       roi.shape()[1], roi.shape()[0]));
  }
  try {
    ParseJpegSample(*in, opts, ctx);
    if (roi.use_roi()) {
      ctx.shape = roi.shape();
    }
    DecodeJpegSample(*in, out.mutable_data<uint8_t>(), opts, ctx);
  } catch (...) {
    return {false, std::current_exception()};
  }

  CUDA_CALL(cudaStreamWaitEvent(stream, ctx.resources.decode_event, 0));
  return {true, nullptr};
}

void NvJpegDecoderInstance::ParseJpegSample(ImageSource& in, DecodeParams opts,
                                            DecodingContext& ctx) {
  int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT], c;
  nvjpegChromaSubsampling_t subsampling;
  CUDA_CALL(nvjpegGetImageInfo(nvjpeg_handle_, in.RawData<unsigned char>(), in.Size(), &c,
                               &subsampling, widths, heights));

  ctx.shape = {heights[0], widths[0], c};
}

void NvJpegDecoderInstance::DecodeJpegSample(ImageSource& in, uint8_t *out, DecodeParams opts,
                                             DecodingContext &ctx) {
  auto& decoder = ctx.resources.decoder_data.decoder;
  auto& state = ctx.resources.decoder_data.state;
  auto& jpeg_stream = ctx.resources.jpeg_stream;
  auto& stream = ctx.resources.stream;
  auto& decode_event = ctx.resources.decode_event;
  auto& device_buffer = ctx.resources.device_buffer;

  CUDA_CALL(nvjpegStateAttachPinnedBuffer(state, ctx.resources.pinned_buffer));
  CUDA_CALL(nvjpegJpegStreamParse(nvjpeg_handle_, in.RawData<unsigned char>(), in.Size(),
                                  false, false, ctx.resources.jpeg_stream));
  CUDA_CALL(nvjpegDecodeJpegHost(nvjpeg_handle_, decoder, state, ctx.resources.params,
                                 jpeg_stream));

  nvjpegImage_t nvjpeg_image;
  // For interleaved, nvjpeg expects a single channel but 3x bigger
  nvjpeg_image.channel[0] = out;
  nvjpeg_image.pitch[0] = ctx.shape[1] * ctx.shape[2];

  CUDA_CALL(cudaEventSynchronize(decode_event));
  CUDA_CALL(nvjpegStateAttachDeviceBuffer(state, device_buffer));
  CUDA_CALL(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, decoder, state, jpeg_stream,
                                             stream));
  CUDA_CALL(nvjpegDecodeJpegDevice(nvjpeg_handle_, decoder, state, &nvjpeg_image, stream));

  if (opts.format == DALI_YCbCr) {
    // We don't decode directly to YCbCr, since we want to control the YCbCr definition,
    // which is different between general color conversion libraries (OpenCV) and
    // what JPEG uses.
    int64_t npixels = ctx.shape[0] * ctx.shape[1];
    Convert_RGB_to_YCbCr(out, out, npixels, stream);
  }

  CUDA_CALL(cudaEventRecord(decode_event, stream));
}

}  // namespace imgcodec
}  // namespace dali
