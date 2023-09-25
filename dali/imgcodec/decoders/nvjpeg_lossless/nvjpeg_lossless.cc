// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/decoders/nvjpeg_lossless/nvjpeg_lossless.h"
#include <map>
#include <string>
#include <utility>
#include "dali/core/device_guard.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_helper.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/registry.h"
#include "dali/imgcodec/util/convert_gpu.h"
#include "dali/imgcodec/util/convert_utils.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace imgcodec {

NvJpegLosslessDecoderInstance::NvJpegLosslessDecoderInstance(
    int device_id, const std::map<std::string, std::any> &params)
    : BatchedApiDecoderImpl(device_id, params),
      event_(CUDAEvent::Create(device_id)) {
  DeviceGuard dg(device_id_);
  // TODO(janton): use custom allocators (?)
  auto ret = nvjpegCreateEx(NVJPEG_BACKEND_LOSSLESS_JPEG, NULL, NULL, 0, &nvjpeg_handle_);
  if (ret != NVJPEG_STATUS_SUCCESS) {
    // some nvJPEG version doesn't support NVJPEG_BACKEND_LOSSLESS_JPEG so disable it if
    // it failed to initialize
    DALI_WARN_ONCE("The available nvJPEG library version doesn't support Lossless format, please "
                   "update to the latest one.");
    is_initialized_ = false;
    return;
  }
  is_initialized_ = true;
  per_thread_resources_.push_back(PerThreadResources{nvjpeg_handle_});
  CUDA_CALL(nvjpegJpegStateCreate(nvjpeg_handle_, &state_));
}

NvJpegLosslessDecoderInstance::~NvJpegLosslessDecoderInstance() {
  if (!is_initialized_) {
    return;
  }
  DeviceGuard dg(device_id_);
  CUDA_DTOR_CALL(cudaEventSynchronize(event_));
  per_thread_resources_.clear();
  CUDA_DTOR_CALL(nvjpegJpegStateDestroy(state_));
  CUDA_DTOR_CALL(nvjpegDestroy(nvjpeg_handle_));
}

NvJpegLosslessDecoderInstance::PerThreadResources::PerThreadResources(nvjpegHandle_t handle) {
  CUDA_CALL(nvjpegJpegStreamCreate(handle, &jpeg_stream));
}

NvJpegLosslessDecoderInstance::PerThreadResources::PerThreadResources(PerThreadResources&& other)
  : jpeg_stream(other.jpeg_stream) {
  other.jpeg_stream = nullptr;
}

NvJpegLosslessDecoderInstance::PerThreadResources::~PerThreadResources() {
  if (jpeg_stream) {
    CUDA_DTOR_CALL(nvjpegJpegStreamDestroy(jpeg_stream));
  }
}

bool NvJpegLosslessDecoderInstance::CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts,
                                              const ROI &roi) {
  if (!is_initialized_) {
    return false;
  }
  if (opts.format != DALI_ANY_DATA && opts.format != DALI_GRAY) {
    return false;
  }

  JpegParser jpeg_parser{};
  if (!jpeg_parser.CanParse(in))
    return false;

  // This decoder only supports SOF-3 (JPEG lossless) samples
  auto ext_info = jpeg_parser.GetExtendedInfo(in);
  std::array<uint8_t, 2> sof3_marker = {0xff, 0xc3};
  bool is_lossless_jpeg = ext_info.sof_marker == sof3_marker;
  return is_lossless_jpeg;
}

void NvJpegLosslessDecoderInstance::Parse(DecodeResultsPromise &promise,
                                          DecodeContext ctx,
                                          cspan<ImageSource *> in,
                                          DecodeParams opts,
                                          cspan<ROI> rois) {
  int nsamples = in.size();
  assert(rois.empty() || rois.size() == nsamples);
  assert(ctx.tp != nullptr);
  int nthreads = ctx.tp->NumThreads();
  if (nthreads > static_cast<int>(per_thread_resources_.size())) {
    per_thread_resources_.reserve(nthreads);
    for (int i = per_thread_resources_.size(); i < nthreads; i++)
      per_thread_resources_.emplace_back(nvjpeg_handle_);
  }
  sample_meta_.clear();
  sample_meta_.resize(nsamples);

  // temporary solution: Just to check when the parsing has finished
  DecodeResultsPromise parse_promise(nsamples);
  for (int i = 0; i < nsamples; i++) {
    int tid = 0;
    ctx.tp->AddWork(
        [&, i](int tid) {
          auto &jpeg_stream = per_thread_resources_[tid].jpeg_stream;
          auto *sample = in[i];
          auto &meta = sample_meta_[i];
          try {
            CUDA_CALL(nvjpegJpegStreamParseHeader(nvjpeg_handle_, sample->RawData<unsigned char>(),
                                                  sample->Size(), jpeg_stream));
            int is_supported = 0;  // 0 means yes
            CUDA_CALL(nvjpegDecodeBatchedSupported(nvjpeg_handle_, jpeg_stream, &is_supported));
            meta.can_decode = (is_supported == 0);
            if (!meta.can_decode) {
              promise.set(i, {false, nullptr});
              parse_promise.set(i, {false, nullptr});
              return;
            }
            meta.needs_processing = opts.dtype != DALI_UINT16;
            meta.needs_processing |= !rois.empty() && rois[i].use_roi();
            if (opts.use_orientation) {
              auto &ori = meta.orientation = JpegParser().GetInfo(sample).orientation;
              meta.needs_processing |= (ori.rotate || ori.flip_x || ori.flip_y);
            }

            unsigned int precision;
            CUDA_CALL(nvjpegJpegStreamGetSamplePrecision(jpeg_stream, &precision));
            meta.dyn_range_multiplier = 1.0f;
            if (NeedDynamicRangeScaling(precision, DALI_UINT16)) {
              meta.dyn_range_multiplier = DynamicRangeMultiplier(precision, DALI_UINT16);
              meta.needs_processing = true;
            }
            parse_promise.set(i, {true, nullptr});
          } catch (...) {
            meta.can_decode = false;
            promise.set(i, {false, std::current_exception()});
            parse_promise.set(i, {false, nullptr});
          }
        }, in[i]->Size());
  }
  ctx.tp->RunAll(false);
  parse_promise.get_future().wait_all();

  batch_sz_ = 0;
  for (auto &meta : sample_meta_) {
    if (meta.can_decode)
      meta.idx_in_batch = batch_sz_++;
  }
}

void NvJpegLosslessDecoderInstance::RunDecode(kernels::DynamicScratchpad& s,
                                              DecodeContext ctx,
                                              span<SampleView<GPUBackend>> out,
                                              cspan<ImageSource *> in,
                                              DecodeParams opts,
                                              cspan<ROI> rois) {
  if (batch_sz_ <= 0)
    return;
  int nsamples = in.size();
  encoded_.clear();
  encoded_.resize(batch_sz_);
  encoded_len_.clear();
  encoded_len_.resize(batch_sz_);
  decoded_.clear();
  decoded_.resize(batch_sz_);
  for (int i = 0; i < nsamples; i++) {
    auto &meta = sample_meta_[i];
    if (!meta.can_decode)
      continue;
    auto *sample = in[i];
    auto &out_sample = out[i];
    auto roi = rois.empty() ? ROI{} : rois[i];
    assert(sample->Kind() == InputKind::HostMemory);
    encoded_[meta.idx_in_batch] = sample->RawData<unsigned char>();
    encoded_len_[meta.idx_in_batch] = sample->Size();
    auto &o = decoded_[meta.idx_in_batch];
    auto sh = out_sample.shape();
    o.pitch[0] = sh[1] * sh[2] * sizeof(uint16_t);
    if (meta.needs_processing) {
      int64_t nbytes = volume(sh) * sizeof(uint16_t);
      o.channel[0] = s.Allocate<mm::memory_kind::device, uint8_t>(nbytes);
    } else {
      o.channel[0] = static_cast<uint8_t *>(out_sample.raw_mutable_data());
    }
  }

  CUDA_CALL(nvjpegDecodeBatchedInitialize(nvjpeg_handle_, state_, batch_sz_, 1,
                                          NVJPEG_OUTPUT_UNCHANGEDI_U16));
  CUDA_CALL(nvjpegDecodeBatched(nvjpeg_handle_, state_, encoded_.data(), encoded_len_.data(),
                                decoded_.data(), ctx.stream));
}


FutureDecodeResults NvJpegLosslessDecoderInstance::ScheduleDecode(DecodeContext ctx,
                                                                  span<SampleView<GPUBackend>> out,
                                                                  cspan<ImageSource *> in,
                                                                  DecodeParams opts,
                                                                  cspan<ROI> rois) {
  int nsamples = in.size();
  assert(out.size() == nsamples);

  DecodeResultsPromise promise(nsamples);
  auto set_promise = [&](DecodeResult result) {
    for (int i = 0; i < nsamples; i++)
      promise.set(i, result);
  };

  if (opts.format != DALI_ANY_DATA && opts.format != DALI_GRAY)
    set_promise({false, std::make_exception_ptr(
                            std::invalid_argument("Only ANY_DATA and GRAY are supported."))});

  // scratchpad should not go out of scope until we launch the postprocessing
  kernels::DynamicScratchpad s({}, ctx.stream);
  Parse(promise, ctx, in, opts, rois);
  try {
    RunDecode(s, ctx, out, in, opts, rois);
  } catch(...) {
    set_promise({false, std::current_exception()});
  }
  Postprocess(promise, ctx, out, opts, rois);
  CUDA_CALL(cudaEventRecord(event_, ctx.stream));
  return promise.get_future();
}

void NvJpegLosslessDecoderInstance::Postprocess(DecodeResultsPromise &promise, DecodeContext ctx,
                                                span<SampleView<GPUBackend>> out, DecodeParams opts,
                                                cspan<ROI> rois) {
  if (batch_sz_ <= 0)
    return;
  int nsamples = out.size();
  for (int i = 0, j = 0; i < nsamples; i++) {
    const auto &meta = sample_meta_[i];
    if (!meta.can_decode)
      continue;  // we didn't try to decode this sample

    const auto &decoded = decoded_[j++];  // decoded only has samples where can_decode == true
    if (!meta.needs_processing) {
      promise.set(i, {true, nullptr});
      continue;
    }
    auto sh = out[i].shape();
    auto roi = rois.empty() ? ROI{} : rois[i];
    SampleView<GPUBackend> decoded_view(decoded.channel[0], sh, DALI_UINT16);
    DALIImageType decoded_format = sh[2] == 1 ? DALI_GRAY : DALI_ANY_DATA;
    try {
      Convert(out[i], "HWC", opts.format, decoded_view, "HWC", decoded_format,
              ctx.stream, roi, meta.orientation, meta.dyn_range_multiplier);
      promise.set(i, {true, nullptr});
    } catch (...) {
      promise.set(i, {false, std::current_exception()});
    }
  }
}

REGISTER_DECODER("JPEG", NvJpegLosslessDecoderFactory, CUDADecoderPriority + 1);

}  // namespace imgcodec
}  // namespace dali
