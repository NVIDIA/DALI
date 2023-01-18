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
    int device_id, const std::map<std::string, any> &params)
    : BatchedApiDecoderImpl(device_id, params),
      event_(CUDAEvent::Create(device_id)) {
  DeviceGuard dg(device_id_);
  CUDA_CALL(nvjpegCreateEx(NVJPEG_BACKEND_LOSSLESS_JPEG, NULL, NULL, 0, &nvjpeg_handle_));
  CUDA_CALL(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_stream_));
  CUDA_CALL(nvjpegJpegStateCreate(nvjpeg_handle_, &state_));
}

NvJpegLosslessDecoderInstance::~NvJpegLosslessDecoderInstance() {
  DeviceGuard dg(device_id_);
  CUDA_CALL(cudaEventSynchronize(event_));
  CUDA_CALL(nvjpegJpegStreamDestroy(jpeg_stream_));
  CUDA_CALL(nvjpegJpegStateDestroy(state_));
  CUDA_CALL(nvjpegDestroy(nvjpeg_handle_));
}

bool NvJpegLosslessDecoderInstance::CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts,
                                              const ROI &roi) {
  if (opts.format != DALI_ANY_DATA && opts.format != DALI_GRAY) {
    return false;
  }

  try {
    CUDA_CALL(nvjpegJpegStreamParseHeader(nvjpeg_handle_, in->RawData<unsigned char>(), in->Size(),
                                          jpeg_stream_));

    int is_supported = 0;  // 0 means yes
    CUDA_CALL(nvjpegDecodeBatchedSupported(nvjpeg_handle_, jpeg_stream_, &is_supported));
    return is_supported == 0;
  } catch (...) {
    return false;
  }
}

FutureDecodeResults NvJpegLosslessDecoderInstance::ScheduleDecode(DecodeContext ctx,
                                                                  span<SampleView<GPUBackend>> out,
                                                                  cspan<ImageSource *> in,
                                                                  DecodeParams opts,
                                                                  cspan<ROI> rois) {
  int nsamples = in.size();
  assert(out.size() == nsamples);
  assert(rois.empty() || rois.size() == nsamples);
  assert(ctx.tp != nullptr);

  DecodeResultsPromise promise(nsamples);
  auto set_promise = [&](DecodeResult result) {
    for (int i = 0; i < nsamples; i++)
      promise.set(i, result);
  };

  kernels::DynamicScratchpad s({}, ctx.stream);
  try {
    if (opts.format != DALI_ANY_DATA && opts.format != DALI_GRAY)
      throw std::invalid_argument("Only ANY_DATA and GRAY are supported.");

    sample_meta_.clear();
    sample_meta_.resize(nsamples);
    encoded_.clear();
    encoded_.resize(nsamples);
    encoded_len_.clear();
    encoded_len_.resize(nsamples);
    decoded_.clear();
    decoded_.resize(nsamples);

    for (int i = 0; i < nsamples; i++) {
      auto *sample = in[i];
      auto &out_sample = out[i];
      assert(sample->Kind() == InputKind::HostMemory);
      auto *data_ptr = sample->RawData<unsigned char>();
      auto data_size = sample->Size();
      encoded_[i] = data_ptr;
      encoded_len_[i] = data_size;
      sample_meta_[i].needs_processing = opts.dtype != DALI_UINT16;
      if (!rois.empty() && rois[i].use_roi()) {
        sample_meta_[i].needs_processing = true;
      }
      if (opts.use_orientation) {
        auto &ori = sample_meta_[i].orientation = JpegParser().GetInfo(in[i]).orientation;
        if (ori.rotate || ori.flip_x || ori.flip_y)
          sample_meta_[i].needs_processing = true;
      }

      CUDA_CALL(nvjpegJpegStreamParseHeader(nvjpeg_handle_, sample->RawData<unsigned char>(),
                                            sample->Size(), jpeg_stream_));
      unsigned int precision;
      CUDA_CALL(nvjpegJpegStreamGetSamplePrecision(jpeg_stream_, &precision));
      sample_meta_[i].dyn_range_multiplier = DynamicRangeMultiplier(precision, DALI_UINT16);

      auto &o = decoded_[i];
      auto sh = out_sample.shape();
      o.pitch[0] = sh[1] * sh[2] * sizeof(uint16_t);
      if (sample_meta_[i].needs_processing) {
        int64_t nbytes = volume(sh) * sizeof(uint16_t);
        o.channel[0] = s.Allocate<mm::memory_kind::device, uint8_t>(nbytes);
      } else {
        o.channel[0] = static_cast<uint8_t *>(out_sample.raw_mutable_data());
      }
    }

    CUDA_CALL(nvjpegDecodeBatchedInitialize(nvjpeg_handle_, state_, nsamples, 1,
                                            NVJPEG_OUTPUT_UNCHANGEDI_U16));
    CUDA_CALL(nvjpegDecodeBatched(nvjpeg_handle_, state_, encoded_.data(), encoded_len_.data(),
                                  decoded_.data(), ctx.stream));
  } catch (...) {
    set_promise({false, std::current_exception()});
    return promise.get_future();
  }

  Postprocess(promise, ctx, out, opts, rois);
  CUDA_CALL(cudaEventRecord(event_, ctx.stream));
  return promise.get_future();
}

void NvJpegLosslessDecoderInstance::Postprocess(DecodeResultsPromise &promise, DecodeContext ctx,
                                                span<SampleView<GPUBackend>> out, DecodeParams opts,
                                                cspan<ROI> rois) {
  int nsamples = out.size();
  for (int i = 0; i < nsamples; i++) {
    if (!sample_meta_[i].needs_processing) {
      promise.set(i, {true, nullptr});
      continue;
    }
    auto sh = out[i].shape();
    SampleView<GPUBackend> decoded_view(decoded_[i].channel[0], sh, DALI_UINT16);
    DALIImageType decoded_format = sh[2] == 1 ? DALI_GRAY : DALI_ANY_DATA;
    try {
      Convert(out[i], "HWC", opts.format, decoded_view, "HWC", decoded_format,
              ctx.stream, rois.empty() ? ROI{} : rois[i], sample_meta_[i].orientation,
              sample_meta_[i].dyn_range_multiplier);
      promise.set(i, {true, nullptr});
    } catch (...) {
      promise.set(i, {false, std::current_exception()});
    }
  }
}

REGISTER_DECODER("JPEG", NvJpegLosslessDecoderFactory, CUDADecoderPriority - 1);

}  // namespace imgcodec
}  // namespace dali
