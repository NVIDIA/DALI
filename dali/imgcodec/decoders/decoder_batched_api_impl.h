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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_BATCHED_API_IMPL_H_
#define DALI_IMGCODEC_DECODERS_DECODER_BATCHED_API_IMPL_H_

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <thread>
#include "dali/core/format.h"
#include "dali/imgcodec/decoders/decoder_impl.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {
namespace imgcodec {

/**
 * @brief A skeleton for implementing a batch-API decoder
 *
 */
class DLL_PUBLIC BatchedApiDecoderImpl : public ImageDecoderImpl {
 public:
  explicit BatchedApiDecoderImpl(int device_id, const std::map<std::string, std::any> &params)
      : ImageDecoderImpl(device_id, params) {}

  using ImageDecoderImpl::Decode;
  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    return ScheduleDecode(std::move(ctx), out, in, opts, rois).get_all();
  }

  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    return ScheduleDecode(ctx, out, in, opts, rois).get_all();
  }

  DecodeResult Decode(DecodeContext ctx, SampleView<CPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), out, in, opts, roi).get_one(0);
  }

  DecodeResult Decode(DecodeContext ctx, SampleView<GPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), out, in, opts, roi).get_one(0);
  }

  FutureDecodeResults ScheduleDecode(DecodeContext ctx, SampleView<CPUBackend> out,
                                     ImageSource *in, DecodeParams opts,
                                     const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), make_span(&out, 1), make_cspan(&in, 1), std::move(opts),
                          make_cspan(&roi, 1));
  }

  FutureDecodeResults ScheduleDecode(DecodeContext ctx, SampleView<GPUBackend> out,
                                     ImageSource *in, DecodeParams opts,
                                     const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), make_span(&out, 1), make_cspan(&in, 1), std::move(opts),
                          make_cspan(&roi, 1));
  }

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<CPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois = {}) override {
    throw std::logic_error("Backend not supported");
  }

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois = {}) override {
    throw std::logic_error("Backend not supported");
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_BATCHED_API_IMPL_H_
