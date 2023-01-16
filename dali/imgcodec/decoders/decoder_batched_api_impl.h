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
  explicit BatchedApiDecoderImpl(int device_id, const std::map<std::string, any> &params)
      : ImageDecoderImpl(device_id, params) {}

  using ImageDecoderImpl::CanDecode;
  bool CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts, const ROI &roi) override {
    return false;  // TODO(janton): implement
  }

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
    int nsamples = in.size();
    assert(out.size() == nsamples);
    assert(rois.empty() || rois.size() == nsamples);
    assert(ctx.tp != nullptr);
    DecodeResultsPromise promise(nsamples);
    try {
      auto res = DecodeImplBatch(out, in, opts, rois);
      SetPromise(promise, res);
    } catch (...) {
      SetPromise(promise, DecodeResult::Failure(std::current_exception()));
    }
    return promise.get_future();
  }

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois = {}) override {
    int nsamples = in.size();
    assert(out.size() == nsamples);
    assert(rois.empty() || rois.size() == nsamples);
    assert(ctx.tp != nullptr);
    DecodeResultsPromise promise(nsamples);
    try {
      std::cout << "ScheduleDecode shape " << out[0].shape()[0] << "x" <<  out[0].shape()[1] << "\n";
      auto res = DecodeImplBatch(ctx.stream, out, in, opts, rois);
      SetPromise(promise, res);
    } catch (...) {
      SetPromise(promise, DecodeResult::Failure(std::current_exception()));
    }
    return promise.get_future();
  }

  /**
   * @brief Batch image decode CPU implementation
   *
   * @param out output sample view
   * @param in encoded image source
   * @param opts decoding parameters
   * @param roi region-of-interest
   * @return std::vector<DecodeResult>
   */
  virtual DecodeResult DecodeImplBatch(span<SampleView<CPUBackend>> out,
                                       cspan<ImageSource *> in,
                                       DecodeParams opts,
                                       cspan<ROI> rois) {
    throw std::logic_error("Backend not supported");
  }

  /**
   * @brief Batch image decode GPU implementation
   *
   * @param stream CUDA stream
   * @param out output sample view
   * @param in encoded image source
   * @param opts decoding parameters
   * @param roi region-of-interest
   * @return std::vector<DecodeResult>
   */
  virtual DecodeResult DecodeImplBatch(cudaStream_t stream,
                                       span<SampleView<GPUBackend>> out,
                                       cspan<ImageSource *> in,
                                       DecodeParams opts,
                                       cspan<ROI> rois) {
    throw std::logic_error("Backend not supported");
  }

 private:
  template <typename Backend>
  int64_t TotalVolume(span<SampleView<Backend>> out) const {
    int64_t vol = 0;
    for (auto &o : out) {
      vol += volume(o.shape());
    }
    return vol;
  };

  void SetPromise(DecodeResultsPromise& promise, DecodeResult result) {
    for (int i = 0; i < promise.num_samples(); i++)
      promise.set(i, result);
  };
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_BATCHED_API_IMPL_H_
