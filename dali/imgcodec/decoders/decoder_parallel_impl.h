// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_PARALLEL_IMPL_H_
#define DALI_IMGCODEC_DECODERS_DECODER_PARALLEL_IMPL_H_

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
 * @brief A skeleton for implementing a batch-parallel decoder
 *
 * This implementation provides the batched implementation
 */
class DLL_PUBLIC BatchParallelDecoderImpl : public ImageDecoderImpl {
 public:
  explicit BatchParallelDecoderImpl(int device_id, const std::map<std::string, std::any> &params)
  : ImageDecoderImpl(device_id, params) {}

  using ImageDecoderImpl::CanDecode;
  std::vector<bool> CanDecode(DecodeContext ctx,
                              cspan<ImageSource *> in,
                              DecodeParams opts,
                              cspan<ROI> rois) override {
    assert(rois.empty() || rois.size() == in.size());
    std::vector<bool> ret(in.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++)
      ret[i] = CanDecode(ctx, in[i], opts, rois.empty() ? no_roi : rois[i]);
    return ret;
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
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    assert(ctx.tp != nullptr);
    DecodeResultsPromise promise(in.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      auto roi = rois.empty() ? no_roi : rois[i];
      ctx.tp->AddWork([=, out = out[i], in = in[i]](int tid) mutable {
          try {
            promise.set(i, DecodeImplTask(tid, out, in, opts, roi));
          } catch (...) {
            promise.set(i, DecodeResult::Failure(std::current_exception()));
          }
        }, volume(out[i].shape()));
    }
    ctx.tp->RunAll(false);
    return promise.get_future();
  }

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois = {}) override {
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    assert(ctx.tp != nullptr);
    DecodeResultsPromise promise(in.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      auto roi = rois.empty() ? no_roi : rois[i];
      ctx.tp->AddWork([=, out = out[i], in = in[i]](int tid) mutable {
          try {
            promise.set(i, DecodeImplTask(tid, ctx.stream, out, in, opts, roi));
          } catch (...) {
            promise.set(i, DecodeResult::Failure(std::current_exception()));
          }
        },
        volume(out[i].shape()));
    }
    ctx.tp->RunAll(false);
    return promise.get_future();
  }

  /**
   * @brief Single image decode CPU implementation, executed in a thread pool context.
   *
   * @param thread_idx thread index in the thread pool
   * @param out output sample view
   * @param in encoded image source
   * @param opts decoding parameters
   * @param roi region-of-interest
   * @return std::vector<DecodeResult>
   */
  virtual DecodeResult DecodeImplTask(int thread_idx,
                                      SampleView<CPUBackend> out,
                                      ImageSource *in,
                                      DecodeParams opts,
                                      const ROI &roi) {
    throw std::logic_error("Backend not supported");
  }

  /**
   * @brief Single image decode GPU implementation, executed in a thread pool context.
   *
   * @param thread_idx thread index in the thread pool
   * @param stream CUDA stream to synchronize with
   * @param out output sample view
   * @param in encoded image source
   * @param opts decoding parameters
   * @param roi region-of-interest
   * @return std::vector<DecodeResult>
   */
  virtual DecodeResult DecodeImplTask(int thread_idx,
                                      cudaStream_t stream,
                                      SampleView<GPUBackend> out,
                                      ImageSource *in,
                                      DecodeParams opts,
                                      const ROI &roi) {
    throw std::logic_error("Backend not supported");
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_PARALLEL_IMPL_H_
