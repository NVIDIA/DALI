// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <utility>
#include <vector>
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
  BatchParallelDecoderImpl(int device_id, ThreadPool *tp)
  : ImageDecoderImpl(device_id, tp) {}

  using ImageDecoderImpl::CanDecode;
  std::vector<bool> CanDecode(cspan<ImageSource *> in,
                              DecodeParams opts,
                              cspan<ROI> rois) override {
    assert(rois.empty() || rois.size() == in.size());
    std::vector<bool> ret(in.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      tp_->AddWork([&, i](int) {
        ret[i] = CanDecode(in[i], opts, rois.empty() ? no_roi : rois[i]);
      });
    }
    tp_->RunAll();
    return ret;
  }

  using ImageDecoderImpl::Decode;
  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    auto results = ScheduleDecode(std::move(ctx), out, in, opts, rois);
    std::vector<DecodeResult> ret(out.size());
    for (int i = 0; i < out.size(); i++)
      ret[i] = results[i].get();
    return ret;
  }

  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    auto results = ScheduleDecode(std::move(ctx), out, in, opts, rois);
    std::vector<DecodeResult> ret(out.size());
    for (int i = 0; i < out.size(); i++)
      ret[i] = results[i].get();
    return ret;
  }

  DecodeResult Decode(DecodeContext ctx, SampleView<CPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), out, in, opts, roi)[0].get();
  }

  DecodeResult Decode(DecodeContext ctx, SampleView<GPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), out, in, opts, roi)[0].get();
  }

  DeferredDecodeResults ScheduleDecode(DecodeContext ctx, SampleView<CPUBackend> out,
                                       ImageSource *in, DecodeParams opts,
                                       const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), make_span(&out, 1), make_cspan(&in, 1), std::move(opts),
                          make_cspan(&roi, 1));
  }

  DeferredDecodeResults ScheduleDecode(DecodeContext ctx, SampleView<GPUBackend> out,
                                       ImageSource *in, DecodeParams opts,
                                       const ROI &roi) override {
    return ScheduleDecode(std::move(ctx), make_span(&out, 1), make_cspan(&in, 1), std::move(opts),
                          make_cspan(&roi, 1));
  }

  DeferredDecodeResults ScheduleDecode(DecodeContext ctx,
                                       span<SampleView<CPUBackend>> out,
                                       cspan<ImageSource *> in,
                                       DecodeParams opts,
                                       cspan<ROI> rois = {}) override {
    DALI_ENFORCE(ctx.tp, "This implementation expects a valid thread pool pointer in the context");
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    assert(ctx.tp != nullptr);
    DeferredDecodeResults ret;
    ret.reserve(in.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      auto p = std::make_shared<std::promise<DecodeResult>>();
      ret.push_back(p->get_future());
      auto roi = rois.empty() ? no_roi : rois[i];
      ctx.tp->AddWork([=, out = out[i], in = in[i]](int tid) {
          p->set_value(
            DecodeImplTask(tid, out, in, opts, roi));
        }, volume(out[i].shape()));
    }
    bool should_wait = false;
    tp_->RunAll(should_wait);
    return ret;
  }

  DeferredDecodeResults ScheduleDecode(DecodeContext ctx,
                                       span<SampleView<GPUBackend>> out,
                                       cspan<ImageSource *> in,
                                       DecodeParams opts,
                                       cspan<ROI> rois = {}) override {
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    assert(ctx.tp != nullptr);
    DeferredDecodeResults ret;
    ret.reserve(in.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      auto p = std::make_shared<std::promise<DecodeResult>>();
      ret.push_back(p->get_future());
      auto roi = rois.empty() ? no_roi : rois[i];
      ctx.tp->AddWork([=, out = out[i], in = in[i]](int tid) {
          p->set_value(
            DecodeImplTask(tid, ctx.stream, out, in, opts, roi));
        },
        volume(out[i].shape()));
    }
    bool should_wait = false;
    tp_->RunAll(should_wait);
    return ret;
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
