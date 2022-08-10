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

#include <vector>
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
  std::vector<DecodeResult> Decode(span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    return ParallelDecodeImpl(out, in, opts, rois);
  }

  std::vector<DecodeResult> Decode(cudaStream_t stream,
                                   span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    return ParallelDecodeImpl(stream, out, in, opts, rois);
  }

  DecodeResult Decode(SampleView<CPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    DecodeResult ret;
    tp_->AddWork([&](int tid) {
      ret = DecodeImplTask(tid, out, in, opts, roi);
    }, volume(out.shape()));
    tp_->RunAll();
    return ret;
  }

  DecodeResult Decode(cudaStream_t stream,
                      SampleView<GPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi) override {
    DecodeResult ret;
    tp_->AddWork([&](int tid) {
      ret = DecodeImplTask(tid, stream, out, in, opts, roi);
    }, volume(out.shape()));
    tp_->RunAll();
    return ret;
  }

  std::vector<DecodeResult> ParallelDecodeImpl(span<SampleView<CPUBackend>> out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois) {
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    std::vector<DecodeResult> ret(out.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      tp_->AddWork([&, i](int tid) {
        ret[i] = DecodeImplTask(tid, out[i], in[i], opts, rois.empty() ? no_roi : rois[i]);
      }, volume(out[i].shape()));
    }
    tp_->RunAll();
    return ret;
  }

  std::vector<DecodeResult> ParallelDecodeImpl(cudaStream_t stream,
                                               span<SampleView<GPUBackend>> out,
                                               cspan<ImageSource *> in,
                                               DecodeParams opts,
                                               cspan<ROI> rois) {
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    std::vector<DecodeResult> ret(out.size());
    ROI no_roi;
    for (int i = 0; i < in.size(); i++) {
      tp_->AddWork([&, i](int tid) {
        ret[i] = DecodeImplTask(tid, stream, out[i], in[i], opts, rois.empty() ? no_roi : rois[i]);
      }, volume(out[i].shape()));
    }
    tp_->RunAll();
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
