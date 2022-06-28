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

#ifndef DALI_IMGCODEC_CODECS_CODEC_PARALLEL_IMPL_H_
#define DALI_IMGCODEC_CODECS_CODEC_PARALLEL_IMPL_H_

#include <vector>
#include "dali/core/format.h"
#include "dali/imgcodec/codecs/codec_impl.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {
namespace imgcodec {

/**
 * @brief A skeleton for implementing a batch-parallel codec
 *
 * This implementation provides the batched implementation
 *
 * @tparam Actual
 */
template <typename Actual>
class DLL_PUBLIC BatchParallelCodecImpl : public ImageCodecImpl<Actual> {
 public:
  BatchParallelCodecImpl(int device_id, ThreadPool *tp) : ImageCodecImpl<Actual>(device_id, tp) {
  }

  using ImageCodecImpl<Actual>::CanDecode;
  std::vector<bool> CanDecode(cspan<ImageSource *> in, cspan<DecodeParams> opts) override{
    assert(opts.size() == in.size());
    std::vector<bool> ret(in.size());
    for (int i = 0; i < in.size(); i++) {
      tp_->AddWork([&, i](int) {
        ret[i] = CanDecode(in[i], opts[i]);
      });
    }
    tp_->RunAll();
    return ret;
  }

  using ImageCodecImpl<Actual>::Decode;
  std::vector<DecodeResult> Decode(span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in, cspan<DecodeParams> opts) override {
    return ParallelDecodeImpl(out, in, opts);
  }

  std::vector<DecodeResult> Decode(span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in, cspan<DecodeParams> opts) override {
    return ParallelDecodeImpl(out, in, opts);
  }

  template <typename Backend>
  std::vector<DecodeResult> ParallelDecodeImpl(span<SampleView<Backend>> out,
                                               cspan<ImageSource *> in,
                                               cspan<DecodeParams> opts) {
    assert(out.size() == in.size());
    assert(out.size() == opts.size() || opts.size() == 1);
    std::vector<DecodeResult> ret(out.size());
    for (int i = 0; i < in.size(); i++) {
      tp_->AddWork([&, i](int) {
        ret[i] = Decode(out[i], in[i], opts.size() > 1 ? opts[i] : opts[0]);
      });
    }
    tp_->RunAll();
    return ret;
  }

 protected:
  using ImageCodecImpl<Actual>::tp_;
  using ImageCodecImpl<Actual>::device_id_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_CODECS_CODEC_PARALLEL_IMPL_H_
