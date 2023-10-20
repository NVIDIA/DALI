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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_
#define DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_

#include <map>
#include <string>
#include <vector>
#include "dali/core/format.h"
#include "dali/imgcodec/image_decoder_interfaces.h"
#include "dali/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC ImageDecoderImpl : public ImageDecoderInstance {
 public:
  explicit ImageDecoderImpl(int device_id, const std::map<std::string, std::any> &)
  : device_id_(device_id) {}

  bool CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts, const ROI &roi) override {
    return true;
  }

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

  /**
   * @brief To be overriden by a CPU codec implementation
   */
  DecodeResult Decode(DecodeContext ctx, SampleView<CPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    throw std::logic_error("Backend not supported");
  }

  std::vector<DecodeResult> Decode(DecodeContext ctx,
                                   span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override {
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    std::vector<DecodeResult> ret(out.size());
    ROI no_roi;
    for (int i = 0 ; i < in.size(); i++)
      ret[i] = Decode(ctx, out[i], in[i], opts, rois.empty() ? no_roi : rois[i]);
    return ret;
  }

  /**
   * @brief To be overriden by a GPU/mixed codec implementation
   */
  DecodeResult Decode(DecodeContext ctx, SampleView<GPUBackend> out, ImageSource *in,
                      DecodeParams opts, const ROI &roi) override {
    throw std::logic_error("Backend not supported");
  }

  std::vector<DecodeResult> Decode(DecodeContext ctx, span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in, DecodeParams opts,
                                   cspan<ROI> rois) override {
    assert(out.size() == in.size());
    assert(rois.empty() || rois.size() == in.size());
    std::vector<DecodeResult> ret(out.size());
    ROI no_roi;
    for (int i = 0 ; i < in.size(); i++)
      ret[i] = Decode(ctx, out[i], in[i], opts, rois.empty() ? no_roi : rois[i]);
    return ret;
  }

  bool SetParam(const char*, const std::any &) override {
    return false;
  }

  std::any GetParam(const char *key) const override {
    return {};
  }

  int SetParams(const std::map<std::string, std::any> &params) override {
    int ret = 0;
    for (auto &[key, value] : params)
      ret += SetParam(key.c_str(), value);
    return ret;
  }

 protected:
  int device_id_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_
