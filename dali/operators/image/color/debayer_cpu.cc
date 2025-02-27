// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <opencv2/imgproc.hpp>

#include "dali/kernels/imgproc/color_manipulation/debayer/debayer.h"
#include "dali/operators/image/color/debayer.h"
#include "dali/util/ocv.h"

namespace dali {

using namespace dali::kernels::debayer;

[[nodiscard]] cv::ColorConversionCodes toOpenCVColorConversionCode(DALIBayerPattern pattern) {
  switch (pattern) {
    case DALIBayerPattern::DALI_BAYER_BG:
      return cv::COLOR_BayerBG2RGB;
    case DALIBayerPattern::DALI_BAYER_GB:
      return cv::COLOR_BayerGB2RGB;
    case DALIBayerPattern::DALI_BAYER_GR:
      return cv::COLOR_BayerGR2RGB;
    case DALIBayerPattern::DALI_BAYER_RG:
      return cv::COLOR_BayerRG2RGB;
    default:
      DALI_FAIL("Unsupported bayer pattern code " + to_string(pattern));
  }
}

class DebayerCPU : public Debayer<CPUBackend> {
 public:
  explicit DebayerCPU(const OpSpec &spec) : Debayer<CPUBackend>(spec) {}

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    output.SetLayout("HWC");

    auto &tPool = ws.GetThreadPool();
    for (int i = 0; i < input.num_samples(); ++i) {
      tPool.AddWork([&, i](int) {
        const auto inImage = input[i];
        auto outImage = output[i];

        const auto &inShape = inImage.shape();
        const auto height = static_cast<int>(inShape[0]);
        const auto width = static_cast<int>(inShape[1]);
        cv::Mat inImg(height, width, OCVMatTypeForDALIData(inImage.type(), 1),
                      const_cast<void *>(inImage.raw_data()));
        cv::Mat outImg(height, width, OCVMatTypeForDALIData(outImage.type(), 3),
                       outImage.raw_mutable_data());
        cv::demosaicing(inImg, outImg, toOpenCVColorConversionCode(pattern_[i]));
      });
    }
    tPool.RunAll();
  }
};

DALI_REGISTER_OPERATOR(experimental__Debayer, DebayerCPU, CPU);

}  // namespace dali
