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

using DALIBayerPattern = dali::kernels::debayer::DALIBayerPattern;

using DALIDebayerAlgorithm = dali::kernels::debayer::DALIDebayerAlgorithm;

cv::ColorConversionCodes toOpenCVColorConversionCode(DALIBayerPattern pattern,
                                                     DALIDebayerAlgorithm algo) {
  switch (algo) {
    // OpenCV Bilinear
    case DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_OCV:
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
          break;
      }
    // OpenCV Edge-aware
    case DALIDebayerAlgorithm::DALI_DEBAYER_EDGEAWARE_OCV:
      switch (pattern) {
        case DALIBayerPattern::DALI_BAYER_BG:
          return cv::COLOR_BayerBG2RGB_EA;
        case DALIBayerPattern::DALI_BAYER_GB:
          return cv::COLOR_BayerGB2RGB_EA;
        case DALIBayerPattern::DALI_BAYER_GR:
          return cv::COLOR_BayerGR2RGB_EA;
        case DALIBayerPattern::DALI_BAYER_RG:
          return cv::COLOR_BayerRG2RGB_EA;
        default:
          break;
      }
    // OpenCV VNG
    case DALIDebayerAlgorithm::DALI_DEBAYER_VNG_OCV:
      switch (pattern) {
        case DALIBayerPattern::DALI_BAYER_BG:
          return cv::COLOR_BayerBG2RGB_VNG;
        case DALIBayerPattern::DALI_BAYER_GB:
          return cv::COLOR_BayerGB2RGB_VNG;
        case DALIBayerPattern::DALI_BAYER_GR:
          return cv::COLOR_BayerGR2RGB_VNG;
        case DALIBayerPattern::DALI_BAYER_RG:
          return cv::COLOR_BayerRG2RGB_VNG;
        default:
          break;
      }
    // OpenCV Gray
    case DALIDebayerAlgorithm::DALI_DEBAYER_GRAY_OCV:
      switch (pattern) {
        case DALIBayerPattern::DALI_BAYER_BG:
          return cv::COLOR_BayerBG2GRAY;
        case DALIBayerPattern::DALI_BAYER_GB:
          return cv::COLOR_BayerGB2GRAY;
        case DALIBayerPattern::DALI_BAYER_GR:
          return cv::COLOR_BayerGR2GRAY;
        case DALIBayerPattern::DALI_BAYER_RG:
          return cv::COLOR_BayerRG2GRAY;
        default:
          break;
      }
    default:
      break;
  }
  DALI_FAIL("Unsupported bayer pattern code " + to_string(pattern) + "and algorithm code " +
            to_string(algo) + " combination for OpenCV debayering.");
}

class DebayerCPU : public Debayer<CPUBackend> {
 public:
  explicit DebayerCPU(const OpSpec &spec) : Debayer<CPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    // If the algorithm is set to default, use bilinear ocv
    if (alg_ == debayer::DALIDebayerAlgorithm::DALI_DEBAYER_DEFAULT) {
      alg_ = debayer::DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_OCV;
    }
    DALI_ENFORCE(alg_ != debayer::DALIDebayerAlgorithm::DALI_DEBAYER_DEFAULT_NPP,
                 "default_npp algorithm is not supported on CPU.");
    if (alg_ == debayer::DALIDebayerAlgorithm::DALI_DEBAYER_VNG_OCV) {
      DALI_ENFORCE(ws.Input<CPUBackend>(0).type() == DALI_UINT8,
                   "VNG debayering only supported with UINT8.");
    }
    return Debayer<CPUBackend>::SetupImpl(output_desc, ws);
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    output.SetLayout("HWC");

    auto &tPool = ws.GetThreadPool();
    for (int i = 0; i < input.num_samples(); ++i) {
      tPool.AddWork([&, i](int) {
        const auto inImage = input[i];
        auto outImage = output[i];

        const auto &oShape = outImage.shape();
        const auto height = static_cast<int>(oShape[0]);
        const auto width = static_cast<int>(oShape[1]);
        const auto outChannels = static_cast<int>(oShape[2]);
        cv::Mat inImg(height, width, OCVMatTypeForDALIData(inImage.type(), 1),
                      const_cast<void *>(inImage.raw_data()));
        cv::Mat outImg(height, width, OCVMatTypeForDALIData(outImage.type(), outChannels),
                       outImage.raw_mutable_data());
        cv::demosaicing(inImg, outImg, toOpenCVColorConversionCode(pattern_[i], alg_));
      });
    }
    tPool.RunAll();
  }
};

DALI_REGISTER_OPERATOR(experimental__Debayer, DebayerCPU, CPU);

}  // namespace dali
