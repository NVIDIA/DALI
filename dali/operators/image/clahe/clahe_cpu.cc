// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <opencv2/opencv.hpp>

#include "dali/core/error_handling.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/util/ocv.h"

namespace dali {

// -----------------------------------------------------------------------------
// CPU CLAHE Operator using OpenCV
// -----------------------------------------------------------------------------
class ClaheCPU : public Operator<CPUBackend> {
 public:
  explicit ClaheCPU(const OpSpec &spec)
      : Operator<CPUBackend>(spec),
        tiles_x_(spec.GetArgument<int>("tiles_x")),
        tiles_y_(spec.GetArgument<int>("tiles_y")),
        clip_limit_(spec.GetArgument<float>("clip_limit")),
        luma_only_(spec.GetArgument<bool>("luma_only")) {
    // Create OpenCV CLAHE object with specified parameters
    clahe_ = cv::createCLAHE(clip_limit_, cv::Size(tiles_x_, tiles_y_));
  }

  bool SetupImpl(std::vector<OutputDesc> &outputs, const Workspace &ws) override {
    const auto &in = ws.Input<CPUBackend>(0);

    if (in.type() != DALI_UINT8) {
      throw std::invalid_argument("ClaheCPU currently supports only uint8 input.");
    }

    outputs.resize(1);
    outputs[0].type = in.type();
    outputs[0].shape = in.shape();  // same layout/shape as input
    return true;
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    auto in_view = view<const uint8_t>(input);
    auto out_view = view<uint8_t>(output);

    int ndim = in_view.shape.sample_dim();
    if (ndim != 2 && ndim != 3) {
      throw std::invalid_argument("ClaheCPU expects HW (grayscale) or HWC (color) input layout.");
    }

    // Warn user about RGB channel order requirement for RGB images
    static bool warned_rgb_order = false;
    if (luma_only_ && !warned_rgb_order && ndim == 3) {
      // Check if we have any RGB samples (3 channels)
      bool has_rgb = false;
      for (int i = 0; i < in_view.num_samples(); i++) {
        if (in_view[i].shape.size() == 3 && in_view[i].shape[2] == 3) {
          has_rgb = true;
          break;
        }
      }
      if (has_rgb) {
        DALI_WARN("CRITICAL: CLAHE expects RGB channel order (Red, Green, Blue). "
                  "If your images are in BGR order (common with OpenCV cv2.imread), "
                  "the luminance calculation will be INCORRECT. "
                  "Convert BGR to RGB using fn.reinterpret or similar operators before CLAHE.");
        warned_rgb_order = true;
      }
    }

    auto &tp = ws.GetThreadPool();
    int num_samples = in_view.num_samples();

    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      tp.AddWork([this, &in_view, &out_view, sample_idx](int) {
        // Create a thread-local CLAHE object to avoid race conditions
        // OpenCV CLAHE objects are not thread-safe
        auto local_clahe = cv::createCLAHE(clip_limit_, cv::Size(tiles_x_, tiles_y_));
        ProcessSample(out_view[sample_idx], in_view[sample_idx], local_clahe);
      }, in_view[sample_idx].shape.num_elements());
    }
    tp.RunAll();
  }

 private:
  template <int ndim>
  void ProcessSample(TensorView<StorageCPU, uint8_t, ndim> out_sample,
                     TensorView<StorageCPU, const uint8_t, ndim> in_sample,
                     cv::Ptr<cv::CLAHE> clahe) {
    auto &shape = in_sample.shape;
    int H = shape[0];
    int W = shape[1];
    int C = (shape.size() >= 3) ? shape[2] : 1;

    if (C != 1 && C != 3) {
      throw std::invalid_argument("ClaheCPU supports 1 or 3 channels.");
    }

    if (C == 1) {
      // Grayscale processing
      cv::Mat src(H, W, CV_8UC1, const_cast<uint8_t *>(in_sample.data));
      cv::Mat dst(H, W, CV_8UC1, out_sample.data);
      clahe->apply(src, dst);
    } else {
      // RGB processing
      cv::Mat src(H, W, CV_8UC3, const_cast<uint8_t *>(in_sample.data));
      cv::Mat dst(H, W, CV_8UC3, out_sample.data);

      if (luma_only_) {
        // Apply CLAHE to luminance channel only (preserves color relationships)
        cv::Mat lab, lab_dst;
        cv::cvtColor(src, lab, cv::COLOR_RGB2Lab);

        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);

        // Apply CLAHE to L (luminance) channel
        clahe->apply(lab_channels[0], lab_channels[0]);

        cv::merge(lab_channels, lab_dst);
        cv::cvtColor(lab_dst, dst, cv::COLOR_Lab2RGB);
      } else {
        // Apply CLAHE to each channel independently
        std::vector<cv::Mat> channels;
        cv::split(src, channels);

        for (auto &channel : channels) {
          clahe->apply(channel, channel);
        }

        cv::merge(channels, dst);
      }
    }
  }

  int tiles_x_, tiles_y_;
  float clip_limit_;
  bool luma_only_;
  cv::Ptr<cv::CLAHE> clahe_;
};

DALI_REGISTER_OPERATOR(Clahe, ClaheCPU, CPU);

}  // namespace dali
