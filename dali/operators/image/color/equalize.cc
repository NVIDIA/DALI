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


#include <limits>
#include <opencv2/opencv.hpp>

#include "dali/operators/image/color/equalize.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(experimental__Equalize)
    .DocStr(R"code(Performs grayscale/per-channel histogram equalization.

The supported inputs are images and videos of uint8_t type.)code")
    .NumInput(1)
    .NumOutput(1)
    .InputLayout(0, {"HW", "HWC", "CHW", "FHW", "FHWC", "FCHW"})
    .AllowSequences();

namespace equalize {

class EqualizeCPU : public Equalize<CPUBackend> {
 public:
  explicit EqualizeCPU(const OpSpec &spec) : Equalize<CPUBackend>(spec) {}

 protected:
  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    auto in_view = view<const uint8_t>(input);
    auto out_view = view<uint8_t>(output);
    int sample_dim = in_view.shape.sample_dim();
    // by the check in Equalize::SetupImpl
    assert(input.type() == type2id<uint8_t>::value);
    // enforced by the layouts specified in operator schema
    assert(sample_dim == 2 || sample_dim == 3);
    output.SetLayout(input.GetLayout());
    int num_samples = in_view.num_samples();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto &in_sample = in_view[sample_idx];
      const auto &in_shape = in_sample.shape;
      int64_t num_channels = sample_dim == 2 ? 1 : in_shape[2];
      DALI_ENFORCE(
          1 <= num_channels && num_channels <= CV_CN_MAX,
          make_string("The CPU equalize operator supports images with number of channels in [1, ",
                      CV_CN_MAX, "] channels. However, the sample at index ", sample_idx, " has ",
                      num_channels, " channels."));
      DALI_ENFORCE(in_shape[0] <= std::numeric_limits<int>::max() &&
                       in_shape[1] <= std::numeric_limits<int>::max(),
                   make_string("The image height and width must not exceed the ",
                               std::numeric_limits<int>::max(), ". However, the sample at index ",
                               sample_idx, " has shape ", in_shape, "."));
    }
    auto &tp = ws.GetThreadPool();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto out_sample = out_view[sample_idx];
      auto in_sample = in_view[sample_idx];
      tp.AddWork([this, out_sample, in_sample](int) { RunSample(out_sample, in_sample); },
                 in_sample.shape.num_elements());
    }
    tp.RunAll();
  }

  template <int ndim>
  void RunSample(TensorView<StorageCPU, uint8_t, ndim> out_sample,
                 TensorView<StorageCPU, const uint8_t, ndim> in_sample) {
    auto &in_sample_shape = in_sample.shape;
    int sample_dim = in_sample_shape.sample_dim();
    int num_channels = sample_dim == 2 ? 1 : in_sample.shape[2];
    int channel_flag = CV_8UC(num_channels);
    int height = in_sample_shape[0], width = in_sample_shape[1];
    const cv::Mat cv_img = CreateMatFromPtr(height, width, channel_flag, in_sample.data);
    cv::Mat out_img = CreateMatFromPtr(height, width, channel_flag, out_sample.data);
    if (num_channels == 1) {
      cv::equalizeHist(cv_img, out_img);
    } else {
      std::vector<cv::Mat> channels(num_channels);
      cv::split(cv_img, channels.data());
      for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        cv::equalizeHist(channels[channel_idx], channels[channel_idx]);
      }
      cv::merge(channels.data(), num_channels, out_img);
    }
  }
};

}  // namespace equalize

DALI_REGISTER_OPERATOR(experimental__Equalize, equalize::EqualizeCPU, CPU);

}  // namespace dali
