// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <opencv2/opencv.hpp>
#include "dali/operators/image/distortion/jpeg_compression_distortion_op.h"

namespace dali {

DALI_SCHEMA(JpegCompressionDistortion)
    .DocStr(R"code(Produces JPEG-like distortion in RGB images.

The level of degradation of the image can be controlled with the ``quality`` argument,
)code")
    .NumInput(1)
    .InputLayout(0, "HWC")
    .NumOutput(1)
    .AddOptionalArg("quality",
        R"code(JPEG compression quality from 1 (lowest quality) to 100 (highest quality).

Any values outside the range 1-100 will be clamped.)code",
                    95, true);

class JpegCompressionDistortionCPU : public JpegCompressionDistortion<CPUBackend> {
 public:
  explicit JpegCompressionDistortionCPU(const OpSpec &spec) : JpegCompressionDistortion(spec) {}
  using Operator<CPUBackend>::RunImpl;

 protected:
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  struct ThreadCtx {
    std::vector<uint8_t> encoded;
  };
  std::vector<ThreadCtx> thread_ctx_;
};

void JpegCompressionDistortionCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto in_shape = input.shape();
  int nsamples = input.size();
  auto& thread_pool = ws.GetThreadPool();
  auto in_view = view<const uint8_t>(input);
  auto out_view = view<uint8_t>(output);

  thread_ctx_.resize(thread_pool.NumThreads());

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    thread_pool.AddWork(
      [&, sample_idx, quality = quality_arg_[sample_idx].data[0]](int thread_id) {
        auto &ctx = thread_ctx_[thread_id];
        auto sh = in_shape.tensor_shape_span(sample_idx);
        cv::Mat in_mat(sh[0], sh[1], CV_8UC3, (void*) in_view[sample_idx].data);  // NOLINT
        cv::Mat out_mat(sh[0], sh[1], CV_8UC3, (void*) out_view[sample_idx].data);  // NOLINT
        cv::cvtColor(in_mat, out_mat, cv::COLOR_RGB2BGR);
        cv::imencode(".jpg", out_mat, ctx.encoded, {cv::IMWRITE_JPEG_QUALITY, quality});
        cv::imdecode(ctx.encoded, cv::IMREAD_COLOR, &out_mat);
        cv::cvtColor(out_mat, out_mat, cv::COLOR_BGR2RGB);
      }, in_shape.tensor_size(sample_idx));
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionCPU, CPU);

}  // namespace dali
