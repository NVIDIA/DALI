// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <opencv2/opencv.hpp>
#include <vector>
#include "dali/operators/image/distortion/jpeg_compression_distortion_op.h"

namespace dali {

DALI_SCHEMA(JpegCompressionDistortion)
    .DocStr(R"code(Introduces JPEG compression artifacts to RGB images.

JPEG is a lossy compression format which exploits characteristics of natural
images and human visual system to achieve high compression ratios. The information
loss originates from sampling the color information at a lower spatial resolution
than the brightness and from representing high frequency components of the image
with a lower effective bit depth. The conversion to frequency domain and quantization
is applied independently to 8x8 pixel blocks, which introduces additional artifacts
at block boundaries.

This operation produces images by subjecting the input to a transformation that
mimics JPEG compression with given `quality` factor followed by decompression.
)code")
    .NumInput(1)
    .InputLayout({"HWC", "FHWC"})
    .NumOutput(1)
    .AddOptionalArg(
        "quality",
        R"code(JPEG compression quality from 1 (lowest quality) to 100 (highest quality).

Any values outside the range 1-100 will be clamped.)code",
        50, true)
    .AllowSequences();

class JpegCompressionDistortionCPU : public JpegCompressionDistortion<CPUBackend> {
 public:
  explicit JpegCompressionDistortionCPU(const OpSpec &spec) : JpegCompressionDistortion(spec) {}
  using Operator<CPUBackend>::RunImpl;

 protected:
  void RunImpl(Workspace &ws) override;

 private:
  struct ThreadCtx {
    std::vector<uint8_t> encoded;
  };
  std::vector<ThreadCtx> thread_ctx_;
};

template <typename ThreadCtx>
static void RunJpegDistortionCPU(ThreadCtx &ctx, const uint8_t *input, uint8_t *output,
                                 size_t width, size_t height, int quality) {
  auto in_mat = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t *>(input));
  auto out_mat = cv::Mat(height, width, CV_8UC3, output);

  cv::cvtColor(in_mat, out_mat, cv::COLOR_RGB2BGR);
  cv::imencode(".jpg", out_mat, ctx.encoded, {cv::IMWRITE_JPEG_QUALITY, quality});
  cv::imdecode(ctx.encoded, cv::IMREAD_COLOR, &out_mat);
  cv::cvtColor(out_mat, out_mat, cv::COLOR_BGR2RGB);
}

void JpegCompressionDistortionCPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  auto layout = input.GetLayout();
  output.SetLayout(layout);
  auto in_shape = input.shape();
  int nsamples = input.num_samples();
  auto& thread_pool = ws.GetThreadPool();
  auto in_view = view<const uint8_t>(input);
  auto out_view = view<uint8_t>(output);

  thread_ctx_.resize(thread_pool.NumThreads());

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto shape = in_shape.tensor_shape_span(sample_idx);
    int ndim = shape.size();

    int w_dim = layout.find('W');
    assert(w_dim >= 0);
    int h_dim = layout.find('H');
    assert(h_dim >= 0);
    int c_dim = layout.find('C');
    assert(c_dim >= 0);
    int f_dim = layout.find('F');

    int64_t nframes =
        volume(&shape[0], &shape[f_dim + 1]);  // note that if f_dim is -1, this evaluates to an
                                               // empty range, which has a volume of 1
    int64_t frame_size = volume(&shape[f_dim + 1], &shape[ndim]);
    int64_t width = shape[w_dim];
    int64_t height = shape[h_dim];
    for (int elem_idx = 0; elem_idx < nframes; elem_idx++) {
      thread_pool.AddWork(
          [&, sample_idx, elem_idx, width, height, frame_size,
           quality = quality_arg_[sample_idx].data[0]](int thread_id) {
            auto *in = in_view[sample_idx].data + elem_idx * frame_size;
            auto *out = out_view[sample_idx].data + elem_idx * frame_size;
            RunJpegDistortionCPU(thread_ctx_[thread_id], in, out, width, height, quality);
          },
          frame_size);
    }
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionCPU, CPU);

}  // namespace dali
