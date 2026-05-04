// Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_cpu_kernel.h"
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
};

void JpegCompressionDistortionCPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  auto layout = input.GetLayout();
  output.SetLayout(layout);
  auto in_shape = input.shape();
  int nsamples = input.num_samples();
  auto &thread_pool = ws.GetThreadPool();
  auto in_view = view<const uint8_t>(input);
  auto out_view = view<uint8_t>(output);

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

    // f_dim == -1 for HWC: nframes = volume([], [0]) = 1.
    int64_t nframes = volume(shape.begin(), shape.begin() + f_dim + 1);
    int64_t frame_size = volume(shape.begin() + f_dim + 1, shape.begin() + ndim);
    int64_t width = shape[w_dim];
    int64_t height = shape[h_dim];

    for (int frame_idx = 0; frame_idx < nframes; frame_idx++) {
      thread_pool.AddWork(
          [&, sample_idx, frame_idx, width, height, frame_size,
           quality = quality_arg_[sample_idx].data[0]](int) {
            const uint8_t *in_ptr = in_view[sample_idx].data + frame_idx * frame_size;
            uint8_t *out_ptr = out_view[sample_idx].data + frame_idx * frame_size;
            TensorShape<3> sh{static_cast<int64_t>(height), static_cast<int64_t>(width), 3};
            TensorView<StorageCPU, const uint8_t, 3> in_tv{in_ptr, sh};
            TensorView<StorageCPU, uint8_t, 3> out_tv{out_ptr, sh};
            kernels::jpeg::JpegCompressionDistortionCPU kernel;
            kernel.RunSample(out_tv, in_tv, quality,
                             /*horz_subsample=*/true, /*vert_subsample=*/true);
          },
          frame_size);
    }
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionCPU, CPU);

}  // namespace dali
