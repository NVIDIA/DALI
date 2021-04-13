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
#include "dali/operators/image/distortion/jpeg_compression_distortion_op.h"
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_kernel.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {

class JpegCompressionDistortionGPU : public JpegCompressionDistortion<GPUBackend> {
 public:
  explicit JpegCompressionDistortionGPU(const OpSpec &spec)
    : JpegCompressionDistortion(spec) {
      kmgr_.Initialize<JpegDistortionKernel>();
      kmgr_.Resize<JpegDistortionKernel>(1, 1);
  }

  using Operator<GPUBackend>::RunImpl;

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override;

 private:
  using JpegDistortionKernel = kernels::jpeg::JpegCompressionDistortionGPU;
  kernels::KernelManager kmgr_;
  std::vector<int> quality_;
};

void JpegCompressionDistortionGPU::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  auto &output = ws.OutputRef<GPUBackend>(0);
  auto in_view = view<const uint8_t, 3>(input);
  auto out_view = view<uint8_t, 3>(output);
  int nsamples = in_view.shape.size();

  quality_.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    quality_[i] = quality_arg_[i].data[0];
  }
  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto req = kmgr_.Setup<JpegDistortionKernel>(0, ctx, in_view.shape, true, true);
  kmgr_.Run<JpegDistortionKernel>(0, 0, ctx, out_view, in_view, make_cspan(quality_));
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionGPU, GPU);

}  // namespace dali
