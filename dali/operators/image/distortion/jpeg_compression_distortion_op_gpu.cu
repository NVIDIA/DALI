// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include "dali/kernels/imgproc/jpeg/jpeg_distortion_gpu_kernel.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/distortion/jpeg_compression_distortion_op.h"

namespace dali {

class JpegCompressionDistortionGPU : public JpegCompressionDistortion<GPUBackend> {
 public:
  explicit JpegCompressionDistortionGPU(const OpSpec &spec) : JpegCompressionDistortion(spec) {
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

template <int ndim>
TensorListShape<ndim - 1> unfold_outer_dim(const TensorListShape<ndim> &shapes) {
  static_assert(ndim > 1,
                "Can't reduce dimentionality of dynamic, or single dimentional TensorListShape");

  constexpr static int out_dim = ndim - 1;

  using OutShapeType = TensorShape<out_dim>;
  std::vector<OutShapeType> result;

  size_t nshapes = 0;

  for (int i = 0; i < shapes.size(); ++i) {
    nshapes += shapes[i][0];
  }

  result.reserve(nshapes);

  for (int i = 0; i < shapes.size(); ++i) {
    auto shape = shapes[i];
    auto nouter_dim = shape[0];
    auto subshape = shape.last(out_dim).template to_static<out_dim>();
    for (int j = 0; j < nouter_dim; j++) {
      result.push_back(subshape);
    }
  }

  return result;
}

template <typename Type>
TensorListView<StorageGPU, Type, 3> frames_to_samples(
    const TensorListView<StorageGPU, Type, 4> &view) {
  TensorListShape<3> new_shape = unfold_outer_dim(view.shape);
  return reinterpret<Type>(view, std::move(new_shape));
}

void JpegCompressionDistortionGPU::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  const auto layout = input.GetLayout();
  const int nsamples = input.num_samples();

  TensorListView<StorageGPU, const uint8_t, 3> in_view;
  TensorListView<StorageGPU, uint8_t, 3> out_view;

  const bool is_sequence = layout.size() == 4;

  if (is_sequence) {
    in_view = frames_to_samples(view<const uint8_t, 4>(input));
    out_view = frames_to_samples(view<uint8_t, 4>(output));
    quality_.resize(in_view.size());
  } else {
    in_view = view<const uint8_t, 3>(input);
    out_view = view<uint8_t, 3>(output);
    quality_.resize(in_view.size());
  }

  // Set quality argument for an image from samples
  if (is_sequence) {
    for (int i = 0; i < nsamples; i++) {
      auto nframes = input.tensor_shape_span(i)[0];
      for (int j = 0; j < nframes; ++j) {
        quality_[i] = quality_arg_[i].data[0];
      }
    }
  } else {
    for (int i = 0; i < nsamples; i++) {
      quality_[i] = quality_arg_[i].data[0];
    }
  }

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  auto req = kmgr_.Setup<JpegDistortionKernel>(0, ctx, in_view.shape, true, true);
  kmgr_.Run<JpegDistortionKernel>(0, 0, ctx, out_view, in_view, make_cspan(quality_));
}

DALI_REGISTER_OPERATOR(JpegCompressionDistortion, JpegCompressionDistortionGPU, GPU);

}  // namespace dali
