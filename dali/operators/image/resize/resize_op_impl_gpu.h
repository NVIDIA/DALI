// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_GPU_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_GPU_H_

#ifndef DALI_RESIZE_BASE_CC
#error This file is a part of resize base implementation and should not be included elsewhere
#endif

#include <cassert>
#include <vector>
#include "dali/operators/image/resize/resize_op_impl.h"
#include "dali/kernels/imgproc/resample.h"

namespace dali {

template <typename Out, typename In, int spatial_ndim>
class ResizeOpImplGPU : public ResizeBase<GPUBackend>::Impl {
 public:
  explicit ResizeOpImplGPU(kernels::KernelManager &kmgr, int minibatch_size)
  : kmgr_(kmgr), minibatch_size_(minibatch_size) {
    kmgr_.Resize(kmgr_.NumThreads(), 0);
  }

  static_assert(spatial_ndim == 2 || spatial_ndim == 3, "Only 2D and 3D resizing is supported");

  using Kernel = kernels::ResampleGPU<Out, In, spatial_ndim>;

  /// Dimensionality of each separate frame. If input contains no channel dimension, one is added
  static constexpr int frame_ndim = spatial_ndim + 1;

  void Setup(TensorListShape<> &out_shape,
             const TensorListShape<> &in_shape,
             int first_spatial_dim,
             span<const kernels::ResamplingParams> params) override {
    // Calculate output shape of the input, as supplied (sequences, planar images, etc)
    GetResizedShape(out_shape, in_shape, params, spatial_ndim, first_spatial_dim);

    // Create "frames" from outer dimensions and "channels" from inner dimensions.
    GetFrameShapesAndParams<spatial_ndim>(in_shape_, params_, in_shape, params,
                                          first_spatial_dim);

    // Now that we have per-frame parameters, we can calculate the output shape of the
    // effective frames (from videos, channel planes, etc).
    GetResizedShape(out_shape_, in_shape_, make_cspan(params_), 0);

    // Now that we know how many logical frames there are, calculate batch subdivision.
    SetNumFrames(in_shape_.num_samples());

    SetupKernel();
  }

  void SetupKernel() {
    const int dim = in_shape_.sample_dim();
    kernels::KernelContext ctx;
    for (int mb_idx = 0, num_mb = minibatches_.size(); mb_idx < num_mb; mb_idx++) {
      auto &mb = minibatches_[mb_idx];
      auto &in_slice = mb.input;
      in_slice.shape.resize(mb.count, dim);
      int end = mb.start + mb.count;
      for (int i = mb.start, j = 0; i < end; i++, j++) {
        for (int d = 0; d < dim; d++)
          in_slice.tensor_shape_span(j)[d] = in_shape_.tensor_shape_span(i)[d];
      }

      auto param_slice = make_span(&params_[mb.start], mb.count);
      kernels::KernelRequirements &req = kmgr_.Setup<Kernel>(mb_idx, ctx, mb.input, param_slice);
      mb.out_shape = req.output_shapes[0].to_static<frame_ndim>();
    }
  }

  void RunResize(DeviceWorkspace &ws,
                 TensorList<GPUBackend> &output,
                 const TensorList<GPUBackend> &input) override {
    auto in_view = view<const In>(input);
    auto in_frames_view = reshape(in_view, in_shape_, true);
    SubdivideInput(in_frames_view);

    auto out_view = view<Out>(output);
    auto out_frames_view = reshape(out_view, out_shape_, true);
    SubdivideOutput(out_frames_view);

    kernels::KernelContext context;
    context.gpu.stream = ws.stream();

    for (size_t b = 0; b < minibatches_.size(); b++) {
      MiniBatch &mb = minibatches_[b];

      kmgr_.Run<Kernel>(0, b, context,
          mb.output, mb.input, make_span(params_.data() + mb.start, mb.count));
    }
  }

  void SetNumFrames(int n) {
    int num_minibatches = CalculateMinibatchPartition(n, minibatch_size_);
    if (static_cast<int>(kmgr_.NumInstances()) < num_minibatches)
      kmgr_.Resize<Kernel>(1, num_minibatches);
  }

  int CalculateMinibatchPartition(int total_frames, int minibatch_size) {
    int num_minibatches = div_ceil(total_frames, minibatch_size);

    minibatches_.resize(num_minibatches);
    int start = 0;
    for (int i = 0; i < num_minibatches; i++) {
      int end = (i + 1) * total_frames / num_minibatches;
      auto &mb = minibatches_[i];
      mb.start = start;
      mb.count = end - start;
      start = end;
    }
    return num_minibatches;
  }

  TensorListShape<frame_ndim> in_shape_, out_shape_;
  std::vector<ResamplingParamsND<spatial_ndim>> params_;

  kernels::KernelManager &kmgr_;

  struct MiniBatch {
    int start, count;
    TensorListShape<> out_shape;
    kernels::InListGPU<In, frame_ndim> input;
    kernels::OutListGPU<Out, frame_ndim> output;
  };

  std::vector<MiniBatch> minibatches_;

  void SubdivideInput(const kernels::InListGPU<In, frame_ndim> &in) {
    for (auto &mb : minibatches_)
      sample_range(mb.input, in, mb.start, mb.start + mb.count);
  }

  void SubdivideOutput(const kernels::OutListGPU<Out, frame_ndim> &out) {
    for (auto &mb : minibatches_)
      sample_range(mb.output, out, mb.start, mb.start + mb.count);
  }

  int minibatch_size_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_GPU_H_
