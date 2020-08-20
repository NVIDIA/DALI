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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_CPU_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_CPU_H_

#ifndef DALI_RESIZE_BASE_CC
#error This file is a part of resize base implementation and should not be included elsewhere
#endif

#include <cassert>
#include <cmath>
#include <vector>
#include "dali/operators/image/resize/resize_op_impl.h"
#include "dali/kernels/imgproc/resample_cpu.h"

namespace dali {

template <typename Out, typename In, int spatial_ndim>
class ResizeOpImplCPU : public ResizeBase<CPUBackend>::Impl {
 public:
  explicit ResizeOpImplCPU(kernels::KernelManager &kmgr, int num_threads) : kmgr_(kmgr) {
    kmgr_.Resize(num_threads, 0);
  }

  static_assert(spatial_ndim == 2 || spatial_ndim == 3, "Only 2D and 3D resizing is supported");

  using Kernel = kernels::ResampleCPU<Out, In, spatial_ndim>;

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
    OnNumFramesUpdated();

    SetupKernel();
  }

  void SetupKernel() {
    const int dim = in_shape_.sample_dim();
    kernels::KernelContext ctx;

    for (int i = 0; i < GetNumFrames(); i++) {
      kernels::InTensorCPU<In, frame_ndim> dummy_input;
      dummy_input.shape = in_shape_[i];
      kernels::KernelRequirements &req = kmgr_.Setup<Kernel>(i, ctx, dummy_input, params_[i]);
      assert(req.output_shapes[0][0] == out_shape_[i]);
    }
  }

  void RunResize(HostWorkspace &ws,
                 TensorVector<CPUBackend> &output,
                 const TensorVector<CPUBackend> &input) override {
    auto in_view = view<const In>(input);
    auto in_frames_view = reshape(in_view, in_shape_, true);
    auto out_view = view<Out>(output);
    auto out_frames_view = reshape(out_view, out_shape_, true);

    ThreadPool &tp = ws.GetThreadPool();

    for (int i = 0; i < GetNumFrames(); i++) {
      auto work = [&, i](int tid) {
        kernels::KernelContext ctx;
        auto out_frame = out_frames_view[i];
        auto in_frame = in_frames_view[i];
        kmgr_.Run<Kernel>(tid, i, ctx, out_frame, in_frame, params_[i]);
      };

      double out_size = volume(out_frames_view.shape.tensor_shape_span(i));
      double in_size = volume(in_frames_view.shape.tensor_shape_span(i));
      double cost = 0;
      double root = 1.0 / spatial_ndim;
      for (int i = 0; i < spatial_ndim; i++) {
        // Approximation for isotropic scaling - each resize stage takes time
        // proportional to the output size of the stage, and the scaling volume ratio
        // is divided equally (geometrically) among stages. Hence, the weighted
        // geometric mean of rank spatial_ndim_.
        //
        // NOTE: This does not account for cost of antialiasing!
        cost += std::pow(std::pow(out_size, spatial_ndim - i) * pow(in_size, i), root);
      }
      tp.AddWork(work, std::llround(cost));
    }
    tp.RunAll();
  }

  void OnNumFramesUpdated() {
    int N = GetNumFrames();
    if (static_cast<int>(kmgr_.NumInstances()) < N)
      kmgr_.Resize<Kernel>(kmgr_.NumThreads(), N);
  }

  int GetNumFrames() const {
    return in_shape_.num_samples();
  }

  kernels::KernelManager &kmgr_;

  TensorListShape<frame_ndim> in_shape_, out_shape_;
  std::vector<ResamplingParamsND<spatial_ndim>> params_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_OP_IMPL_CPU_H_
