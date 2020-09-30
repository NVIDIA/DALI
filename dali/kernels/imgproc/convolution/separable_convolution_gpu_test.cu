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

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/imgproc/convolution/baseline_convolution.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/scratch.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {

/**
 * @brief Simple sanity test for SeparableConvolutionGPU
 */
template <int Axes, bool Channels, bool Frames>
class SepearableConvolutionGpuTestImpl {
  constexpr static int kAxes = Axes;
  constexpr static int kChannels = Channels;
  constexpr static int kFrames = Frames;
  constexpr static int kNdim = kFrames + kAxes + kChannels;

  // Predefined window shape per axis
  int win_sizes[3] = {3, 5, 7};

  std::array<TestTensorList<float, 1>, kAxes> kernel_window_;
  std::array<TensorListShape<1>, kAxes> window_dims_;
  TestTensorList<float, kNdim> input_;
  TestTensorList<float, kNdim> output_;
  TestTensorList<float, kNdim> baseline_output_;
  TensorListShape<kNdim> data_shape_;
  int num_samples_;

  void SetDataShape() {
    TensorShape<> target_shape = {64, 64, 64};
    target_shape = target_shape.last(kAxes);
    if (kChannels)
      target_shape = shape_cat(target_shape, 3);
    if (kFrames)
      target_shape = shape_cat(20, target_shape);
    data_shape_ = uniform_list_shape<kNdim>(1, target_shape.to_static<kNdim>());
  }

  void ReshapeData() {
    SetDataShape();
    int num_samples_ = data_shape_.size();

    for (int i = 0; i < kAxes; i++) {
      window_dims_[i] = uniform_list_shape<1>(num_samples_, {win_sizes[i]});
      kernel_window_[i].reshape(window_dims_[i]);
    }

    input_.reshape(data_shape_);
    output_.reshape(data_shape_);
    baseline_output_.reshape(data_shape_);
  }

  void FillData() {
    ConstantFill(input_.cpu(), 1);
    for (int i = 0; i < kAxes; i++) {
      ConstantFill(kernel_window_[i].cpu(), 1);
    }

    // Calculate the baseline, windows are filled with ones.
    int result = 1;
    for (int i = kAxes - 1; i >= 0; i--) {
      result *= win_sizes[i];
    }
    ConstantFill(baseline_output_.cpu(), result);
  }

 public:
  void RunTest() {
    ReshapeData();
    FillData();

    auto baseline_out_v = baseline_output_.cpu();
    auto in_gpu_v = input_.gpu();
    auto out_gpu_v = output_.gpu();
    std::array<TensorListView<StorageCPU, const float, 1>, kAxes> window_v;

    for (int i = 0; i < kAxes; i++) {
      window_v[i] = kernel_window_[i].cpu();
    }

    SeparableConvolutionGpu<float, float, float, kAxes, kChannels, kFrames> kernel_gpu;
    KernelContext ctx_gpu;

    ctx_gpu.gpu.stream = 0;

    auto req = kernel_gpu.Setup(ctx_gpu, data_shape_, window_dims_);

    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx_gpu.scratchpad = &scratchpad;

    kernel_gpu.Run(ctx_gpu, out_gpu_v, in_gpu_v, window_v);

    auto out_cpu_v = output_.cpu(0);
    cudaDeviceSynchronize();
    CUDA_CALL(cudaGetLastError());
    Check(out_cpu_v, baseline_out_v);
  }
};

TEST(SeparableConvolutionGpuTest, Axes1NoChannels) {
  SepearableConvolutionGpuTestImpl<1, false, false> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, Axes1Channels) {
  SepearableConvolutionGpuTestImpl<1, true, false> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, FramesAxes1Channels) {
  SepearableConvolutionGpuTestImpl<1, true, true> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, Axes2NoChannels) {
  SepearableConvolutionGpuTestImpl<2, false, false> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, Axes2Channels) {
  SepearableConvolutionGpuTestImpl<2, true, false> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, FramesAxes2Channels) {
  SepearableConvolutionGpuTestImpl<2, true, true> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, Axes3NoChannels) {
  SepearableConvolutionGpuTestImpl<3, false, false> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, Axes3Channels) {
  SepearableConvolutionGpuTestImpl<3, true, false> impl;
  impl.RunTest();
}

TEST(SeparableConvolutionGpuTest, FramesAxes3Channels) {
  SepearableConvolutionGpuTestImpl<3, true, true> impl;
  impl.RunTest();
}

}  // namespace kernels
}  // namespace dali
