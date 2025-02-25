// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/imgproc/convolution/separable_convolution_cpu.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/dynamic_scratchpad.h"

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
  static const int num_samples_ = 4;


  TensorShape<> GetScaledShape(float scale) {
    TensorShape<> target_shape = {static_cast<int64_t>(32 * scale),
                                  static_cast<int64_t>(14 * scale),
                                  static_cast<int64_t>(22 * scale)};
    target_shape = target_shape.last(kAxes);
    if (kChannels)
      target_shape = shape_cat(target_shape, 3);
    if (kFrames)
      target_shape = shape_cat(6, target_shape);
    return target_shape;
  }

  void SetDataShape(float scale_size) {
    data_shape_.resize(num_samples_);
    for (int i = 0; i < num_samples_; i++) {
      data_shape_.set_tensor_shape(i, GetScaledShape(scale_size * (i + 1)));
    }
  }

  void ReshapeData(float scale_size = 1.0f) {
    SetDataShape(scale_size);

    for (int i = 0; i < kAxes; i++) {
      window_dims_[i] = uniform_list_shape<1>(num_samples_, {win_sizes[i]});
      kernel_window_[i].reshape(window_dims_[i]);
    }

    input_.reshape(data_shape_);
    output_.reshape(data_shape_);
    baseline_output_.reshape(data_shape_);
  }

  void FillData() {
    std::mt19937 rng;
    UniformRandomFill(input_.cpu(), rng, 0, 5);
    for (int i = 0; i < kAxes; i++) {
      for (int sample = 0; sample < kernel_window_[i].cpu().num_samples(); sample++) {
        testing::InitTriangleWindow(kernel_window_[i].cpu()[sample]);
      }
    }

    ConstantFill(baseline_output_.cpu(), 0);
  }

 public:
  void RunTest() {
    SeparableConvolutionGpu<float, float, float, kAxes, kChannels, kFrames> kernel_gpu;

    for (float scale : {0.75f, 1.0f, 1.25f, 0.5f}) {
      ReshapeData(scale);
      FillData();

      auto baseline_out_v = baseline_output_.cpu();
      auto in_cpu_v = input_.cpu();
      auto in_gpu_v = input_.gpu();
      auto out_gpu_v = output_.gpu();
      std::array<TensorListView<StorageCPU, const float, 1>, kAxes> window_v;

      for (int i = 0; i < kAxes; i++) {
        window_v[i] = kernel_window_[i].cpu();
      }

      KernelContext ctx_gpu, ctx_cpu;
      ctx_gpu.gpu.stream = 0;

      auto req_gpu = kernel_gpu.Setup(ctx_gpu, data_shape_, window_dims_);

      DynamicScratchpad dyn_scratchpad_gpu(AccessOrder(ctx_gpu.gpu.stream));
      ctx_gpu.scratchpad = &dyn_scratchpad_gpu;

      kernel_gpu.Run(ctx_gpu, out_gpu_v, in_gpu_v, window_v);
      int nsamples = in_gpu_v.num_samples();

      for (int i = 0; i < nsamples; i++) {
        SeparableConvolutionCpu<float, float, float, kAxes, kChannels> kernel_cpu;
        std::array<int, kAxes> window_dims;
        for (int axis = 0; axis < kAxes; axis++) {
          window_dims[axis] = window_dims_[axis][i][0];
        }

        int seq_elements = 1;
        int64_t stride = 0;
        auto sample_shape = data_shape_[i];
        auto element_shape = sample_shape.template last<kNdim - kFrames>();
        if (kFrames) {
          seq_elements = volume(sample_shape.begin(), sample_shape.begin() + 1);
          stride = volume(element_shape);
        }
        for (int frame = 0; frame < seq_elements; frame++) {
          auto req_cpu = kernel_cpu.Setup(ctx_cpu, element_shape, window_dims);

          DynamicScratchpad dyn_scratchpad_cpu(AccessOrder::host());
          ctx_cpu.scratchpad = &dyn_scratchpad_cpu;

          std::array<TensorView<StorageCPU, const float, 1>, kAxes> windows;
          for (int axis = 0; axis < kAxes; axis++) {
            windows[axis] = window_v[axis][i];
          }
          auto in_view = TensorView<StorageCPU, const float, kNdim - kFrames>{
              in_cpu_v[i].data + stride * frame, element_shape};
          auto out_view = TensorView<StorageCPU, float, kNdim - kFrames>{
              baseline_out_v[i].data + stride * frame, element_shape};
          kernel_cpu.Run(ctx_cpu, out_view, in_view, windows);
        }
      }


      auto out_cpu_v = output_.cpu(0);
      CUDA_CALL(cudaDeviceSynchronize());
      CUDA_CALL(cudaGetLastError());
      double eps = 0.001;
      Check(out_cpu_v, baseline_out_v, EqualEps(eps));
    }
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
