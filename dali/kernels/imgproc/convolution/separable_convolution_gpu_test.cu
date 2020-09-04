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



template <typename T>
void print_mat(int rows, int cols, const T *mat, int max_rows = -1, int max_cols = -1) {
  if (max_rows == -1)
    max_rows = rows;
  if (max_cols == -1)
    max_cols = cols;
  for (int r = 0; r < max_rows; r++) {
    std::cout << "{";
    for (int c = 0; c < max_cols; c++) {
      std::cout << std::setw(3) << (float)mat[r * cols + c] << ", ";
      if (c % 16 == 0 && c) {
        std::cout << std::endl;
      }
    }
    std::cout << "}\n";
  }
  std::cout << std::endl;
}



template <typename T>
void InitNoOpWindow(const TensorListView<StorageCPU, T, 1> &windows) {
  for (int sample = 0; sample < windows.num_samples(); sample++) {
    auto window = windows[sample];
    int radius = window.num_elements() / 2;
    // for (int i = 0; i < radius; i++) {
    //   *window(i) = i + 1;
    //   *window(window.num_elements() - i - 1) = i + 1;
    // }
    for (int i = 0; i < window.num_elements(); i++) {
      *window(i) = 0;
    }
    *window(radius) = 1;
  }
}

template <typename T>
void InitTriangleWindow(const TensorListView<StorageCPU, T, 1> &windows) {
  for (int sample = 0; sample < windows.num_samples(); sample++) {
    auto window = windows[sample];
    int radius = window.num_elements() / 2;
    for (int i = 0; i < radius; i++) {
      *window(i) = i + 1;
      *window(window.num_elements() - i - 1) = i + 1;
    }
    *window(radius) = radius + 1;
  }
}

// TEST(SeparableConvolutionTest, Axes1WithChannels) {
//   std::array<int, 1> window_dims = {5};
//   TestTensorList<float, 1> kernel_window;
//   TestTensorList<float, 2> input;
//   TestTensorList<int, 2> output, baseline_output;

//   TensorListShape<2> data_shape = uniform_list_shape<2>(1, {16, 3});

//   kernel_window.reshape(uniform_list_shape<1>(1, {window_dims[0]}));
//   input.reshape(data_shape);
//   output.reshape(data_shape);
//   baseline_output.reshape(data_shape);

//   auto kernel_window_v = kernel_window.cpu()[0];
//   auto in_v = input.cpu()[0];
//   auto out_v = output.cpu()[0];
//   auto baseline_out_v = baseline_output.cpu()[0];

//   std::mt19937 rng;
//   UniformRandomFill(in_v, rng, 0, 255);
//   InitTriangleWindow(kernel_window_v);

//   SeparableConvolutionCpu<int, float, float, 1, true> kernel;
//   KernelContext ctx;

//   auto req = kernel.Setup(ctx, data_shape[0], window_dims);

//   ScratchpadAllocator scratch_alloc;
//   scratch_alloc.Reserve(req.scratch_sizes);
//   auto scratchpad = scratch_alloc.GetScratchpad();
//   ctx.scratchpad = &scratchpad;

//   kernel.Run(ctx, out_v, in_v,
//              uniform_array<1, TensorView<StorageCPU, const float, 1>>(kernel_window_v));
//   testing::BaselineConvolve(baseline_out_v, in_v, kernel_window_v, 0, window_dims[0] / 2);
//   Check(out_v, baseline_out_v);
// }

// TEST(SeparableConvolutionTest, Axes1NoChannels) {
//   std::array<int, 1> window_dims = {5};
//   TestTensorList<float, 1> kernel_window;
//   TestTensorList<float, 2> input;
//   TestTensorList<int, 1> output;
//   TestTensorList<int, 2> baseline_output;

//   TensorListShape<2> data_shape = uniform_list_shape<2>(1, {16, 1});

//   kernel_window.reshape(uniform_list_shape<1>(1, {window_dims[0]}));
//   input.reshape(data_shape);
//   output.reshape(data_shape.first<1>());
//   baseline_output.reshape(data_shape);

//   auto kernel_window_v = kernel_window.cpu()[0];
//   auto baseline_in_v = input.cpu()[0];
//   TensorView<StorageCPU, float, 1> in_v = {baseline_in_v.data, baseline_in_v.shape.first<1>()};
//   auto out_v = output.cpu()[0];
//   auto baseline_out_v = baseline_output.cpu()[0];

//   std::mt19937 rng;
//   UniformRandomFill(in_v, rng, 0, 255);
//   InitTriangleWindow(kernel_window_v);

//   SeparableConvolutionCpu<int, float, float, 1, false> kernel;
//   KernelContext ctx;

//   auto req = kernel.Setup(ctx, data_shape[0].first<1>(), window_dims);

//   ScratchpadAllocator scratch_alloc;
//   scratch_alloc.Reserve(req.scratch_sizes);
//   auto scratchpad = scratch_alloc.GetScratchpad();
//   ctx.scratchpad = &scratchpad;

//   kernel.Run(ctx, out_v, in_v,
//              uniform_array<1, TensorView<StorageCPU, const float, 1>>(kernel_window_v));
//   testing::BaselineConvolve(baseline_out_v, baseline_in_v, kernel_window_v, 0, window_dims[0] / 2);
//   TensorView<StorageCPU, int, 1> compare_v = {baseline_out_v.data, baseline_out_v.shape.first<1>()};
//   Check(out_v, compare_v);
// }

// TEST(SeparableConvolutionTest, Axes2WithChannels) {
//   std::array<int, 2> window_dims = {5, 7};
//   TestTensorList<float, 1> kernel_window_0, kernel_window_1;
//   TestTensorList<int, 3> input;
//   TestTensorList<float, 3> intermediate;
//   TestTensorList<int, 3> output, baseline_output;

//   TensorListShape<3> data_shape = uniform_list_shape<3>(1, {20, 16, 3});

//   kernel_window_0.reshape(uniform_list_shape<1>(1, {window_dims[0]}));
//   kernel_window_1.reshape(uniform_list_shape<1>(1, {window_dims[1]}));
//   input.reshape(data_shape);
//   intermediate.reshape(data_shape);
//   output.reshape(data_shape);
//   baseline_output.reshape(data_shape);

//   auto kernel_window_0_v = kernel_window_0.cpu()[0];
//   auto kernel_window_1_v = kernel_window_1.cpu()[0];
//   auto in_v = input.cpu()[0];
//   auto interm_v = intermediate.cpu()[0];
//   auto out_v = output.cpu()[0];
//   auto baseline_out_v = baseline_output.cpu()[0];

//   std::mt19937 rng;
//   UniformRandomFill(in_v, rng, 0, 255);
//   InitTriangleWindow(kernel_window_0_v);
//   InitTriangleWindow(kernel_window_1_v);

//   SeparableConvolutionCpu<int, int, float, 2, true> kernel;
//   static_assert(
//       std::is_same<typename SeparableConvolutionCpu<int, int, float, 2, true>::Intermediate,
//                    float>::value,
//       "Unexpected intermediate type");
//   KernelContext ctx;

//   auto req = kernel.Setup(ctx, data_shape[0], window_dims);

//   ScratchpadAllocator scratch_alloc;
//   scratch_alloc.Reserve(req.scratch_sizes);
//   auto scratchpad = scratch_alloc.GetScratchpad();
//   ctx.scratchpad = &scratchpad;

//   kernel.Run(ctx, out_v, in_v, {kernel_window_0_v, kernel_window_1_v});
//   testing::BaselineConvolve(interm_v, in_v, kernel_window_1_v, 1, window_dims[1] / 2);
//   testing::BaselineConvolve(baseline_out_v, interm_v, kernel_window_0_v, 0, window_dims[0] / 2);
//   Check(out_v, baseline_out_v);
// }

TEST(SeparableConvolutionGpuTest, Axes2NoChannels) {
  TestTensorList<float, 1> kernel_window_0, kernel_window_1;
  TestTensorList<float, 2> input;
  TestTensorList<float, 2> intermediate;
  TestTensorList<float, 2> output;
  TestTensorList<float, 2> output_baseline;

  TensorListShape<2> data_shape = std::vector<TensorShape<2>>{TensorShape<2>{145, 128}};
  int num_samples = data_shape.size();

  std::array<TensorListShape<1>, 2> window_dims = {uniform_list_shape<1>(num_samples, {5}), uniform_list_shape<1>(num_samples, {7})};

  kernel_window_0.reshape(window_dims[0]);
  kernel_window_1.reshape(window_dims[1]);
  input.reshape(data_shape);
  // intermediate.reshape(data_shape);
  output.reshape(data_shape);
  output_baseline.reshape(data_shape);

  auto kernel_window_0_v = kernel_window_0.cpu();
  auto kernel_window_1_v = kernel_window_1.cpu();
  auto in_cpu_v = input.cpu();
  // auto interm_cpu_v = intermediate.cpu();
  auto out_gpu_v = output.gpu();
  auto out_cpu_baseline_v = output_baseline.cpu();

  std::mt19937 rng;
  // UniformRandomFill(in_cpu_v, rng, 0, 255);
  // SequentialFill(in_cpu_v);
  ConstantFill(in_cpu_v, 42.f);
  InitTriangleWindow(kernel_window_0_v);
  // InitNoOpWindow(kernel_window_0_v);
  InitTriangleWindow(kernel_window_1_v);

  auto in_gpu_v = input.gpu();

  SeparableConvolutionCpu<float, float, float, 2, false> kernel_cpu;
  SeparableConvolutionGpu<float, float, float, 2, false> kernel_gpu;
  // static_assert(
  //     std::is_same<typename SeparableConvolutionCpu<int, int, float, 2, false>::Intermediate,
  //                  float>::value,
  //     "Unexpected intermediate type");
  KernelContext ctx_cpu, ctx_gpu;

  ctx_gpu.gpu.stream = 0;

  for (int sample = 0; sample < num_samples; sample++) {
    std::array<int, 2> window_sample_dims;
    for (int axis = 0; axis < 2; axis++) {
      window_sample_dims[axis] = window_dims[axis][sample][0];
    }
    auto req = kernel_cpu.Setup(ctx_cpu, data_shape[sample], window_sample_dims);

    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx_cpu.scratchpad = &scratchpad;

    kernel_cpu.Run(ctx_cpu, out_cpu_baseline_v[sample], in_cpu_v[sample], {kernel_window_0_v[sample], kernel_window_1_v[sample]});
  }

  auto req = kernel_gpu.Setup(ctx_gpu, data_shape, window_dims);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx_gpu.scratchpad = &scratchpad;

  kernel_gpu.Run(ctx_gpu, out_gpu_v, in_gpu_v, {kernel_window_0_v, kernel_window_1_v});

  auto out_cpu_v = output.cpu(0);
  cudaDeviceSynchronize();
  CUDA_CALL(cudaGetLastError());

  double eps = 1e-2;
  Check(out_cpu_v, out_cpu_baseline_v, EqualEps(eps));
    printf("CUTLASS\n\n");
    print_mat(out_cpu_v[0].shape[0], out_cpu_v[0].shape[1], out_cpu_v[0].data);
    printf("CPU baseline\n\n");
    print_mat(out_cpu_baseline_v[0].shape[0], out_cpu_baseline_v[0].shape[1], out_cpu_baseline_v[0].data);
}

// TEST(SeparableConvolutionTest, Axes3WithChannels) {
//   std::array<int, 3> window_dims = {5, 7, 3};
//   TestTensorList<uint16_t, 1> kernel_window_0, kernel_window_1, kernel_window_2;
//   TestTensorList<int16_t, 4> input;
//   TestTensorList<int, 4> intermediate_0, intermediate_1;
//   TestTensorList<int16_t, 4> output, baseline_output;

//   TensorListShape<4> data_shape = uniform_list_shape<4>(1, {14, 20, 16, 3});

//   kernel_window_0.reshape(uniform_list_shape<1>(1, {window_dims[0]}));
//   kernel_window_1.reshape(uniform_list_shape<1>(1, {window_dims[1]}));
//   kernel_window_2.reshape(uniform_list_shape<1>(1, {window_dims[2]}));
//   input.reshape(data_shape);
//   intermediate_0.reshape(data_shape);
//   intermediate_1.reshape(data_shape);
//   output.reshape(data_shape);
//   baseline_output.reshape(data_shape);

//   auto kernel_window_0_v = kernel_window_0.cpu()[0];
//   auto kernel_window_1_v = kernel_window_1.cpu()[0];
//   auto kernel_window_2_v = kernel_window_2.cpu()[0];
//   auto in_v = input.cpu()[0];
//   auto interm_0_v = intermediate_0.cpu()[0];
//   auto interm_1_v = intermediate_1.cpu()[0];
//   auto out_v = output.cpu()[0];
//   auto baseline_out_v = baseline_output.cpu()[0];

//   std::mt19937 rng;
//   UniformRandomFill(in_v, rng, 0, 255);
//   InitTriangleWindow(kernel_window_0_v);
//   InitTriangleWindow(kernel_window_1_v);
//   InitTriangleWindow(kernel_window_2_v);

//   SeparableConvolutionCpu<int16_t, int16_t, uint16_t, 3, true> kernel;
//   static_assert(
//       std::is_same<
//           typename SeparableConvolutionCpu<int16_t, int16_t, uint16_t, 3, true>::Intermediate,
//           int>::value,
//       "Unexpected intermediate type");
//   KernelContext ctx;

//   auto req = kernel.Setup(ctx, data_shape[0], window_dims);

//   ScratchpadAllocator scratch_alloc;
//   scratch_alloc.Reserve(req.scratch_sizes);
//   auto scratchpad = scratch_alloc.GetScratchpad();
//   ctx.scratchpad = &scratchpad;

//   kernel.Run(ctx, out_v, in_v, {kernel_window_0_v, kernel_window_1_v, kernel_window_2_v});

//   testing::BaselineConvolve(interm_0_v, in_v, kernel_window_2_v, 2, window_dims[2] / 2);
//   testing::BaselineConvolve(interm_1_v, interm_0_v, kernel_window_1_v, 1, window_dims[1] / 2);
//   testing::BaselineConvolve(baseline_out_v, interm_1_v, kernel_window_0_v, 0, window_dims[0] / 2);
//   Check(out_v, baseline_out_v);
// }

// TEST(SeparableConvolutionTest, Axes3NoChannels) {
//   std::array<int, 3> window_dims = {5, 7, 3};
//   TestTensorList<float, 1> kernel_window_0, kernel_window_1, kernel_window_2;
//   TestTensorList<int, 4> input;
//   TestTensorList<float, 4> intermediate_0, intermediate_1;
//   TestTensorList<float, 3> output;
//   TestTensorList<float, 4> baseline_output;

//   TensorListShape<4> data_shape = uniform_list_shape<4>(1, {14, 20, 16, 1});

//   kernel_window_0.reshape(uniform_list_shape<1>(1, {window_dims[0]}));
//   kernel_window_1.reshape(uniform_list_shape<1>(1, {window_dims[1]}));
//   kernel_window_2.reshape(uniform_list_shape<1>(1, {window_dims[2]}));
//   input.reshape(data_shape);
//   intermediate_0.reshape(data_shape);
//   intermediate_1.reshape(data_shape);
//   output.reshape(data_shape.first<3>());
//   baseline_output.reshape(data_shape);

//   auto kernel_window_0_v = kernel_window_0.cpu()[0];
//   auto kernel_window_1_v = kernel_window_1.cpu()[0];
//   auto kernel_window_2_v = kernel_window_2.cpu()[0];
//   auto baseline_in_v = input.cpu()[0];
//   TensorView<StorageCPU, int, 3> in_v = {baseline_in_v.data, baseline_in_v.shape.first<3>()};
//   auto interm_0_v = intermediate_0.cpu()[0];
//   auto interm_1_v = intermediate_1.cpu()[0];
//   auto out_v = output.cpu()[0];
//   auto baseline_out_v = baseline_output.cpu()[0];

//   std::mt19937 rng;
//   UniformRandomFill(in_v, rng, 0, 255);
//   InitTriangleWindow(kernel_window_0_v);
//   InitTriangleWindow(kernel_window_1_v);
//   InitTriangleWindow(kernel_window_2_v);

//   SeparableConvolutionCpu<float, int, float, 3, false> kernel;
//   static_assert(
//       std::is_same<typename SeparableConvolutionCpu<float, int, float, 3, false>::Intermediate,
//                    float>::value,
//       "Unexpected intermediate type");
//   KernelContext ctx;

//   auto req = kernel.Setup(ctx, data_shape[0].first<3>(), window_dims);

//   ScratchpadAllocator scratch_alloc;
//   scratch_alloc.Reserve(req.scratch_sizes);
//   auto scratchpad = scratch_alloc.GetScratchpad();
//   ctx.scratchpad = &scratchpad;

//   kernel.Run(ctx, out_v, in_v, {kernel_window_0_v, kernel_window_1_v, kernel_window_2_v});

//   testing::BaselineConvolve(interm_0_v, baseline_in_v, kernel_window_2_v, 2, window_dims[2] / 2);
//   testing::BaselineConvolve(interm_1_v, interm_0_v, kernel_window_1_v, 1, window_dims[1] / 2);
//   testing::BaselineConvolve(baseline_out_v, interm_1_v, kernel_window_0_v, 0, window_dims[0] / 2);
//   TensorView<StorageCPU, float, 3> compare_v = {baseline_out_v.data,
//                                                 baseline_out_v.shape.first<3>()};
//   Check(out_v, compare_v);
// }

}  // namespace kernels
}  // namespace dali
