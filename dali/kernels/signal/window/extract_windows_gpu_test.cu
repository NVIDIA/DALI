// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <vector>
#include "dali/core/util.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/signal/window/extract_windows_gpu.cuh"
#include "dali/kernels/signal/window/window_functions.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {
namespace signal {

TEST(ExtractWindowsGPU, NonBatchedKernel) {
  float *in_gpu, *out_gpu;
  int winlen = 60;
  int outwinlen = 63;
  int windows = 80;
  int stride = windows;
  int step = 10;
  int length = windows * step - 100;
  int center = 5;
  CUDA_CALL(cudaMalloc(&in_gpu, sizeof(float)*length));
  CUDA_CALL(cudaMalloc(&out_gpu, sizeof(float)*windows*outwinlen));
  std::vector<float> in(length), out(windows*outwinlen);

  for (int i = 0; i < length; i++) {
    in[i] = i + 1000;
  }

  for (bool reflect : {true, false}) {
    CUDA_CALL(cudaMemcpy(in_gpu, in.data(), sizeof(float)*length, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(out_gpu, 0xff, sizeof(float)*windows*outwinlen));
    int xblocks = div_ceil(length, 32);
    int yblocks = div_ceil(winlen, 32);
    int win_data_start = (outwinlen - winlen) / 2;

    window::ExtractVerticalWindowsKernel<<<dim3(xblocks, yblocks), dim3(32, 32)>>>(
        out_gpu, windows, stride, in_gpu, length, nullptr, outwinlen, win_data_start, winlen,
        center, step, reflect);
    CUDA_CALL(cudaMemcpy(out.data(), out_gpu, sizeof(float) * outwinlen * windows,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for (int w = 0; w < windows; w++) {
      int i = 0;
      for (; i < win_data_start; i++) {
        EXPECT_EQ(out[w + i*stride], 0)
          << "padding @ window = " << w << ", index = " << i;
      }
      for (int j = 0; j < winlen; j++, i++) {
        int idx = w * step + j - center;
        if (reflect)
          idx = boundary::idx_reflect_101(idx, 0, length);

        float ref = idx >= 0 && idx < length ? in[idx] : 0;
        EXPECT_EQ(out[w + i*stride], ref)
          << "@ window = " << w << ", index = " << i;
      }
      for (; i < outwinlen; i++) {
        EXPECT_EQ(out[w + i*stride], 0)
          << "padding @ window = " << w << ", index = " << i;
      }
    }

    if (HasFailure()) {
      std::cout << "Debug: Extract window actual output:\n";
      for (int i = 0; i < outwinlen; i++) {
        for (int j = 0; j < windows; j++) {
          std::cout << out[i*stride+j] << " ";
        }
        std::cout << "\n";
      }
      std::cout << std::flush;
    }
  }

  cudaFree(in_gpu);
  cudaFree(out_gpu);
}


void TestBatchedExtract(
    ExtractWindowsImplGPU<float, float> *extract,
    const TensorListShape<1> &lengths,
    bool concatenate,
    Padding padding,
    span<const float> window,
    int out_win_len = -1) {
  bool vertical = extract->IsVertical();

  int N = lengths.num_samples();


  TestTensorList<float, 1> in_list;
  in_list.reshape(lengths);
  auto in_cpu = in_list.cpu();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < lengths[i][0]; j++)
      in_cpu[i].data[j] = 1000*(i+1)+j;
  }

  ExtractWindowsArgs args;
  args.window_length = window.empty() ? 55 : window.size();
  args.window_center = window.empty() ? 21 : window.size()/2;
  args.window_step = 2;
  args.padding = padding;

  int out_win_len_actual = out_win_len < 0 ? args.window_length : out_win_len;
  int in_win_start = (out_win_len_actual - args.window_length) / 2;

  KernelContext ctx;
  ctx.gpu.stream = 0;

  auto in_gpu = in_list.gpu(0);

  auto req = extract->Setup(ctx, make_span(lengths.shapes), args, concatenate, out_win_len);
  ASSERT_EQ(req.output_shapes.size(), 1u);
  ASSERT_EQ(req.output_shapes[0].num_samples(), concatenate ? 1 : N);

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;


  TestTensorList<float, 2> out;
  mm::uptr<float> gpu_win;
  if (!window.empty()) {
    gpu_win = mm::alloc_raw_unique<float, mm::memory_kind::device>(window.size());
    CUDA_CALL(
      cudaMemcpy(gpu_win.get(), window.data(), sizeof(float)*window.size(),
                 cudaMemcpyHostToDevice));
  }
  auto window_gpu = make_tensor_gpu<1>(gpu_win.get(), { window.size() });
  out.reshape(req.output_shapes[0].to_static<2>());
  auto out_gpu = out.gpu(0);
  CUDA_CALL(cudaMemset(out_gpu.data[0], 0xff, sizeof(float)*out_gpu.shape.num_elements()));
  extract->Run(ctx, out_gpu, in_gpu, window_gpu);
  auto out_cpu = out.cpu();

  ptrdiff_t ofs = 0;

  for (int sample = 0; sample < N; sample++) {
    ptrdiff_t length = lengths[sample][0];
    int nwnd = args.num_windows(length);

    int out_sample = 0;
    if (!concatenate) {
      ofs = 0;
      out_sample = sample;
    }
    ptrdiff_t sample_stride = vertical ? out_cpu.shape[out_sample][1] : 1;
    ptrdiff_t window_stride = vertical ? 1 : out_cpu.shape[out_sample][1];

    for (int w = 0; w < nwnd; w++, ofs += window_stride) {
      int i = 0;
      for (; i < in_win_start; i++) {
        ASSERT_EQ(out_cpu.data[out_sample][ofs + i * sample_stride], 0)
            << "padding @ sample = " << sample << ", window = " << w << ", index = " << i;
      }
      for (int j = 0 ; j < args.window_length; j++, i++) {
        ptrdiff_t idx = w * args.window_step + j - args.window_center;
        if (args.padding == Padding::Reflect) {
          idx = boundary::idx_reflect_101(idx, length);
        }
        float ref = idx >= 0 && idx < length ? in_cpu.data[sample][idx] : 0.0f;
        if (!window.empty())
          ref *= window[j];
        ASSERT_EQ(out_cpu.data[out_sample][ofs + i * sample_stride], ref)
            << "@ sample = " << sample << ", window = " << w << ", index = " << i;
      }
      for (; i < out_win_len_actual; i++) {
        ASSERT_EQ(out_cpu.data[out_sample][ofs + i * sample_stride], 0)
            << "padding @ sample = " << sample << ", window = " << w << ", index = " << i;
      }
    }
  }
}

void TestBatchedExtract(
    const TensorListShape<1> &lengths,
    bool concatenate,
    Padding padding,
    bool vertical,
    span<const float> window,
    int out_win_len = -1) {
  std::unique_ptr<ExtractWindowsImplGPU<float, float>> extract;
  if (vertical)
    extract = std::make_unique<ExtractVerticalWindowsImplGPU<float, float>>();
  else
    extract = std::make_unique<ExtractHorizontalWindowsImplGPU<float, float>>();

  TestBatchedExtract(extract.get(), lengths, concatenate, padding, window, out_win_len);
}

void TestBatchedExtract(
    bool concatenate,
    Padding padding,
    bool vertical,
    span<const float> window,
    int out_win_len = -1) {
  std::unique_ptr<ExtractWindowsImplGPU<float, float>> extract;
  if (vertical)
    extract = std::make_unique<ExtractVerticalWindowsImplGPU<float, float>>();
  else
    extract = std::make_unique<ExtractHorizontalWindowsImplGPU<float, float>>();

  TensorListShape<1> lengths = {{ 5, 305, 157 }};
  TestBatchedExtract(extract.get(), lengths, concatenate, padding, window, out_win_len);

  if (vertical)
    extract = std::make_unique<ExtractVerticalWindowsImplGPU<float, float>>();
  else
    extract = std::make_unique<ExtractHorizontalWindowsImplGPU<float, float>>();
  lengths = {{ 137, 203, 150, 12 }};
  TestBatchedExtract(extract.get(), lengths, concatenate, padding, window, out_win_len);
}

TEST(ExtractVerticalWindowsGPU, BatchedConcat) {
  TestBatchedExtract(true, Padding::Reflect, true, {});
}

TEST(ExtractVerticalWindowsGPU, BatchedSeparate) {
  TestBatchedExtract(false, Padding::Zero, true, {});
}

TEST(ExtractVerticalWindowsGPU, BatchedConcatWindowFunc) {
  vector<float> window(60);
  HannWindow(make_span(window));
  TestBatchedExtract(true, Padding::Zero, true, make_cspan(window));
}

TEST(ExtractVerticalWindowsGPU, BatchedSeparateWindowFunc) {
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(false, Padding::Reflect, true, make_cspan(window));
}

TEST(ExtractVerticalWindowsGPU, BatchedSeparateWindowFuncPad) {
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(true, Padding::Reflect, true, make_cspan(window), 72);
}

TEST(ExtractHorizontalWindowsGPU, BatchedConcat) {
  TestBatchedExtract(true, Padding::Reflect, false, {});
}

TEST(ExtractHorizontalWindowsGPU, BatchedSeparate) {
  TestBatchedExtract(false, Padding::Zero, false, {});
}

TEST(ExtractHorizontalWindowsGPU, BatchedConcatWindowFunc) {
  vector<float> window(60);
  HannWindow(make_span(window));
  TestBatchedExtract(true, Padding::Zero, false, make_cspan(window));
}

TEST(ExtractHorizontalWindowsGPU, BatchedSeparateWindowFunc) {
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(false, Padding::Reflect, false, make_cspan(window));
}

TEST(ExtractHorizontalWindowsGPU, BatchedSeparateWindowFuncPad) {
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(false, Padding::Reflect, false, make_cspan(window), 72);
}

TEST(ExtractHorizontalWindowsGPU, BatchedConcatWindowFuncPad) {
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(false, Padding::Reflect, true, make_cspan(window), 72);
}

TEST(ExtractHorizontalWindowsGPU, SizeSweep) {
  int max_size = 2048;
  std::vector<TensorShape<1>> lengths;
  int step = 1;
  for (int s = 1; s <= max_size; s+=step) {
    if ((s&255) == 0) {
      if (step > 1)  // add 2^n-1
        lengths.push_back({s-1});
      step += step;
    }
    lengths.push_back({s});
  }
  TensorListShape<1> shape(lengths);
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(shape, false, Padding::Reflect, false, make_cspan(window));
}

}  // namespace signal
}  // namespace kernels
}  // namespace dali
