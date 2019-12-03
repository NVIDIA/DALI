// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include "dali/core/util.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/signal/window/extract_windows_gpu.cuh"
#include "dali/kernels/scratch.h"
#include "dali/kernels/signal/window/window_functions.h"

namespace dali {
namespace kernels {
namespace signal {

TEST(ExtractWindowsGPU, NonBatchedKernel) {
  float *in_gpu, *out_gpu;
  int winlen = 60;
  int windows = 80;
  int stride = windows;
  int step = 10;
  int length = windows * step - 100;;
  int center = 5;
  bool reflect = false;
  cudaMalloc(&in_gpu, sizeof(float)*length);
  cudaMalloc(&out_gpu, sizeof(float)*windows*winlen);
  std::vector<float> in(length), out(windows*winlen);

  for (int i = 0; i < length; i++) {
    in[i] = i + 1000;
  }

  cudaMemcpy(in_gpu, in.data(), sizeof(float)*length, cudaMemcpyHostToDevice);
  cudaMemset(out_gpu, 0, sizeof(float)*windows*winlen);
  int xblocks = div_ceil(length, 32);
  int yblocks = div_ceil(winlen, 32);
  window::ExtractWindowsKernel<<<dim3(xblocks, yblocks), dim3(32, 32)>>>(
    out_gpu, windows, stride, in_gpu, length, nullptr, winlen, center, step, reflect);
  cudaMemcpy(out.data(), out_gpu, sizeof(float)*winlen*windows, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int w = 0; w < windows; w++) {
    for (int i = 0; i < winlen; i++) {
      int idx = w * step + i - center;
      if (reflect) {
        for (;;) {
          if (idx < 0)
            idx = -idx;
          else if (idx >= length)
            idx = 2*(length-1) - idx;
          else
            break;
        }
      }
      float ref = idx >= 0 && idx < length ? in[idx] : 0;
      EXPECT_EQ(out[w + i*stride], ref)
        << "@ window = " << w << ", index = " << i;
    }
  }

  if (HasFailure()) {
    std::cout << "Debug: Extract window actual output:\n";
    for (int i = 0; i < winlen; i++) {
      for (int j = 0; j < windows; j++) {
        std::cout << out[i*stride+j] << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::flush;
  }

  cudaFree(in_gpu);
  cudaFree(out_gpu);
}



void TestBatchedExtract(bool concatenate, Padding padding, span<const float> window) {
  ExtractWindowsGPUImpl<float, float> extract;

  TensorListShape<1> lengths({ TensorShape<1>{5}, TensorShape<1>{305}, TensorShape<1>{157} });
  int N = lengths.num_samples();

  ptrdiff_t total_length = 0;
  for (int i = 0; i < N; i++) {
    total_length += lengths[i][0];
  }

  TestTensorList<float, 1> in_list;
  in_list.reshape(lengths);
  auto in_cpu = in_list.cpu();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < lengths[j][0]; j++)
      in_cpu[i].data[j] = 1000*(i+1)+j;
  }

  ExtractWindowsArgs args;
  args.window_length = window.empty() ? 55 : window.size();
  args.window_center = window.empty() ? 21 : window.size()/2;
  args.window_step = 2;
  args.padding = padding;

  KernelContext ctx;
  ScratchpadAllocator sa;

  auto in_gpu = in_list.gpu(0);

  auto req = extract.Setup(ctx, in_gpu, args, concatenate);
  ASSERT_EQ(req.output_shapes.size(), 1u);
  ASSERT_EQ(req.output_shapes[0].num_samples(), concatenate ? 1 : N);

  sa.Reserve(req.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  TestTensorList<float, 2> out;
  memory::KernelUniquePtr<float> gpu_win;
  if (!window.empty()) {
    gpu_win = memory::alloc_unique<float>(AllocType::GPU, window.size());
    cudaMemcpy(gpu_win.get(), window.data(), sizeof(float)*window.size(), cudaMemcpyHostToDevice);
  }
  auto window_gpu = make_tensor_gpu<1>(gpu_win.get(), { window.size() });
  out.reshape(req.output_shapes[0].to_static<2>());
  extract.Run(ctx, out.gpu(0), in_gpu, window_gpu);
  auto out_cpu = out.cpu();

  int ofs = 0;

  for (int sample = 0; sample < N; sample++) {
    ptrdiff_t length = lengths[sample][0];
    int nwnd = args.num_windows(length);

    int out_sample = 0;
    if (!concatenate) {
      ofs = 0;
      out_sample = sample;
    }
    ptrdiff_t stride = out_cpu.shape[out_sample][1];

    for (int w = 0; w < nwnd; w++, ofs++) {
        for (int i = 0; i < args.window_length; i++) {
        ptrdiff_t idx = w * args.window_step + i - args.window_center;
        if (args.padding == Padding::Reflect) {
          for (;;) {
            if (idx < 0)
              idx = -idx;
            else if (idx >= length)
              idx = 2*(length-1) - idx;
            else
              break;
          }
        }
        float ref = idx >= 0 && idx < length ? in_cpu.data[sample][idx] : 0;
        if (!window.empty())
          ref *= window[i];
        ASSERT_EQ(out_cpu.data[out_sample][ofs + i*stride], ref)
          << "@ sample = " << sample
          << ", window = " << w << ", index = " << i;
      }
    }
  }
}

TEST(ExtractWindowsGPU, BatchedConcat) {
  TestBatchedExtract(true, Padding::Reflect, {});
}

TEST(ExtractWindowsGPU, BatchedSeparate) {
  TestBatchedExtract(false, Padding::Zero, {});
}

TEST(ExtractWindowsGPU, BatchedConcatWindowFunc) {
  vector<float> window(60);
  HannWindow(make_span(window));
  TestBatchedExtract(true, Padding::Zero, make_cspan(window));
}

TEST(ExtractWindowsGPU, BatchedSeparateWindowFunc) {
  vector<float> window(60);
  HammingWindow(make_span(window));
  TestBatchedExtract(false, Padding::Reflect, make_cspan(window));
}


}  // namespace signal
}  // namespace kernels
}  // namespace dali
