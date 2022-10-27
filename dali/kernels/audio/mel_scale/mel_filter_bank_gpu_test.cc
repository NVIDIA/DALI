// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/audio/mel_scale/mel_filter_bank_gpu.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_test.h"
#include <gtest/gtest.h>
#include <cassert>
#include <random>
#include <tuple>
#include <vector>
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_event.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {
namespace kernels {
namespace audio {
namespace test {


using TestBase = ::testing::TestWithParam<
    std::tuple<std::vector<TensorShape<>>,   // data shape
               int,                          // nfilter
               float,                        // sample_rate
               float,                        // fmin
               float,                        // fmax
               int64_t>>;                    // axis

class MelScaleGpuTest : public TestBase {
 public:
  MelScaleGpuTest()
      : data_shape_(std::get<0>(GetParam()))
      , nfilter_(std::get<1>(GetParam()))
      , sample_rate_(std::get<2>(GetParam()))
      , freq_low_(std::get<3>(GetParam()))
      , freq_high_(std::get<4>(GetParam()))
      , axis_(std::get<5>(GetParam())) {
    in_.reshape(data_shape_);
  }

 protected:
  void SetUp() final {
    std::mt19937 rng;
    UniformRandomFill(in_.cpu(), rng, 0.0, 1.0);
  }
  TensorListShape<> data_shape_;
  int nfilter_ = 4;
  float sample_rate_ = 16000, freq_low_ = 0, freq_high_ = 8000;
  TestTensorList<float> in_;
  int64_t axis_;
};

TEST_P(MelScaleGpuTest, MelScaleGpuTest) {
  using T = float;
  HtkMelScale<float> mel_scale;
  constexpr int ndim = 4;
  auto shape = data_shape_;
  auto batch_size = data_shape_.num_samples();
  int nfft = (shape[0][axis_] - 1) * 2;
  std::vector<int> nwin(batch_size, 1);
  if (axis_ < ndim - 1) {
    for (int i = 0; i < batch_size; ++i) {
        auto sh = shape[i];
        nwin[i] = volume(sh.begin() + axis_ + 1, sh.end());
    }
  }
  std::vector<int> nframes(batch_size, 1);
  if (axis_ > 0) {
    for (int i = 0; i < batch_size; ++i) {
      auto sh = shape[i];
      nframes[i] = volume(sh.begin(), sh.begin() + axis_);
    }
  }

  TensorListShape<> out_shape = data_shape_;
  for (int i = 0; i < batch_size; ++i) {
    out_shape.tensor_shape_span(i)[axis_] = nfilter_;
  }
  std::vector<T> out_sizes;
  for (int i = 0; i < batch_size; ++i) {
    out_sizes.push_back(volume(out_shape[i]));
  }

  T mel_low = mel_scale.hz_to_mel(freq_low_);
  T mel_high = mel_scale.hz_to_mel(freq_high_);

  T mel_delta = (mel_high - mel_low) / (nfilter_ + 1);
  T mel = mel_low;
  LOG_LINE << "Mel frequency grid (Hz):";
  for (int i = 0; i < nfilter_+1; i++, mel += mel_delta) {
    LOG_LINE << " " << mel_scale.mel_to_hz(mel);
  }
  LOG_LINE << " " << mel_scale.mel_to_hz(mel_high) << "\n";

  LOG_LINE << "FFT bin frequencies (Hz):";
  for (int k = 0; k < nfft / 2 + 1; k++) {
    LOG_LINE << " " << (k * sample_rate_ / nfft);
  }
  LOG_LINE << "\n";

  auto fbanks = ReferenceFilterBanks(nfilter_, nfft, sample_rate_, freq_low_, freq_high_);
  std::vector<std::vector<float>> expected_out;
  for (int i = 0; i < batch_size; ++i) {
    expected_out.emplace_back(out_sizes[i], 0.f);
  }

  for (int b = 0; b < batch_size; ++b) {
    auto in_view = in_.cpu()[b];
    for (int s = 0; s < nframes[b]; ++s) {
      for (int j = 0; j < nfilter_; j++) {
        for (int t = 0; t < nwin[b]; t++) {
          auto &out_val = expected_out[b][(s * nfilter_ + j) * nwin[b] + t];
          for (int i = 0; i < nfft/2+1; i++) {
            out_val += fbanks[j][i] * in_view.data[(s * (nfft/2+1) + i) * nwin[b] + t];
          }
        }
      }
    }
  }

  KernelContext ctx;
  ctx.gpu.stream = 0;
  kernels::audio::MelFilterBankArgs args;
  kernels::KernelManager kmgr;
  args.axis = axis_;
  args.nfft = nfft;
  args.nfilter = nfilter_;
  args.sample_rate = sample_rate_;
  args.freq_low = freq_low_;
  args.freq_high = freq_high_;
  args.mel_formula = MelScaleFormula::HTK;
  args.normalize = false;

  using Kernel = kernels::audio::MelFilterBankGpu<T>;
  kmgr.Resize<Kernel>(1);
  auto in_view = in_.gpu();
  auto req = kmgr.Setup<Kernel>(0, ctx, in_view, args);
  ASSERT_EQ(out_shape, req.output_shapes[0]);
  TestTensorList<float> out;
  out.reshape(out_shape);

  auto out_view = out.gpu();
  kmgr.Run<Kernel>(0, ctx, out_view, in_view);
  auto out_view_cpu = out.cpu();
  CUDA_CALL(cudaStreamSynchronize(0));
  for (int b = 0; b < batch_size; ++b) {
    for (int idx = 0; idx < out_sizes[b]; idx++) {
      ASSERT_NEAR(expected_out[b][idx], out_view_cpu.tensor_data(b)[idx], 1e-5) <<
        "Output data doesn't match in sample " << b << " reference (idx=" << idx << ")";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MelScaleGpuTest, MelScaleGpuTest, testing::Combine(
    testing::Values(std::vector<TensorShape<>>{TensorShape<>{10, 4, 6, 12}},
                    std::vector<TensorShape<>>{TensorShape<>{4, 5, 6, 5},
                                               TensorShape<>{4, 8, 6, 5}}),  // shape
    testing::Values(4, 8),  // nfilter
    testing::Values(16000.0f),  // sample rate
    testing::Values(0.0f, 1000.0f),  // fmin
    testing::Values(5000.0f, 8000.0f),  // fmax
    testing::Values(0, 2, 3)));  // axis

using BenchBase = ::testing::TestWithParam<std::tuple<
    int,                  // axis
    int,                  // channels
    std::tuple<int, int, int>,  // batch_size, min_length, max_length
    int,                  // nfft
    int,                  // nfilter
    float,                // sample_rate
    float,                // fmin
    float>>;              // fmax

class MelScaleGpuBench : public BenchBase {
 public:
  MelScaleGpuBench() {
    std::tuple<int, int, int> data_sizes;
    std::tie(
      axis_,
      channels_,
      data_sizes,
      nfft_,
      nfilter_,
      sample_rate_,
      freq_low_,
      freq_high_) = GetParam();
    std::tie(batch_size_, min_length_, max_length_) = data_sizes;
    assert(axis_ == 0 || axis_ == 1);
  }

 protected:
  void SetUp() final {
    std::mt19937 rng(12345);
    in_shape_.resize(batch_size_, channels_ > 1 ? 3 : 2);
    std::uniform_int_distribution<int> length_dist(min_length_, max_length_);
    int length_axis = 1 - axis_;
    for (int i = 0; i < batch_size_; i++) {
      auto sh = in_shape_.tensor_shape_span(i);
      sh[axis_] = nfft_;
      sh[length_axis] = length_dist(rng);
      if (channels_ > 1)
        sh[2] = channels_;
    }
    out_shape_ = in_shape_;
    for (int i = 0; i < batch_size_; ++i) {
      out_shape_.tensor_shape_span(i)[axis_] = nfilter_;
    }

    in_.reshape(in_shape_);
    auto in_view = in_.gpu();
    for (int i = 0; i < in_view.num_samples(); i++)
      CUDA_CALL(cudaMemset(in_view.data[i], rng(), in_view[i].num_elements() * sizeof(float)));
  }
  int axis_ = 0;
  int channels_ = 1;
  int batch_size_ = 16;
  int min_length_ = 1000;
  int max_length_ = 10000;
  int nfft_ = 257;
  int nfilter_ = 80;
  float sample_rate_ = 16000, freq_low_ = 0, freq_high_ = 8000;
  TensorListShape<> in_shape_, out_shape_;
  TestTensorList<float> in_;
};

TEST_P(MelScaleGpuBench, Benchmark) {
  auto stream = CUDAStreamPool::instance().Get();

  KernelContext ctx;
  ctx.gpu.stream = stream;
  kernels::audio::MelFilterBankArgs args;
  kernels::KernelManager kmgr;
  args.axis = axis_;
  args.nfft = nfft_;
  args.nfilter = nfilter_;
  args.sample_rate = sample_rate_;
  args.freq_low = freq_low_;
  args.freq_high = freq_high_;
  args.mel_formula = MelScaleFormula::Slaney;
  args.normalize = false;

  print(std::cout,
    "axis     ", axis_, "\n"
    "channels ", channels_, "\n"
    "NFFT     ", nfft_, "\n"
    "filters  ", nfilter_, "\n"
    "batch    ", batch_size_, "\n"
    "min_len  ", min_length_, "\n"
    "max_len  ", max_length_, "\n"
    "rate     ", sample_rate_, " Hz\n"
    "freq_lo  ", freq_low_, " Hz\n"
    "freq_hi  ", freq_high_, " Hz\n");
  std::flush(cout);

  using Kernel = kernels::audio::MelFilterBankGpu<float>;
  kmgr.Resize<Kernel>(1);
  auto in_view = in_.gpu();
  auto req = kmgr.Setup<Kernel>(0, ctx, in_view, args);
  ASSERT_EQ(out_shape_, req.output_shapes[0]);
  TestTensorList<float> out;
  out.reshape(out_shape_);

  auto out_view = out.gpu();
  // warm-up
  kmgr.Run<Kernel>(0, ctx, out_view, in_view);

  CUDAEvent started  = CUDAEvent::CreateWithFlags(0);  // timing enabled
  CUDAEvent finished = CUDAEvent::CreateWithFlags(0);

  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaEventRecord(started, ctx.gpu.stream));
  kmgr.Run<Kernel>(0, ctx, out_view, in_view);
  CUDA_CALL(cudaEventRecord(finished, ctx.gpu.stream));
  CUDA_CALL(cudaDeviceSynchronize());
  float time_ms = 0;
  CUDA_CALL(cudaEventElapsedTime(&time_ms, started, finished));

  int64_t data_size = (in_view.num_elements() + out_view.num_elements()) * sizeof(float);
  std::cout << data_size * 1e-6 / time_ms << " GB/s" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(MelScaleGpuBench, MelScaleGpuBench, testing::Combine(
    testing::Values(1, 0),        // axis
    testing::Values(1),           // channels
    /*testing::Values(std::make_tuple(1, 10000, 10000),  // batch size, min_length, max_length
                    std::make_tuple(1, 10000, 100000),
                    std::make_tuple(16, 1000, 10000),
                    std::make_tuple(64, 2000, 4000),
                    std::make_tuple(128, 1000, 3000),
                    std::make_tuple(512, 1000, 1500)),*/
    testing::Values(std::make_tuple(512, 1601, 1601)),  // batch size, min_length, max_length
    testing::Values(257, 513),    // nfft
    testing::Values(80),          // nfilter
    testing::Values(16000.0f),    // sample rate
    testing::Values(0.0f),        // fmin
    testing::Values(8000.0f)));   // fmax


}  // namespace test
}  // namespace audio
}  // namespace kernels
}  // namespace dali
