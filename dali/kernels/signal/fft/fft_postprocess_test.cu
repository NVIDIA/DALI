// Copyright (c) 2020, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>
#include "dali/kernels/common/copy.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/signal/fft/fft_postprocess.cuh"

namespace dali {


template <>
bool EqualEps::operator()<float2, float2>(const float2 &a, const float2 &b) const {
  return std::abs(b.x - a.x) <= eps && std::abs(b.y - a.y) <= eps;
}

template <typename StorageBackend, int ndim, typename RandomGenerator>
void UniformRandomFill(const TensorListView<StorageBackend, float2, ndim> &tlv,
                       RandomGenerator &rng, float lo, float hi) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorListView backend");
  auto dist = uniform_distribution(lo, hi);
  auto gen = [&]() {
    return float2{ dist(rng), dist(rng) };
  };
  Fill(tlv, gen);
}

namespace kernels {
namespace signal {
namespace fft_postprocess {

TEST(FFTPostprocess, Norm2) {
  norm2 f;
  EXPECT_EQ(f(float2{0, 0}), 0);
  EXPECT_NEAR(f(float2{1, 1}), sqrt(2), 1e-5f);
  EXPECT_NEAR(f(float2{-1, 1}), sqrt(2), 1e-5f);
  EXPECT_NEAR(f(float2{-3, -4}), 5, 1e-5f);
}

TEST(FFTPostprocess, Norm2Square) {
  norm2square f;
  EXPECT_EQ(f(float2{0, 0}), 0.0f);
  EXPECT_EQ(f(float2{1, 1}), 2.0f);
  EXPECT_EQ(f(float2{-1, 1}), 2.0f);
  EXPECT_EQ(f(float2{-3, -4}), 25.0f);
}

TEST(FFTPostprocess, Power_dB) {
  power_dB f;
  EXPECT_NEAR(f(float2{0, 0}), -80, 1e-5f);
  EXPECT_EQ(f(float2{1, 0}), 0);
  EXPECT_EQ(f(float2{0, -1}), 0);

  EXPECT_NEAR(f(float2{1, 1}), 3.01029995664, 1e-5f);
  EXPECT_NEAR(f(float2{2, 0}), 6.02059991328, 1e-5f);
  EXPECT_NEAR(f(float2{2, 2}), 9.03089986992, 1e-5f);
  EXPECT_NEAR(f(float2{1e-3, 0}), -60, 1e-5f);

  f = power_dB(-40);
  EXPECT_NEAR(f(float2{0, 0}), -40, 1e-5f);
  EXPECT_NEAR(f(float2{1e-1, 0}), -20, 1e-5f);
  EXPECT_NEAR(f(float2{1e-2, 0}), -40, 1e-5f);
}

template <typename Out, typename In, typename Convert = identity>
struct FFTPostprocessArgs {};

template <typename TestArgs>
class FFTPostprocessTest;

template <typename Out, typename In, typename Convert>
class FFTPostprocessTest<FFTPostprocessArgs<Out, In, Convert>> : public ::testing::Test {
 public:
  void ToFreqMajor() {
    std::mt19937_64 rng;
    TensorListShape<2> in_shape, out_shape;
    std::uniform_int_distribution<int> dist(1, 500);
    int N = 10;
    int fft = 200;  // deliberately not a multiple of 32
    in_shape.resize(N);
    out_shape.resize(N);
    for (int i = 0; i < N; i++) {
      int len = dist(rng);
      in_shape.set_tensor_shape(i, { len, fft });
      out_shape.set_tensor_shape(i, { fft, len });
    }
    TestTensorList<In, 2> in;
    TestTensorList<Out, 2> out, ref;
    in.reshape(in_shape);
    auto cpu_in = in.cpu();
    UniformRandomFill(cpu_in, rng, -1, 1);

    ToFreqMajorSpectrum<Out, In, Convert> tr;
    KernelContext ctx;
    ctx.gpu.stream = 0;
    KernelRequirements req = tr.Setup(ctx, in_shape);
    ASSERT_EQ(req.output_shapes.size(), 1u);
    ASSERT_EQ(req.output_shapes[0], out_shape);
    DynamicScratchpad scratchpad;
    ctx.scratchpad = &scratchpad;
    out.reshape(out_shape);
    tr.Run(ctx, out.gpu(), in.gpu());
    CUDA_CALL(cudaGetLastError());

    auto cpu_out = out.cpu();
    ref.reshape(out_shape);
    auto cpu_ref = ref.cpu();

    Convert convert;
    for (int i = 0; i < N; i++) {
      TensorView<StorageCPU, In, 2> in_tv = cpu_in[i];
      TensorView<StorageCPU, Out, 2> ref_tv = cpu_ref[i];
      for (int y = 0; y < in_tv.shape[0]; y++)
        for (int x = 0; x < in_tv.shape[1]; x++)
          *ref_tv(x, y) = convert(*in_tv(y, x));
    }

    double eps = std::is_same<Convert, identity>::value ? 0 : 1e-5;

    Check(cpu_out, cpu_ref, EqualEps(eps));
  }

  void ConvertTimeMajorInPlace() {
    std::mt19937_64 rng;
    TensorListShape<2> in_shape, out_shape;
    std::uniform_int_distribution<int> dist(1, 500);
    int N = 10;
    int fft = 200;  // deliberately not a multiple of 32
    in_shape.resize(N);
    out_shape.resize(N);
    for (int i = 0; i < N; i++) {
      int len = dist(rng);
      int ratio = sizeof(In) / sizeof(Out);
      in_shape.set_tensor_shape(i, { len, fft });
      out_shape.set_tensor_shape(i, { len, fft * ratio });
    }
    TestTensorList<In, 2> in;
    TestTensorList<Out, 2> out, ref;
    TensorListView<StorageGPU, Out, 2> out_gpu;
    in.reshape(in_shape);
    auto cpu_in = in.cpu();
    UniformRandomFill(cpu_in, rng, -1, 1);

    ref.reshape(out_shape);
    auto cpu_ref = ref.cpu();

    Convert convert;
    for (int i = 0; i < N; i++) {
      TensorView<StorageCPU, In, 2> in_tv = cpu_in[i];
      TensorView<StorageCPU, Out, 2> ref_tv = cpu_ref[i];
      for (int i = 0; i < in_tv.shape[0]; i++)
        for (int j = 0; j < in_tv.shape[1]; j++)
          *ref_tv(i, j) = convert(*in_tv(i, j));
    }

    out_gpu = make_tensor_list_gpu(reinterpret_cast<Out*>(in.gpu().data[0]), out_shape);

    ConvertTimeMajorSpectrum<Out, In, Convert> tr;
    KernelContext ctx;
    ctx.gpu.stream = 0;
    tr.Setup(ctx, in_shape);
    tr.Run(ctx, out_gpu, in.gpu());
    CUDA_CALL(cudaGetLastError());
    out.reshape(out_shape);
    auto cpu_out = out.cpu();

    for (int i = 0; i < N; i++)
      copy(cpu_out[i], out_gpu[i], ctx.gpu.stream);
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));

    Compare(cpu_out, cpu_ref, in_shape);
  }

  void ConvertTimeMajorPadded() {
    std::mt19937_64 rng;
    TensorListShape<2> shape, padded_shape;
    std::uniform_int_distribution<int> dist(1, 500);
    int N = 10;
    int padded_nfft = 256;
    int nfft = 200;  // deliberately not a multiple of 32
    shape.resize(N);
    padded_shape.resize(N);

    for (int i = 0; i < N; i++) {
      int len = dist(rng);
      shape.set_tensor_shape(i, { len, nfft });
      padded_shape.set_tensor_shape(i, { len, padded_nfft });
    }
    TestTensorList<In, 2> in;
    TestTensorList<Out, 2> out, ref;
    in.reshape(padded_shape);
    out.reshape(shape);
    auto cpu_in = in.cpu();
    UniformRandomFill(cpu_in, rng, -1, 1);

    ref.reshape(shape);
    auto cpu_ref = ref.cpu();

    Convert convert;
    for (int s = 0; s < N; s++) {
      TensorView<StorageCPU, In, 2> in_tv = cpu_in[s];
      TensorView<StorageCPU, Out, 2> ref_tv = cpu_ref[s];
      auto sh = shape.tensor_shape_span(s);
      for (int i = 0; i < sh[0]; i++)
        for (int j = 0; j < sh[1]; j++)
          *ref_tv(i, j) = convert(*in_tv(i, j));
    }

    ConvertTimeMajorSpectrum<Out, In, Convert> tr;
    KernelContext ctx;
    ctx.gpu.stream = 0;
    tr.Setup(ctx, padded_shape);
    tr.Run(ctx, out.gpu(), in.gpu());
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));

    Compare(out.cpu(), cpu_ref, shape);
  }

  /**
   * @brief Compares tensor list views excluding the padding area
   */
  void Compare(const TensorListView<StorageCPU, Out, 2> &cpu_out,
               const TensorListView<StorageCPU, Out, 2> &cpu_ref,
               const TensorListShape<>& tlv_sh) {
    int nsamples = cpu_out.size();
    // data requires padding - clear it so we don't make the comparison fail
    for (int s = 0; s < nsamples; s++) {
      TensorView<StorageCPU, Out, 2> ref_tv = cpu_ref[s];
      TensorView<StorageCPU, Out, 2> out_tv = cpu_out[s];
      auto sh = tlv_sh[s];
      for (int i = 0; i < ref_tv.shape[0]; i++) {
        for (int j = 0; j < ref_tv.shape[1]; j++) {
          if (i >= sh[0] || j >= sh[1]) {
            *ref_tv(i, j) = {};
            *out_tv(i, j) = {};
          }
        }
      }
    }
    double eps = std::is_same<Convert, identity>::value ? 0 : 1e-5;
    Check(cpu_out, cpu_ref, EqualEps(eps));
  }
};

using FFTPosprocessTestTypes = ::testing::Types<
  FFTPostprocessArgs<float, float,   identity>,
  FFTPostprocessArgs<float2, float2, identity>,
  FFTPostprocessArgs<float, float2,  norm2>,
  FFTPostprocessArgs<float, float2,  norm2square>,
  FFTPostprocessArgs<float, float2,  power_dB>
>;

TYPED_TEST_SUITE(FFTPostprocessTest, FFTPosprocessTestTypes);


TYPED_TEST(FFTPostprocessTest, ToFreqMajor) {
  this->ToFreqMajor();
}

TYPED_TEST(FFTPostprocessTest, ConvertTimeMajorInPlace) {
  this->ConvertTimeMajorInPlace();
}

TYPED_TEST(FFTPostprocessTest, ConvertTimeMajorPadded) {
  this->ConvertTimeMajorPadded();
}


}  // namespace fft_postprocess
}  // namespace signal
}  // namespace kernels
}  // namespace dali
