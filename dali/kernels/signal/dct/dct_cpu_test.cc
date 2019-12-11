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
#include <tuple>
#include <vector>
#include <cmath>
#include <complex>
#include <cmath>
#include "dali/kernels/scratch.h"
#include "dali/kernels/signal/dct/dct_cpu.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

#undef LOG_LINE
#define LOG_LINE std::cout

namespace dali {
namespace kernels {
namespace signal {
namespace dct {
namespace test {

template <typename T>
void ReferenceDctTypeI(span<T> out, span<const T> in) {
	int64_t in_length = in.size();
  int64_t out_length = out.size();
	T phase_mul = M_PI / (in_length - 1);
	for (int64_t k = 0; k < out_length; k++) {
    T sign = (k % 2 == 0) ? T(1) : T(-1);
		T sum = T(0.5) * (in[0] + sign * in[in_length-1]);
		for (int64_t n = 0; n < in_length; n++) {
			sum += in[n] * std::cos(phase_mul * n * k);
    }
		out[k] = sum;
	}
}

template <typename T>
void ReferenceDctTypeII(span<T> out, span<const T> in) {
	int64_t in_length = in.size();
  int64_t out_length = out.size();
	T phase_mul = M_PI / in_length;
	for (int64_t k = 0; k < out_length; k++) {
		T sum = 0;
		for (int64_t n = 0; n < in_length; n++) {
			sum += in[n] * std::cos(phase_mul * (n + T(0.5)) * k);
    }
		out[k] = sum;
	}
}

template <typename T>
void ReferenceDctTypeIII(span<T> out, span<const T> in) {
	int64_t in_length = in.size();
  int64_t out_length = out.size();
	T phase_mul = M_PI / in_length;
	for (int64_t k = 0; k < out_length; k++) {
		T sum = T(0.5) * in[0];
		for (int64_t n = 0; n < in_length; n++) {
			sum += in[n] * std::cos(phase_mul * n * (k + T(0.5)));
    }
		out[k] = sum;
	}
}

template <typename T>
void ReferenceDctTypeIV(span<T> out, span<const T> in) {
	int64_t in_length = in.size();
  int64_t out_length = out.size();
	T phase_mul = M_PI / in_length;
	for (int64_t k = 0; k < out_length; k++) {
		T sum = 0;
		for (int64_t n = 0; n < in_length; n++) {
			sum += in[n] * std::cos(phase_mul * (n + T(0.5)) * (k + T(0.5)));
    }
		out[k] = sum;
	}
}


class Dct1DCpuTest : public::testing::TestWithParam<
  std::tuple<std::array<int64_t, 2>, int, int>> {
 public:
  Dct1DCpuTest()
    : data_shape_(std::get<0>(GetParam()))
    , dct_type_(std::get<1>(GetParam()))
    , axis_(std::get<2>(GetParam()))
    , data_(volume(data_shape_))
    , in_view_(data_.data(), data_shape_) {}

  ~Dct1DCpuTest() override = default;

 protected:
  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(in_view_, rng, 0., 1.);
  }
  TensorShape<2> data_shape_;
  int dct_type_ = 2;
  int axis_ = 1;

  std::vector<float> data_;
  OutTensorCPU<float, 2> in_view_;
};

TEST_P(Dct1DCpuTest, DctTest) {
  using OutputType = float;
  using InputType = float;
  constexpr int Dims = 2;
  Dct1DCpu<OutputType, InputType, Dims> kernel;
  check_kernel<decltype(kernel)>();

  KernelContext ctx;

  auto in_shape = in_view_.shape;
  ASSERT_TRUE(volume(in_shape) > 0);
  ASSERT_TRUE(axis_ >= 0 && axis_ < in_shape.size());
  ASSERT_TRUE(dct_type_ >= 1 && dct_type_ <= 4);

  auto n = in_shape[axis_];
  LOG_LINE << "Test n=" << n << " axis=" << axis_ << std::endl;

  DctArgs args;
  args.axis = axis_;
  args.dct_type = dct_type_;

  KernelRequirements reqs = kernel.Setup(ctx, in_view_, args);

  ScratchpadAllocator scratch_alloc;
  scratch_alloc.Reserve(reqs.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;

  TensorShape<> expected_out_shape = in_shape;
  auto out_shape = reqs.output_shapes[0][0];
  ASSERT_EQ(expected_out_shape, out_shape);

  auto out_size = volume(out_shape);
  std::vector<OutputType> out_data(out_size);

  auto out_view = OutTensorCPU<OutputType, 2>(out_data.data(), out_shape.to_static<2>());
  kernel.Run(ctx, out_view, in_view_, args);

  LOG_LINE << "Calculating reference\n";
  std::vector<OutputType> ref(n, 0);
  switch (dct_type_) {
    case 1:
      ReferenceDctTypeI(make_span(ref), make_cspan(in_view_.data, n));  
      break;

    case 2:
      ReferenceDctTypeII(make_span(ref), make_cspan(in_view_.data, n));
      break;

    case 3:
      ReferenceDctTypeIII(make_span(ref), make_cspan(in_view_.data, n));
      break;

    case 4:
      ReferenceDctTypeIV(make_span(ref), make_cspan(in_view_.data, n));
      break;

    default:
      ASSERT_TRUE(false);
  }

  LOG_LINE << "Reference DCT:" << std::endl;
  for (int k = 0; k < n; k++) {
    LOG_LINE << " " << ref[k] << "\n";
    ASSERT_NEAR(ref[k], out_data[k], 1e-4);
  }

}

INSTANTIATE_TEST_SUITE_P(Dct1DCpuTest, Dct1DCpuTest, testing::Combine(
    testing::Values(std::array<int64_t, 2>{8, 8}),  // shape
    testing::Values(1, 2, 3, 4),  // dct_type
    testing::Values(1)  // axis
  ));

}  // namespace test
}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
