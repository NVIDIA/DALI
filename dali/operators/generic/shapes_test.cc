// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/generic/shapes.h"
#include "dali/test/dali_operator_test.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {

template <typename Backend, typename RNG>
void GenerateShapeTestInputs(TensorList<Backend> &out, RNG &rng, int num_samples, int sample_dim) {
  TensorListShape<> shape;
  // this should give a distribution such that the batch is no bigger than 1e+8 elements
  int max = std::ceil(std::pow(1e+6 / num_samples, 1.0 / sample_dim));
  std::uniform_int_distribution<int> dist(1, max);

  shape.resize(num_samples, sample_dim);
  for (int i = 0; i < num_samples; i++) {
    for (int j = 0; j < sample_dim; j++) {
      shape.tensor_shape_span(i)[j] = dist(rng);
    }
  }
  out.Reset();
  out.Resize(shape);
  out.set_type(TypeTable::GetTypeInfoFromStatic<uint8_t>());
}

template <typename OutputBackend, typename InputBackend, typename OutputType>
struct ShapesTestArgs {
  using out_backend = OutputBackend;
  using in_backend = InputBackend;
  using output_type = OutputType;

  static DALIDataType type_id() {
    return TypeTable::GetTypeID<output_type>();
  }
};

template <typename TestArgs>
class ShapesOpTest;

template <typename OutputBackend, typename InputBackend, typename OutputType>
class ShapesOpTest<ShapesTestArgs<OutputBackend, InputBackend, OutputType>>
: public testing::DaliOperatorTest {
 public:
  using TestArgs = ShapesTestArgs<OutputBackend, InputBackend, OutputType>;

  std::vector<std::unique_ptr<TensorList<InputBackend>>> inputs;

  ShapesOpTest() {
    std::mt19937_64 rng(12345);
    for (int dim = 1; dim <= 6; dim++) {
      inputs.emplace_back(new TensorList<InputBackend>());
      inputs.back()->set_pinned(false);
      int num_samples = 1 << (8-dim);  // Start with 128 and halve with each new dimension
      GenerateShapeTestInputs(*inputs.back(), rng, num_samples, dim);
    }
  }

  testing::GraphDescr GenerateOperatorGraph() const override {
      return {"Shapes"};
  }

  void Run() {
    testing::Arguments args;
    args.emplace("dtype", TestArgs::type_id());
    args.emplace("device", testing::detail::BackendStringName<OutputBackend>());
    for (auto &in : inputs) {
      testing::TensorListWrapper out;
      this->RunTest(in.get(), out, args, VerifyShape);
    }
  }

  static void VerifyShape(
      const testing::TensorListWrapper &in_wrapper,
      const testing::TensorListWrapper &out_wrapper,
      const testing::Arguments &args) {
    ASSERT_TRUE(in_wrapper.has<InputBackend>());
    ASSERT_TRUE(out_wrapper.has<OutputBackend>());
    auto &in = *in_wrapper.get<InputBackend>();
    auto &out = *out_wrapper.get<OutputBackend>();
    VerifyShapeImpl(in, out);
  }

  static void VerifyShapeImpl(
      const TensorList<CPUBackend> &in,
      const TensorList<GPUBackend> &out) {
    TensorList<CPUBackend> tmp;
    tmp.Copy(out, 0);
    CUDA_CALL(cudaDeviceSynchronize());
    VerifyShapeImpl(in, tmp);
  }

  static void VerifyShapeImpl(
      const TensorList<CPUBackend> &in,
      const TensorList<CPUBackend> &out) {
    auto shape = in.shape();
    auto out_shape = out.shape();
    const int N = shape.num_samples();
    const int D = shape.sample_dim();
    ASSERT_EQ(N, out_shape.num_samples());
    ASSERT_TRUE(is_uniform(out_shape));
    ASSERT_EQ(out_shape.sample_dim(), 1);
    ASSERT_EQ(out_shape[0][0], D);

    for (int i = 0; i < N; i++) {
      const OutputType *shape_data = out.template tensor<OutputType>(i);
      auto tshape = shape.tensor_shape_span(i);
      for (int j = 0; j < D; j++) {
        EXPECT_EQ(shape_data[j], static_cast<OutputType>(tshape[j]));
      }
    }
  }
};

using ShapesOpArgs = ::testing::Types<
  ShapesTestArgs<CPUBackend, CPUBackend, int32_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, uint32_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, int64_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, uint64_t>,
  ShapesTestArgs<CPUBackend, CPUBackend, float>,
  ShapesTestArgs<CPUBackend, CPUBackend, double>,

  ShapesTestArgs<GPUBackend, CPUBackend, int32_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, uint32_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, int64_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, uint64_t>,
  ShapesTestArgs<GPUBackend, CPUBackend, float>,
  ShapesTestArgs<GPUBackend, CPUBackend, double>>;

TYPED_TEST_SUITE(ShapesOpTest, ShapesOpArgs);

TYPED_TEST(ShapesOpTest, All) {
    this->Run();
}


}  // namespace dali
