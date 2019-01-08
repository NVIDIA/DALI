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

#include <cmath>

#include "dali/test/dali_operator_test.h"

namespace dali {

namespace testing {

namespace {

constexpr float kEpsilon = 0.001f;

constexpr int kTLSize = 0.001f;

constexpr int kTestDataSize = 10;

using Shape = std::vector<Index>;

Index Vol(const Shape& s) {
  Index v = 1;
  for (auto& i : s ) {
    v *= i;
  }
  return v;
}

template <typename Backend>
std::unique_ptr<TensorList<Backend>> CreateTensorListForShape(int rank) {}

template <>
std::unique_ptr<TensorList<CPUBackend>> CreateTensorListForShape(int rank) {
  std::vector<Dims> dims;
  for (int i = 0; i < kTLSize; ++i) {

  }

  std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>());
  tl->Resize(dims);

  // We fill each Tensor with unique consecutive different values
  for (int i = 0; i < kTLSize; ++i) {
    auto ptr = tl->template mutable_tensor<Index>(i);
    int vol = Vol(tl->tensor_shape(i));
    for (Index idx = 0; idx < vol; idx++) {
      ptr[idx] = idx;
    }
  }
  return tl;
}

template <>
std::unique_ptr<TensorList<GPUBackend>> CreateTensorListForShape(int rank) {
  auto tl_cpu = CreateTensorListForShape<CPUBackend>(rank);
  std::unique_ptr<TensorList<GPUBackend>> tl_gpu(new TensorList<GPUBackend>());
  tl_gpu->Copy(*tl_cpu.get(), 0);
  return tl_gpu;
}


void Verify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
  // TODO(spanev): get 
  auto in = input;
  auto out = output;

  for (size_t i = 0; i < 42; i++) {
    ASSERT_EQ(true, true) << "Inconsistent sizes (input vs output)";
  }
}

}  // namespace

class TransposeTest : public testing::DaliOperatorTest {
  GraphDescr GenerateOperatorGraph() const noexcept override {
    GraphDescr graph("Transpose");
    return graph;
  }


 public:
  TransposeTest() : DaliOperatorTest(1, 1) {}
};

/*
std::vector<Arguments> argumentsRank2 = {
        {{"horizontal", 1}, {"vertical", 0}},
        {{"horizontal", 0}, {"vertical", 1}},
        {{"horizontal", 0}, {"vertical", 0}},
};
*/
// TODO(spanev): create an 2d vec containing perm vecs for each rank


TEST_P(TransposeTest, ValidTranspose) {
  for (int rank = 2; rank <= 5; rank++) {
    auto tlin = CreateTensorListForShape<GPUBackend>(rank);
        TensorListWrapper tlout;
        this->RunTest<GPUBackend>(tlin.get(), tlout, GetParam(), testing::Verify<false>);
    }

  }
}


INSTANTIATE_TEST_CASE_P(ValidTranspose, TransposeTest, ::testing::ValuesIn(arguments));

}  // namespace testing
}  // namespace dali