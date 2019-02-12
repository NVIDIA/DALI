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

#include <memory>
#include <string>
#include <vector>

#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"
#include "dali/pipeline/operators/transpose/transpose.h"

namespace dali {

namespace {

// Fill tensors of consecutive numbers
template <typename T>
void Arrange(T* ptr, const std::vector<Index>& shape) {
  auto volume = Volume(shape);
  for (int i = 0; i < volume; ++i) {
    ptr[i] = static_cast<T>(i);
  }
}

std::vector<int> GetStrides(const std::vector<Index>& shape) {
  std::vector<int> strides(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

}  // namespace

class TransposeTest : public testing::DaliOperatorTest {
  testing::GraphDescr GenerateOperatorGraph() const noexcept override {
    return {"Transpose"};
  }

 public:
  std::unique_ptr<TensorList<CPUBackend>> GetInput(int rank) {
    constexpr int batch_size = 10;
    std::vector<Index> seq_shape{4, 8, 6};
    if (rank == 4) {
      seq_shape.push_back(2);
    }
    std::vector<std::vector<Index>> batch_shape;
    for (int i = 0; i < batch_size; i++) {
      batch_shape.push_back(seq_shape);
    }

    std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
    tl->Resize(batch_shape);
    tl->set_type(TypeInfo::Create<int>());

    for (int i = 0; i < batch_size; i++) {
      Arrange(tl->mutable_tensor<int>(i), seq_shape);
    }
    return tl;
  }
};

class TransposeTestRank3 : public TransposeTest {};
class TransposeTestRank4 : public TransposeTest {};

std::vector<testing::Arguments> permutationsRank3 = {
    {{"perm", std::vector<int>{0, 1, 2}}},
    {{"perm", std::vector<int>{1, 2, 0}}},
    {{"perm", std::vector<int>{2, 1, 0}}},
    {{"perm", std::vector<int>{2, 0, 1}}},
    {{"perm", std::vector<int>{0, 2, 1}}},
    {{"perm", std::vector<int>{1, 0, 2}}},
};

std::vector<testing::Arguments> permutationsRank4 = {
    {{"perm", std::vector<int>{0, 1, 2, 3}}},
    {{"perm", std::vector<int>{3, 2, 1, 0}}},
    {{"perm", std::vector<int>{2, 3, 1, 0}}},
    {{"perm", std::vector<int>{1, 3, 2, 0}}},
    {{"perm", std::vector<int>{3, 1, 2, 0}}},
    {{"perm", std::vector<int>{1, 2, 3, 0}}},
    {{"perm", std::vector<int>{3, 2, 0, 1}}},
    {{"perm", std::vector<int>{2, 3, 0, 1}}},
    {{"perm", std::vector<int>{0, 3, 2, 1}}},
    {{"perm", std::vector<int>{3, 0, 2, 1}}},
    {{"perm", std::vector<int>{0, 2, 3, 1}}},
    {{"perm", std::vector<int>{3, 0, 1, 2}}},
    {{"perm", std::vector<int>{0, 3, 1, 2}}},
    {{"perm", std::vector<int>{1, 3, 0, 2}}},
    {{"perm", std::vector<int>{3, 1, 0, 2}}},
    {{"perm", std::vector<int>{1, 0, 3, 2}}},
    {{"perm", std::vector<int>{0, 2, 1, 3}}},
    {{"perm", std::vector<int>{2, 0, 1, 3}}},
    {{"perm", std::vector<int>{1, 0, 2, 3}}},
    {{"perm", std::vector<int>{1, 2, 0, 3}}},
};

std::vector<testing::Arguments> devices = {
// CPU transpose not supported yet
//    {{"device", std::string{"cpu"}}},
    {{"device", std::string{"gpu"}}},
};

template <typename T>
void CheckTransposition(const T* in_tensor, const T* out_tensor,
                        const std::vector<Index>& old_shape,
                        const std::vector<Index>& new_shape,
                        const std::vector<int>& perm) {
  auto old_volume = Volume(old_shape);
  auto new_volume = Volume(new_shape);
  ASSERT_EQ(old_volume, new_volume);


  auto old_strides = GetStrides(old_shape);
  auto new_strides = GetStrides(new_shape);
  if (new_shape.size() == 3) {
    for (int i = 0; i < new_shape[0]; ++i) {
      for (int j = 0; j < new_shape[1]; ++j) {
        for (int k = 0; k < new_shape[2]; ++k) {
          int in_idx = old_strides[perm[0]] * i
                       + old_strides[perm[1]] * j
                       + old_strides[perm[2]] * k;
          int out_idx = new_strides[0] * i
                        + new_strides[1] * j
                        + new_strides[2] * k;
          EXPECT_EQ(in_tensor[in_idx], out_tensor[out_idx]);
        }
      }
    }
  } else if (new_shape.size() == 4) {
    for (int i = 0; i < new_shape[0]; ++i) {
      for (int j = 0; j < new_shape[1]; ++j) {
        for (int k = 0; k < new_shape[2]; ++k) {
          for (int l = 0; l < new_shape[3]; ++l) {
            int in_idx = old_strides[perm[0]] * i
                         + old_strides[perm[1]] * j
                         + old_strides[perm[2]] * k
                         + old_strides[perm[3]] * l;
            int out_idx = new_strides[0] * i
                          + new_strides[1] * j
                          + new_strides[2] * k
                          + new_strides[3] * l;
            EXPECT_EQ(in_tensor[in_idx], out_tensor[out_idx]);
          }
        }
      }
    }
  }
}

void TransposeVerify(const testing::TensorListWrapper& input,
                     const testing::TensorListWrapper& output, const testing::Arguments& args) {
  auto in = input.CopyTo<CPUBackend>();
  auto out = output.CopyTo<CPUBackend>();
  auto perm = args.at(testing::ArgumentKey("perm")).GetValue<std::vector<int>>();
  for (decltype(out->ntensor()) i = 0; i < out->ntensor(); i++) {
    CheckTransposition(in->tensor<int>(i),
                       out->tensor<int>(i),
                       in->tensor_shape(i),
                       out->tensor_shape(i),
                       perm);
  }
}

TEST_P(TransposeTestRank3, TransposeRank3) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  this->RunTest(GetInput(3).get(), tlout, args, TransposeVerify);
}

TEST_P(TransposeTestRank4, TransposeRank4) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  this->RunTest(GetInput(4).get(), tlout, args, TransposeVerify);
}

INSTANTIATE_TEST_CASE_P(TransposeRank3Suite, TransposeTestRank3,
                        ::testing::ValuesIn(cartesian(devices, permutationsRank3)));
INSTANTIATE_TEST_CASE_P(TransposeRank4Suite, TransposeTestRank4,
                        ::testing::ValuesIn(cartesian(devices, permutationsRank4)));

}  // namespace dali
