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

#include <iostream>

#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"
#include "dali/pipeline/operators/sequence/seq_rearrange.h"

namespace dali {

namespace {

template <typename T>
void FillSeq(T* ptr, const std::vector<Index>& shape) {
  auto element_size = Product(shape) / GetSeqLength(shape);
  for (int i = 0; i < GetSeqLength(shape); i++) {
    auto ith_offset = element_size * i;
    for (int j = 0; j < element_size; j++) {
      ptr[ith_offset + j] = i;
    }
  }
}

}  // namespace

class SequenceRearrangeBaseTest : public testing::DaliOperatorTest {
  testing::GraphDescr GenerateOperatorGraph() const noexcept override {
    return {"SequenceRearrange"};
  }

 public:
  std::unique_ptr<TensorList<CPUBackend>> getInput() {
    constexpr int  batch_size = 10;
    std::vector<Index> seq_shape{8, 4, 2, 2};
    std::vector<std::vector<Index>> batch_shape;
    // repeat seq_shape for whole batch
    for (int i = 0; i < batch_size; i++) {
      batch_shape.push_back(seq_shape);
    }

    std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
    tl->set_type(TypeInfo::Create<int>());
    tl->Resize(batch_shape);

    // Fill frames with consecutive numbers
    for (int i = 0; i < batch_size; i++) {
      FillSeq(tl->mutable_tensor<int>(i), seq_shape);
    }
    return tl;
  }
};

struct SequenceRearrangeValidTest : public SequenceRearrangeBaseTest {};
struct SequenceRearrangeInvalidTest : public SequenceRearrangeBaseTest {
  std::unique_ptr<TensorList<CPUBackend>> getInputInvalid() {
    constexpr int  batch_size = 10;
    std::vector<Index> seq_shape{8};
    std::vector<std::vector<Index>> batch_shape;
    // repeat seq_shape for whole batch
    for (int i = 0; i < batch_size; i++) {
      batch_shape.push_back(seq_shape);
    }

    std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
    tl->set_type(TypeInfo::Create<int>());
    tl->Resize(batch_shape);
    return tl;
  }
};

std::vector<testing::Arguments> reorders = {
      {{"new_order", std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}}},
      {{"new_order", std::vector<int>{0, 1, 2, 3, 4, 5, 6}}},
      {{"new_order", std::vector<int>{0, 7}}},
      {{"new_order", std::vector<int>{0, 0, 0}}},
      {{"new_order", std::vector<int>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}}},
      {{"new_order", std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0}}},
};

std::vector<testing::Arguments> wrong_reorders = {
      {{"new_order", std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8}}},
      {{"new_order", std::vector<int>{9}}},
      {{"new_order", std::vector<int>{-1}}},
      {{"new_order", std::vector<int>{1, -1}}},
      {{"new_order", std::vector<int>{}}},
};

std::vector<testing::Arguments> devices = {
  {{"device", std::string{"cpu"}}},
  // {{"device", std::string{"gpu"}}},
};



template <typename T>
void CheckRearrange(T *ptr, const std::vector<Index>& old_shape, const std::vector<Index>& new_shape, const std::vector<int>& new_order) {
  auto old_element_size = Product(old_shape) / GetSeqLength(old_shape);
  auto new_element_size = Product(new_shape) / GetSeqLength(new_shape);
  ASSERT_EQ(old_element_size, new_element_size);
  ASSERT_EQ(new_order.size(), GetSeqLength(new_shape));
  for (int i = 0; i < GetSeqLength(new_shape); i++) {
    auto elem_offset = new_element_size * i;
    for (int j = 0; j < new_element_size; j++) {
      EXPECT_EQ(ptr[elem_offset + j], new_order[i]);
    }
  }

}


// TODO unify Verify
void SeqRearrangeVerify(const testing::TensorListWrapper& input, const testing::TensorListWrapper& output, const testing::Arguments& args) {
  auto* in = input.get<CPUBackend>();
  auto* out = output.get<CPUBackend>();

  auto order = args.at(testing::ArgumentKey("new_order")).GetValue<std::vector<int>>();
  for (int i = 0; i < out->ntensor(); i++) {
    CheckRearrange(out->tensor<int>(i), in->tensor_shape(i), out->tensor_shape(i), order);
  }
}


TEST_P(SequenceRearrangeValidTest, GoodRearranges) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  auto in = getInput();
  testing::TensorListWrapper tlin(in.get());
  this->RunTest<CPUBackend>(tlin, tlout, args, SeqRearrangeVerify);
}

TEST_P(SequenceRearrangeInvalidTest, InvalidArgs) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  auto in = getInput();
  testing::TensorListWrapper tlin(in.get());
  ASSERT_THROW(this->RunTest<CPUBackend>(tlin, tlout, args, SeqRearrangeVerify), std::runtime_error);
}

TEST_P(SequenceRearrangeInvalidTest, InvalidInputs) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  auto in = getInputInvalid();
  testing::TensorListWrapper tlin(in.get());
  ASSERT_THROW(this->RunTest<CPUBackend>(tlin, tlout, args, SeqRearrangeVerify), std::runtime_error);
}


INSTANTIATE_TEST_CASE_P(SequenceRearrangeSuite, SequenceRearrangeValidTest, ::testing::ValuesIn(cartesian(devices, reorders)));
INSTANTIATE_TEST_CASE_P(SequenceRearrangeSuite, SequenceRearrangeInvalidTest, ::testing::ValuesIn(cartesian(devices, wrong_reorders)));

}  // namespace dali
