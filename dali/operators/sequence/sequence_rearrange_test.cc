// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/sequence/sequence_rearrange.h"
#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"
#include "dali/core/tensor_shape.h"

namespace dali {

namespace {

template <typename T>
void FillSeq(T* ptr, const TensorShape<>& shape) {
  auto element_size = volume(shape.last(shape.sample_dim() - 1));
  for (int i = 0; i < GetSeqLength(shape); i++) {
    auto ith_offset = element_size * i;
    for (int j = 0; j < element_size; j++) {
      ptr[ith_offset + j] = i;
    }
  }
}

class SequenceRearrangeBaseTest : public testing::DaliOperatorTest {
  testing::GraphDescr GenerateOperatorGraph() const noexcept override {
    return {"SequenceRearrange"};
  }

 public:
  std::unique_ptr<TensorList<CPUBackend>> getInput() {
    constexpr int batch_size = 5;
    TensorShape<> seq_shape_tmp{8, 4, 2, 2};
    TensorListShape<> batch_shape(batch_size, seq_shape_tmp.sample_dim());
    for (int i = 0; i < batch_size; i++) {
      batch_shape.set_tensor_shape(i, seq_shape_tmp);
      seq_shape_tmp[2]++;
    }

    std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
    tl->Resize(batch_shape);
    tl->set_type(TypeInfo::Create<int>());

    // Fill frames with consecutive numbers
    for (int i = 0; i < batch_size; i++) {
      FillSeq(tl->mutable_tensor<int>(i), batch_shape[i]);
    }
    return tl;
  }
};

struct SequenceRearrangeValidTest : public SequenceRearrangeBaseTest {};
struct SequenceRearrangeInvalidTest : public SequenceRearrangeBaseTest {
  std::unique_ptr<TensorList<CPUBackend>> getInputInvalid() {
    constexpr int batch_size = 5;
    TensorListShape<> batch_shape = uniform_list_shape(batch_size, {8});

    std::unique_ptr<TensorList<CPUBackend>> tl(new TensorList<CPUBackend>);
    tl->Resize(batch_shape);
    tl->set_type(TypeInfo::Create<int>());
    return tl;
  }
};

std::vector<testing::Arguments> reorders = {
    {{"new_order", std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}}},
    {{"new_order", std::vector<int>{0, 1, 2, 3, 4, 5, 6}}},
    {{"new_order", std::vector<int>{0, 7}}},
    {{"new_order", std::vector<int>{3}}},
    {{"new_order", std::vector<int>{2, 0, 1}}},
    {{"new_order", std::vector<int>{0, 0, 0}}},
    {{"new_order", std::vector<int>{5, 5, 5}}},
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
    {{"device", "cpu"}},
    {{"device", "gpu"}},
};

template <typename T>
void CheckRearrange(const T* ptr, const TensorShape<>& old_shape, const TensorShape<>& new_shape,
                    const std::vector<int>& new_order) {
  auto old_element_size = volume(old_shape.last(old_shape.sample_dim() - 1));
  auto new_element_size = volume(new_shape.last(new_shape.sample_dim() - 1));
  ASSERT_EQ(old_element_size, new_element_size);
  ASSERT_EQ(new_order.size(), GetSeqLength(new_shape));
  for (int i = 0; i < GetSeqLength(new_shape); i++) {
    auto elem_offset = new_element_size * i;
    for (int j = 0; j < new_element_size; j++) {
      EXPECT_EQ(ptr[elem_offset + j], new_order[i]);
    }
  }
}

void SeqRearrangeVerify(const testing::TensorListWrapper& input,
                        const testing::TensorListWrapper& output, const testing::Arguments& args) {
  auto in = input.CopyTo<CPUBackend>();
  auto out = output.CopyTo<CPUBackend>();
  auto order = args.at(testing::ArgumentKey("new_order")).GetValue<std::vector<int>>();
  for (size_t i = 0; i < out->ntensor(); i++) {
    CheckRearrange(out->tensor<int>(i), in->tensor_shape(i), out->tensor_shape(i), order);
  }
}

}  // namespace

TEST_P(SequenceRearrangeValidTest, GoodRearranges) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  this->RunTest(getInput().get(), tlout, args, SeqRearrangeVerify);
}

TEST_P(SequenceRearrangeInvalidTest, InvalidArgs) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  ASSERT_THROW(this->RunTest(getInput().get(), tlout, args, SeqRearrangeVerify),
               std::runtime_error);
}

TEST_P(SequenceRearrangeInvalidTest, InvalidInputs) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  ASSERT_THROW(this->RunTest(getInputInvalid().get(), tlout, args, SeqRearrangeVerify),
               std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(SequenceRearrangeSuite, SequenceRearrangeValidTest,
                         ::testing::ValuesIn(cartesian(devices, reorders)));
INSTANTIATE_TEST_SUITE_P(SequenceRearrangeSuite, SequenceRearrangeInvalidTest,
                         ::testing::ValuesIn(cartesian(devices, wrong_reorders)));

}  // namespace dali
