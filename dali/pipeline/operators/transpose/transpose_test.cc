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
#include <numeric>
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
  auto vol = volume(shape);
  for (int i = 0; i < vol; ++i) {
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
    if (rank > 3) {
      seq_shape.push_back(2);
    }
    if (rank > 4) {
      seq_shape.push_back(3);
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
class TransposeTestRank5 : public TransposeTest {};

inline int Factorial(int n) {
  int ret = 1;
  for (; n > 0; --n) {
    ret *= n;
  }
  return ret;
}

std::vector<testing::Arguments> GetPermutations(int rank) {
  std::vector<int> to_permute(rank);
  std::iota(to_permute.begin(), to_permute.end(), 0);

  std::vector<testing::Arguments> perms;
  perms.reserve(Factorial(to_permute.size()));
  do {
    perms.push_back({{"perm", to_permute}});
  } while (std::next_permutation(to_permute.begin(), to_permute.end()));
  return perms;
}

std::vector<testing::Arguments> devices = {
// CPU transpose not supported yet
//    {{"device", std::string{"cpu"}}},
    {{"device", std::string{"gpu"}}},
};

namespace detail {

template <typename T, int Rank, int CurrDim>
inline typename std::enable_if<Rank == CurrDim>::type
tensor_loop_impl(const T* in_tensor,
                 const T* out_tensor,
                 const std::vector<Index>& /*unused*/,
                 const std::vector<int>& /*unused*/, const std::vector<int>& /*unused*/,
                 const std::vector<int>& /*unused*/,
                 int in_idx, int out_idx) {
  EXPECT_EQ(in_tensor[in_idx], out_tensor[out_idx]);
}

template <typename T, int Rank, int CurrDim>
inline typename std::enable_if<Rank != CurrDim>::type
tensor_loop_impl(const T* in_tensor,
                 const T* out_tensor,
                 const std::vector<Index>& shape,
                 const std::vector<int>& old_strides, const std::vector<int>& new_strides,
                 const std::vector<int>& perm,
                 int in_idx, int out_idx) {
  for (int i = 0; i < shape[CurrDim]; ++i) {
    tensor_loop_impl<T, Rank, CurrDim +1>(in_tensor,
                                      out_tensor,
                                      shape, old_strides, new_strides, perm,
                                      in_idx + old_strides[perm[CurrDim]] * i,
                                      out_idx + new_strides[CurrDim] * i);
  }
}

template <typename T, int Rank>
inline void tensor_loop(const T* in_tensor,
                        const T* out_tensor,
                        const std::vector<Index>& shape,
                        const std::vector<int>& old_strides, const std::vector<int>& new_strides,
                        const std::vector<int>& perm) {
  detail::tensor_loop_impl<T, Rank, 0>(in_tensor, out_tensor,
                                       shape, old_strides, new_strides, perm,
                                       0, 0);
}

}  // namespace detail

template <typename T>
void CheckTransposition(const T* in_tensor, const T* out_tensor,
                        const std::vector<Index>& old_shape,
                        const std::vector<Index>& new_shape,
                        const std::vector<int>& perm) {
  auto old_volume = volume(old_shape);
  auto new_volume = volume(new_shape);
  ASSERT_EQ(old_volume, new_volume);

  auto old_strides = GetStrides(old_shape);
  auto new_strides = GetStrides(new_shape);

  if (new_shape.size() == 3) {
    detail::tensor_loop<T, 3>(in_tensor, out_tensor, new_shape, old_strides, new_strides, perm);
  } else if (new_shape.size() == 4) {
    detail::tensor_loop<T, 4>(in_tensor, out_tensor, new_shape, old_strides, new_strides, perm);
  } else if (new_shape.size() == 5) {
    detail::tensor_loop<T, 5>(in_tensor, out_tensor, new_shape, old_strides, new_strides, perm);
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

TEST_P(TransposeTestRank5, TransposeRank5) {
  auto args = GetParam();
  testing::TensorListWrapper tlout;
  this->RunTest(GetInput(5).get(), tlout, args, TransposeVerify);
}

INSTANTIATE_TEST_SUITE_P(TransposeRank3Suite, TransposeTestRank3,
                        ::testing::ValuesIn(cartesian(devices, GetPermutations(3))));
INSTANTIATE_TEST_SUITE_P(TransposeRank4Suite, TransposeTestRank4,
                        ::testing::ValuesIn(cartesian(devices, GetPermutations(4))));
INSTANTIATE_TEST_SUITE_P(TransposeRank5Suite, TransposeTestRank5,
                        ::testing::ValuesIn(cartesian(devices, GetPermutations(5))));

}  // namespace dali
