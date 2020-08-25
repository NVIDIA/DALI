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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/constant_storage.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

class ConstantStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    constant_ptrs_.reserve(constant_nodes_.size());
    for (auto &c : constant_nodes_) {
      constant_ptrs_.push_back(&c);
    }

    good_spec_.AddArg("integer_constants", integers_);
    good_spec_.AddArg("real_constants", reals_);

    bad_spec_.AddArg("integer_constants", std::vector<int>{1, 42});
    bad_spec_.AddArg("real_constants", std::vector<float>{0.1f, 0.42f, 54.f});
  }

  std::vector<ExprConstant> constant_nodes_ = {
    {0, DALIDataType::DALI_UINT8},
    {1, DALIDataType::DALI_INT32},
    {0, DALIDataType::DALI_FLOAT},
    {2, DALIDataType::DALI_INT32},
    {1, DALIDataType::DALI_FLOAT16},
    {3, DALIDataType::DALI_BOOL},
  };

  std::vector<ExprConstant*> constant_ptrs_;

  OpSpec good_spec_;
  std::vector<int> integers_ = {1, 1024, 42, false};
  std::vector<float> reals_ = {0.1f, 0.42f};

  OpSpec bad_spec_;
};

TEST_F(ConstantStorageTest, CpuValid) {
  ConstantStorage<CPUBackend> st;
  st.Initialize(good_spec_, 0, constant_ptrs_);
  for (auto &node : constant_nodes_) {
    auto type_id = node.GetTypeId();
    TYPE_SWITCH(type_id, type2id, Type, CONSTANT_STORAGE_ALLOWED_TYPES, (
          auto idx = node.GetConstIndex();
          auto *ptr = reinterpret_cast<const Type *>(st.GetPointer(idx, type_id));
          if (IsIntegral(type_id)) {
            EXPECT_EQ(*ptr, static_cast<Type>(integers_[idx]));
          } else {
            EXPECT_EQ(*ptr, static_cast<Type>(reals_[idx]));
          }
      ), DALI_FAIL(make_string("Unsupported type: ", type_id)););  // NOLINT(whitespace/parens)
  }
}

TEST_F(ConstantStorageTest, GpuValid) {
  ConstantStorage<GPUBackend> st;
  st.Initialize(good_spec_, 0, constant_ptrs_);
  char buf[sizeof(int64_t)];
  for (auto &node : constant_nodes_) {
    auto type_id = node.GetTypeId();
    TYPE_SWITCH(type_id, type2id, Type, CONSTANT_STORAGE_ALLOWED_TYPES, (
          auto idx = node.GetConstIndex();
          const auto *dev_ptr = st.GetPointer(idx, type_id);
          cudaMemcpy(buf, dev_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost);
          auto *ptr = reinterpret_cast<const Type *>(buf);
          if (IsIntegral(type_id)) {
            EXPECT_EQ(*ptr, static_cast<Type>(integers_[idx]));
          } else {
            EXPECT_EQ(*ptr, static_cast<Type>(reals_[idx]));
          }
      ), DALI_FAIL(make_string("Unsupported type: ", type_id)););  // NOLINT(whitespace/parens)
  }
}

TEST_F(ConstantStorageTest, Invalid) {
  ConstantStorage<CPUBackend> cpu_st;
  ASSERT_THROW(cpu_st.Initialize(bad_spec_, 0, constant_ptrs_), std::runtime_error);
  ConstantStorage<GPUBackend> gpu_st;
  ASSERT_THROW(gpu_st.Initialize(bad_spec_, 0, constant_ptrs_), std::runtime_error);
}

}  // namespace dali
