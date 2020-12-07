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
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace testing {

OpSpec MakeOpSpec(const std::string& operator_name) {
  return OpSpec(operator_name)
    .AddArg("num_threads", 1)
    .AddArg("max_batch_size", 32)
    .AddArg("device", "cpu");
}

TEST(InstantiateOperator, ValidOperatorName) {
  ASSERT_NE(nullptr,
    InstantiateOperator(
      MakeOpSpec("Crop")));
}

TEST(InstantiateOperator, InvalidOperatorName) {
  ASSERT_THROW(
    InstantiateOperator(
      MakeOpSpec("DoesNotExist")),
    std::runtime_error);
}

TEST(InstantiateOperator, RunMethodIsAccessible) {
  HostWorkspace ws;
  auto op = InstantiateOperator(MakeOpSpec("ImageDecoder"));
  // We just want to test that Run method is visible (exported to the so file)
  // It is expected that the call throws as the worspace is empty
  ASSERT_THROW(op->Run(ws), std::runtime_error);
}


enum TestEnum : int {
  TEST_ENUM = 42
};

template<typename T>
class OperatorDiagnosticsTest : public ::testing::Test {
 protected:
  void SetUp() final {
    assign_value();
    auto op_spec = OpSpec("CoinFlip").AddArg("num_threads", 1).AddArg("max_batch_size", 1);
    operator_ = std::make_unique<OperatorBase>(op_spec);
  }

  void assign_value() {
    this->value_ = 42;
  }

  std::unique_ptr<OperatorBase> operator_;
  std::string value_name_ = "Lorem ipsum";
  T value_;
};

template<>
void OperatorDiagnosticsTest<bool>::assign_value() {
  this->value_ = true;
}

template<>
void OperatorDiagnosticsTest<TestEnum>::assign_value() {
  this->value_ = TEST_ENUM;
}

using DiagnosticsTypes = ::testing::Types<int, unsigned int, int8_t, uint16_t, int32_t, uint64_t,
                                          float, double, half_float::half, bool, TestEnum>;
TYPED_TEST_SUITE(OperatorDiagnosticsTest, DiagnosticsTypes);

TYPED_TEST(OperatorDiagnosticsTest, DiagnosticsTest) {
  (this->operator_)->RegisterDiagnostic(this->value_name_, &this->value_);
  auto cnt = this->operator_->template GetDiagnostic<TypeParam>(this->value_name_);
  ASSERT_EQ(this->value_, cnt);
}


TYPED_TEST(OperatorDiagnosticsTest, DiagnosticsCollisionTest) {
  (this->operator_)->RegisterDiagnostic(this->value_name_, &this->value_);
  EXPECT_THROW((this->operator_)->RegisterDiagnostic(this->value_name_, &this->value_),
               std::runtime_error);
}


TYPED_TEST(OperatorDiagnosticsTest, IncorrectTypeTest) {
  (this->operator_)->RegisterDiagnostic(this->value_name_, &this->value_);
  EXPECT_THROW(this->operator_->template GetDiagnostic<int64_t>(this->value_name_),
               std::runtime_error);
}


TYPED_TEST(OperatorDiagnosticsTest, NonexistingParameterTest) {
  EXPECT_THROW(this->operator_->template GetDiagnostic<TypeParam>(this->value_name_),
               std::runtime_error);
}


}  // namespace testing
}  // namespace dali
