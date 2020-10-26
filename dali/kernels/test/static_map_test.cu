// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <typeinfo>
#include <vector>
#include <unordered_map>

#include "dali/kernels/kernel.h"
#include "dali/core/static_map.h"
#include "dali/kernels/type_tag.h"

#define TEST_TYPES_MAP ( \
    ((uint8_t), (uint8_t, uint64_t, float)), \
    ((int8_t), (int64_t)), \
    ((uint16_t), (uint16_t, uint64_t)), \
    ((int16_t), (int16_t)), \
    ((uint32_t), (uint32_t, float)), \
    ((int32_t), (int32_t)), \
    ((int64_t), (int64_t)), \
    ((float), (float)))

namespace {
template <typename T, typename S>
void TypedFunc() {}

using TypeTagMap = std::unordered_map<int, std::vector<int>>;

template <typename T>
struct StaticMapNVCC : public testing::Test {
  StaticMapNVCC() {
    type_mapping_[dali::TypeTag<uint8_t>()] = {
      dali::TypeTag<uint8_t>(), dali::TypeTag<uint64_t>(), dali::TypeTag<float>() };
    type_mapping_[dali::TypeTag<int8_t>()] = { dali::TypeTag<int64_t>() };
    type_mapping_[dali::TypeTag<uint16_t>()] = {
      dali::TypeTag<uint16_t>(), dali::TypeTag<uint64_t>() };
    type_mapping_[dali::TypeTag<int16_t>()] = { dali::TypeTag<int16_t>() };
    type_mapping_[dali::TypeTag<uint32_t>()] = {
      dali::TypeTag<uint32_t>(), dali::TypeTag<float>() };
    type_mapping_[dali::TypeTag<int32_t>()] = { dali::TypeTag<int32_t>() };
    type_mapping_[dali::TypeTag<int64_t>()] = { dali::TypeTag<int64_t>() };
    type_mapping_[dali::TypeTag<float>()] = { dali::TypeTag<float>() };
  }

 protected:
  template <typename P, typename R>
  void TypedMethod(int input_arg_type, int output_arg_type) {
    int input_call_type = dali::TypeTag<P>();
    int output_call_type = dali::TypeTag<R>();

    ASSERT_EQ(input_arg_type, input_call_type);
    ASSERT_EQ(output_arg_type, output_call_type);
  }
  TypeTagMap type_mapping_;
};

}  // namespace

typedef testing::Types<
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, int64_t, float> TypesToTest;

TYPED_TEST_SUITE(StaticMapNVCC, TypesToTest);

TYPED_TEST(StaticMapNVCC, ValidTypes) {
  using TestedType = gtest_TypeParam_;

  int input_type = dali::TypeTag<TestedType>();
  for (auto &output_type : this->type_mapping_[input_type]) {
    TYPE_MAP(
      input_type,
      output_type,
      dali::TypeTag,
      InputType,
      OutputType,
      TEST_TYPES_MAP,
      (this->template TypedMethod<InputType, OutputType>(input_type, output_type);),
      (FAIL()),
      (FAIL()))
  }
}

TEST(StaticMapFailureNVCC, InvalidInputType) {
  int input_type = dali::TypeTag<uint64_t>();
  int output_type = dali::TypeTag<int32_t>();
  TYPE_MAP(
    input_type,
    output_type,
    dali::TypeTag,
    InputType,
    OutputType,
    TEST_TYPES_MAP,
    (TypedFunc<InputType, OutputType>(); FAIL();),
    (SUCCEED()),
    (FAIL()))
}

TEST(StaticMapFailureNVCC, InvalidOutputType) {
  int input_type = dali::TypeTag<float>();
  int output_type = dali::TypeTag<int32_t>();
  TYPE_MAP(
    input_type,
    output_type,
    dali::TypeTag,
    InputType,
    OutputType,
    TEST_TYPES_MAP,
    (TypedFunc<InputType, OutputType>(); FAIL();),
    (FAIL()),
    (SUCCEED()))
}
