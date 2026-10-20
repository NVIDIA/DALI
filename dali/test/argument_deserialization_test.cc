// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/operators/reader/parser/tf_feature.h"
#include "dali/pipeline/dali.pb.h"
#include "dali/pipeline/operator/argument.h"

namespace dali {

// Exercise the same entry point used by serialized pipeline arguments.
template <typename T>
class ArgumentDeserializationTest : public ::testing::Test {};

using ScalarArgumentTypes = ::testing::Types<int64_t, float, bool, std::string>;
TYPED_TEST_SUITE(ArgumentDeserializationTest, ScalarArgumentTypes);

TYPED_TEST(ArgumentDeserializationTest, MissingScalarValue) {
  dali_proto::Argument proto;
  DaliProtoPriv arg(&proto);
  arg.set_name("missing_value");
  arg.set_type(serialize_type(TypeParam{}));
  arg.set_is_vector(false);
  EXPECT_THROW(DeserializeProtobuf(arg), std::out_of_range);

  // A scalar nested in a vector must be checked as well.
  dali_proto::Argument vector_proto;
  DaliProtoPriv vector_arg(&vector_proto);
  vector_arg.set_name("vector");
  vector_arg.set_type(arg.type());
  vector_arg.set_is_vector(true);
  *vector_proto.add_extra_args() = proto;
  EXPECT_THROW(DeserializeProtobuf(vector_arg), std::out_of_range);
}

TYPED_TEST(ArgumentDeserializationTest, RoundTrip) {
  TypeParam nonzero;
  if constexpr (std::is_same_v<TypeParam, std::string>) {
    nonzero = "value";
  } else {
    nonzero = TypeParam{1};
  }
  for (const auto &value : {TypeParam{}, nonzero}) {
    dali_proto::Argument proto;
    DaliProtoPriv arg(&proto);
    Argument::Store("value", value)->SerializeToProtobuf(&arg);
    EXPECT_EQ(DeserializeProtobuf(arg)->template Get<TypeParam>(), value);
  }
  for (const auto &values : {std::vector<TypeParam>{}, std::vector<TypeParam>(2, nonzero)}) {
    dali_proto::Argument proto;
    DaliProtoPriv arg(&proto);
    Argument::Store("values", values)->SerializeToProtobuf(&arg);
    EXPECT_EQ(DeserializeProtobuf(arg)->template Get<std::vector<TypeParam>>(), values);
  }
}

TEST(DaliProtoIndexTest, Bounds) {
  dali_proto::Argument proto;
  DaliProtoPriv arg(&proto);
  arg.set_name("bounds");
  auto expect_invalid_index = [&](auto access, const char *field, int index, int size) {
    try {
      access(index);
      FAIL() << "Expected std::out_of_range for " << field << " index " << index;
    } catch (const std::out_of_range &e) {
      EXPECT_EQ(e.what(), make_string("Invalid ", field, " index ", index,
                                     " in argument \"bounds\"; expected 0 <= index < ", size, "."));
    }
  };
  for (int index : {-1, 0}) {
    expect_invalid_index([&](int i) { arg.ints(i); }, "int value", index, 0);
    expect_invalid_index([&](int i) { arg.floats(i); }, "float value", index, 0);
    expect_invalid_index([&](int i) { arg.bools(i); }, "bool value", index, 0);
    expect_invalid_index([&](int i) { arg.strings(i); }, "string value", index, 0);
    expect_invalid_index([&](int i) { arg.extra_args(i); }, "extra argument", index, 0);
  }
  arg.add_ints(42);
  arg.add_floats(1.5f);
  arg.add_bools(true);
  arg.add_strings("value");
  arg.add_extra_args().set_name("child");
  EXPECT_EQ(arg.ints(0), 42);
  EXPECT_EQ(arg.floats(0), 1.5f);
  EXPECT_TRUE(arg.bools(0));
  EXPECT_EQ(arg.strings(0), "value");
  EXPECT_EQ(arg.extra_args(0).name(), "child");
  for (int index : {-1, 1}) {
    expect_invalid_index([&](int i) { arg.ints(i); }, "int value", index, 1);
    expect_invalid_index([&](int i) { arg.floats(i); }, "float value", index, 1);
    expect_invalid_index([&](int i) { arg.bools(i); }, "bool value", index, 1);
    expect_invalid_index([&](int i) { arg.strings(i); }, "string value", index, 1);
    expect_invalid_index([&](int i) { arg.extra_args(i); }, "extra argument", index, 1);
  }
}

#ifdef DALI_BUILD_PROTO3
namespace {
dali_proto::Argument SerializeFeature(TFUtil::FeatureType type, bool has_shape) {
  TFUtil::Feature::Value value = {};
  value.int64 = 42;
  value.float32 = 1.5f;
  value.str = "default";
  auto feature = has_shape ? TFUtil::Feature({2, 3}, type, value)
                           : TFUtil::Feature(type, value);
  dali_proto::Argument proto;
  DaliProtoPriv arg(&proto);
  Argument::Store("feature", feature)->SerializeToProtobuf(&arg);
  return proto;
}
}  // namespace

TEST(FeatureDeserializationTest, MissingExtraArguments) {
  for (int count = 0; count < 4; count++) {
    auto proto = SerializeFeature(TFUtil::int64, false);
    while (proto.extra_args_size() > count)
      proto.mutable_extra_args()->RemoveLast();
    SCOPED_TRACE(count);
    EXPECT_THROW(DeserializeProtobuf(DaliProtoPriv(&proto)), std::out_of_range);
  }
}

TEST(FeatureDeserializationTest, MissingValues) {
  for (auto type : {TFUtil::int64, TFUtil::float32, TFUtil::string}) {
    // The shape may be empty; the type, has_shape and default value may not.
    for (int child : {0, 1, 3}) {
      SCOPED_TRACE(::testing::Message() << "type=" << type << ", child=" << child);
      auto proto = SerializeFeature(type, false);
      auto *field = proto.mutable_extra_args(child);
      field->clear_ints();
      field->clear_floats();
      field->clear_bools();
      field->clear_strings();
      EXPECT_THROW(DeserializeProtobuf(DaliProtoPriv(&proto)), std::out_of_range);
    }
  }
}

TEST(FeatureDeserializationTest, RoundTrip) {
  for (auto type : {TFUtil::int64, TFUtil::float32, TFUtil::string}) {
    for (bool has_shape : {false, true}) {
      auto proto = SerializeFeature(type, has_shape);
      auto restored = DeserializeProtobuf(DaliProtoPriv(&proto));
      const auto &feature = restored->Get<TFUtil::Feature>();
      EXPECT_EQ(feature.GetType(), type);
      EXPECT_EQ(feature.HasShape(), has_shape);
      EXPECT_EQ(feature.Shape(), has_shape ? (std::vector<Index>{2, 3}) : std::vector<Index>{});
      switch (type) {
        case TFUtil::int64:
          EXPECT_EQ(feature.GetValue().int64, 42);
          break;
        case TFUtil::float32:
          EXPECT_EQ(feature.GetValue().float32, 1.5f);
          break;
        case TFUtil::string:
          EXPECT_EQ(feature.GetValue().str, "default");
          break;
      }
    }
  }
}
#endif  // DALI_BUILD_PROTO3

}  // namespace dali
