// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/test/dali_test.h"

namespace dali {

DALI_SCHEMA(Dummy1)
  .NumInput(1)
  .NumOutput(1);

class OpSchemaTest : public DALITest {
 public:
  inline void SetUp() override {}
  inline void TearDown() override {}
};

TEST(OpSchemaTest, SimpleTest) {
  auto &schema = SchemaRegistry::GetSchema("Dummy1");

  ASSERT_EQ(schema.MaxNumInput(), 1);
  ASSERT_EQ(schema.MaxNumInput(), 1);
  ASSERT_EQ(schema.NumOutput(), 1);
}

DALI_SCHEMA(Dummy2)
  .NumInput(1, 2)
  .OutputFn([](const OpSpec& spec) {
    return spec.NumInput() * 2;
  });

TEST(OpSchemaTest, OutputFNTest) {
  auto spec = OpSpec("Dummy2").AddInput("in", "cpu");
  auto &schema = SchemaRegistry::GetSchema("Dummy2");

  ASSERT_EQ(schema.CalculateOutputs(spec), 2);
}

DALI_SCHEMA(Dummy3)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("foo", "foo", 1.5f)
  .AddOptionalArg<int>("no_default", "argument without default", nullptr);

TEST(OpSchemaTest, OptionalArgumentDefaultValue) {
  auto spec = OpSpec("Dummy3");
  auto &schema = SchemaRegistry::GetSchema("Dummy3");

  ASSERT_TRUE(schema.HasOptionalArgument("foo"));
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("foo"), 1.5f);

  ASSERT_TRUE(schema.HasOptionalArgument("no_default"));

  ASSERT_TRUE(schema.HasArgumentDefaultValue("foo"));

  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default"));
  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::runtime_error);

  ASSERT_THROW(schema.HasArgumentDefaultValue("don't have this one"), std::runtime_error);
}

DALI_SCHEMA(Dummy4)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("bar", "var", 17.f)
  .AddOptionalArg<bool>("no_default2", "argument without default", nullptr)
  .AddParent("Dummy3");

TEST(OpSchemaTest, OptionalArgumentDefaultValueInheritance) {
  auto spec = OpSpec("Dummy4");
  auto &schema = SchemaRegistry::GetSchema("Dummy4");

  ASSERT_TRUE(schema.HasOptionalArgument("foo"));
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("foo"), 1.5f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("bar"), 17);

  ASSERT_TRUE(schema.HasOptionalArgument("no_default"));
  ASSERT_TRUE(schema.HasOptionalArgument("no_default2"));

  ASSERT_TRUE(schema.HasArgumentDefaultValue("foo"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("bar"));

  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default"));
  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default2"));

  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::runtime_error);
  ASSERT_THROW(schema.GetDefaultValueForArgument<bool>("no_default2"), std::runtime_error);
}

DALI_SCHEMA(Dummy5)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("baz", "baz", 2.f)
  .AddParent("Dummy4");

TEST(OpSchemaTest, OptionalArgumentDefaultValueMultipleInheritance) {
  auto spec = OpSpec("Dummy5");
  auto &schema = SchemaRegistry::GetSchema("Dummy5");

  ASSERT_TRUE(schema.HasOptionalArgument("foo"));
  ASSERT_TRUE(schema.HasOptionalArgument("bar"));
  ASSERT_TRUE(schema.HasOptionalArgument("baz"));

  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("foo"), 1.5f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("bar"), 17.f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("baz"), 2.f);

  ASSERT_TRUE(schema.HasOptionalArgument("no_default"));
  ASSERT_TRUE(schema.HasOptionalArgument("no_default2"));

  ASSERT_TRUE(schema.HasArgumentDefaultValue("foo"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("bar"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("baz"));

  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default"));
  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default2"));

  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::runtime_error);
  ASSERT_THROW(schema.GetDefaultValueForArgument<bool>("no_default2"), std::runtime_error);
}

DALI_SCHEMA(Dummy6)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("dummy", "dummy", 1.85f)
  .AddOptionalArg<float>("no_default3", "argument without default", nullptr);

DALI_SCHEMA(Dummy7)
  .NumInput(1).NumOutput(1)
  .AddParent("Dummy5")
  .AddParent("Dummy6");

TEST(OpSchemaTest, OptionalArgumentDefaultValueMultipleParent) {
  auto &schema = SchemaRegistry::GetSchema("Dummy7");

  ASSERT_TRUE(schema.HasOptionalArgument("foo"));
  ASSERT_TRUE(schema.HasOptionalArgument("bar"));
  ASSERT_TRUE(schema.HasOptionalArgument("baz"));
  ASSERT_TRUE(schema.HasOptionalArgument("dummy"));

  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("foo"), 1.5f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("bar"), 17.f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("baz"), 2.f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("dummy"), 1.85f);

  ASSERT_TRUE(schema.HasOptionalArgument("no_default"));
  ASSERT_TRUE(schema.HasOptionalArgument("no_default2"));
  ASSERT_TRUE(schema.HasOptionalArgument("no_default3"));

  ASSERT_TRUE(schema.HasArgumentDefaultValue("foo"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("bar"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("baz"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("dummy"));

  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default"));
  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default2"));
  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default3"));

  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::runtime_error);
  ASSERT_THROW(schema.GetDefaultValueForArgument<bool>("no_default2"), std::runtime_error);
  ASSERT_THROW(schema.GetDefaultValueForArgument<float>("no_default3"), std::runtime_error);
}

DALI_SCHEMA(Dummy8)
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("extra_out", R"code()code", 1, true)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<int>("extra_out"));
  });

TEST(OpSchemaTest, AdditionalOutputFNTest) {
  auto spec = OpSpec("Dummy8")
              .AddInput("in", "cpu")
              .AddArg("extra_out", 3);
  auto spec2 = OpSpec("Dummy8")
              .AddInput("in", "cpu")
              .AddArg("extra_out", 0);
  auto &schema = SchemaRegistry::GetSchema("Dummy8");

  ASSERT_EQ(schema.CalculateOutputs(spec), 1);
  ASSERT_EQ(schema.CalculateAdditionalOutputs(spec), 3);

  ASSERT_EQ(schema.CalculateOutputs(spec2), 1);
  ASSERT_EQ(schema.CalculateAdditionalOutputs(spec2), 0);
}

}  // namespace dali
