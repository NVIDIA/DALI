// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  auto spec = OpSpec("Dummy2").AddInput("in", StorageDevice::CPU);
  auto &schema = SchemaRegistry::GetSchema("Dummy2");

  ASSERT_EQ(schema.CalculateOutputs(spec), 2);
}

DALI_SCHEMA(DummForwardRefParent)
  .AddParent("Dummy3")  // not yet defined
  .AddOptionalArg("foo", "foo", 2);

TEST(OpSchemaTest, InitalizationOrder) {
  auto spec = OpSpec("DummForwardRefParent");
  auto &schema = SchemaRegistry::GetSchema("DummForwardRefParent");
  EXPECT_EQ(&spec.GetSchema(), &schema);
  EXPECT_EQ(schema.GetDefaultValueForArgument<int>("foo"), 2);
  EXPECT_NO_THROW(
    EXPECT_EQ(spec.GetArgument<int>("foo"), 2);
  );  // NOLINT
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
  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::invalid_argument);

  ASSERT_THROW(schema.HasArgumentDefaultValue("don't have this one"), invalid_key);
}

DALI_SCHEMA(Dummy4)
  .NumInput(1).NumOutput(1)
  .AddParent("Dummy3")
  .AddOptionalArg("bar", "var", 17.f)
  .AddOptionalArg("foo", "foo", 2)  // shadow an argument from a parent
  .AddOptionalArg<bool>("no_default2", "argument without default", nullptr);

TEST(OpSchemaTest, OptionalArgumentDefaultValueInheritance) {
  auto spec = OpSpec("Dummy4");
  auto &schema = SchemaRegistry::GetSchema("Dummy4");

  ASSERT_TRUE(schema.HasOptionalArgument("foo"));
  ASSERT_EQ(schema.GetDefaultValueForArgument<int>("foo"), 2);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("bar"), 17);

  ASSERT_TRUE(schema.HasOptionalArgument("no_default"));
  ASSERT_TRUE(schema.HasOptionalArgument("no_default2"));

  ASSERT_TRUE(schema.HasArgumentDefaultValue("foo"));
  ASSERT_TRUE(schema.HasArgumentDefaultValue("bar"));

  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default"));
  ASSERT_FALSE(schema.HasArgumentDefaultValue("no_default2"));

  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::invalid_argument);
  ASSERT_THROW(schema.GetDefaultValueForArgument<bool>("no_default2"), std::invalid_argument);
}

DALI_SCHEMA(Circular1)
  .AddParent("Circular2");

DALI_SCHEMA(Circular2)
  .AddParent("Circular1");

TEST(OpSchemaTest, CircularInheritance) {
  EXPECT_THROW(SchemaRegistry::GetSchema("Circular1").HasArgument("foo"), std::logic_error);
  EXPECT_THROW(SchemaRegistry::GetSchema("Circular2").HasArgument("foo"), std::logic_error);
}

DALI_SCHEMA(Dummy5)
  .DocStr("Foo")
  .AddParent("Dummy4")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("foo", "foo", 1.50f)  // shadow an argument from a parent
  .AddOptionalArg("baz", "baz", 2.f);

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

  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::invalid_argument);
  ASSERT_THROW(schema.GetDefaultValueForArgument<bool>("no_default2"), std::invalid_argument);
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

  ASSERT_THROW(schema.GetDefaultValueForArgument<int>("no_default"), std::invalid_argument);
  ASSERT_THROW(schema.GetDefaultValueForArgument<bool>("no_default2"), std::invalid_argument);
  ASSERT_THROW(schema.GetDefaultValueForArgument<float>("no_default3"), std::invalid_argument);
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
              .AddInput("in", StorageDevice::CPU)
              .AddArg("extra_out", 3);
  auto spec2 = OpSpec("Dummy8")
              .AddInput("in", StorageDevice::CPU)
              .AddArg("extra_out", 0);
  auto &schema = SchemaRegistry::GetSchema("Dummy8");

  ASSERT_EQ(schema.CalculateOutputs(spec), 1);
  ASSERT_EQ(schema.CalculateAdditionalOutputs(spec), 3);

  ASSERT_EQ(schema.CalculateOutputs(spec2), 1);
  ASSERT_EQ(schema.CalculateAdditionalOutputs(spec2), 0);
}

DALI_SCHEMA(DummyWithHiddenArg)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("dummy", "dummy", 1.85f)
  .AddOptionalArg<float>("_dummy", "hidden argument", 2.f);

DALI_SCHEMA(DummyWithHiddenArg2)
  .NumInput(1).NumOutput(1)
  .AddOptionalTypeArg("_dtype", "hidden dtype arg", DALI_INT16)
  .AddParent("DummyWithHiddenArg");

TEST(OpSchemaTest, OptionalHiddenArg) {
  auto &schema = SchemaRegistry::GetSchema("DummyWithHiddenArg");

  ASSERT_TRUE(schema.HasOptionalArgument("dummy"));
  ASSERT_TRUE(schema.HasOptionalArgument("_dummy"));

  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("dummy"), 1.85f);
  ASSERT_EQ(schema.GetDefaultValueForArgument<float>("_dummy"), 2.f);

  auto names = schema.GetArgumentNames();
  ASSERT_NE(std::find(names.begin(), names.end(), "dummy"), names.end());
  ASSERT_EQ(std::find(names.begin(), names.end(), "_dummy"), names.end());

  auto &schema2 = SchemaRegistry::GetSchema("DummyWithHiddenArg2");

  ASSERT_TRUE(schema2.HasOptionalArgument("dummy"));
  ASSERT_TRUE(schema2.HasOptionalArgument("_dummy"));
  ASSERT_TRUE(schema2.HasOptionalArgument("_dtype"));

  ASSERT_EQ(schema2.GetDefaultValueForArgument<float>("dummy"), 1.85f);
  ASSERT_EQ(schema2.GetDefaultValueForArgument<float>("_dummy"), 2.f);
  ASSERT_EQ(schema2.GetDefaultValueForArgument<DALIDataType>("_dtype"), DALI_INT16);

  auto names2 = schema2.GetArgumentNames();
  ASSERT_NE(std::find(names2.begin(), names2.end(), "dummy"), names2.end());
  ASSERT_EQ(std::find(names2.begin(), names2.end(), "_dummy"), names2.end());
  ASSERT_EQ(std::find(names2.begin(), names2.end(), "_dtype"), names2.end());
}

}  // namespace dali
