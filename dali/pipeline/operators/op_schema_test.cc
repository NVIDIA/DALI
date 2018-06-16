// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/op_schema.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/test/dali_test.h"

namespace dali {

DALI_SCHEMA(Dummy1)
  .NumInput(1)
  .NumOutput(1);

class OpSchemaTest : public DALITest {
 public:
  inline void SetUp() {}
  inline void TearDown() {}
};

TEST(OpSchemaTest, SimpleTest) {
  auto schema = SchemaRegistry::GetSchema("Dummy1");

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
  auto schema = SchemaRegistry::GetSchema("Dummy2");

  ASSERT_EQ(schema.CalculateOutputs(spec), 2);
}

DALI_SCHEMA(Dummy3)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("foo", "foo", 1.5f);

TEST(OpSchemaTest, OptionalArgumentDefaultValue) {
  auto spec = OpSpec("Dummy3");
  auto schema = SchemaRegistry::GetSchema("Dummy3");

  ASSERT_TRUE(schema.OptionalArgumentExists("foo"));
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("foo"), 1.5f);
}

DALI_SCHEMA(Dummy4)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("bar", "var", 17.f)
  .AddParent("Dummy3");

TEST(OpSchemaTest, OptionalArgumentDefaultValueInheritance) {
  auto spec = OpSpec("Dummy4");
  auto schema = SchemaRegistry::GetSchema("Dummy4");

  ASSERT_TRUE(schema.OptionalArgumentExists("foo"));
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("foo"), 1.5f);
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("bar"), 17);
}

DALI_SCHEMA(Dummy5)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("baz", "baz", 2.f)
  .AddParent("Dummy4");

TEST(OpSchemaTest, OptionalArgumentDefaultValueMultipleInheritance) {
  auto spec = OpSpec("Dummy5");
  auto schema = SchemaRegistry::GetSchema("Dummy5");

  ASSERT_TRUE(schema.OptionalArgumentExists("foo"));
  ASSERT_TRUE(schema.OptionalArgumentExists("bar"));
  ASSERT_TRUE(schema.OptionalArgumentExists("baz"));

  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("foo"), 1.5f);
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("bar"), 17.f);
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("baz"), 2.f);
}

DALI_SCHEMA(Dummy6)
  .NumInput(1).NumOutput(1)
  .AddOptionalArg("dummy", "dummy", 1.85f);

DALI_SCHEMA(Dummy7)
  .NumInput(1).NumOutput(1)
  .AddParent("Dummy5")
  .AddParent("Dummy6");

TEST(OpSchemaTest, OptionalArgumentDefaultValueMultipleParent) {
  auto schema = SchemaRegistry::GetSchema("Dummy7");

  ASSERT_TRUE(schema.OptionalArgumentExists("foo"));
  ASSERT_TRUE(schema.OptionalArgumentExists("bar"));
  ASSERT_TRUE(schema.OptionalArgumentExists("baz"));
  ASSERT_TRUE(schema.OptionalArgumentExists("dummy"));

  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("foo"), 1.5f);
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("bar"), 17.f);
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("baz"), 2.f);
  ASSERT_EQ(schema.GetDefaultValueForOptionalArgument<float>("dummy"), 1.85f);
}

}  // namespace dali
