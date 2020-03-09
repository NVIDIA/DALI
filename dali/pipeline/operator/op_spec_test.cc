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

#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/test/dali_test.h"

using namespace std::string_literals;  // NOLINT(build/namespaces)

namespace dali {

DALI_SCHEMA(DummyOpForSpecTest)
  .NumInput(0).NumOutput(0)
  .AddArg("required", "required argument", DALIDataType::DALI_INT32)
  .AddOptionalArg("default", "argument with default", 11)
  .AddOptionalArg<int>("no_default", "argument without default", nullptr)
  .AddArg("required_vec", "required argument", DALIDataType::DALI_INT_VEC)
  .AddOptionalArg("default_vec", "argument with default vec", std::vector<int32_t>{0, 1})
  .AddOptionalArg<std::vector<int>>("no_default_vec", "argument without default", nullptr)
  .AddArg("required_tensor", "required argument", DALIDataType::DALI_INT32, true)
  .AddOptionalArg("default_tensor", "argument with default", 11, true)
  .AddOptionalArg<int>("no_default_tensor", "argument without default", nullptr, true);

TEST(OpSpecTest, GetArgumentTensorSet) {
  // Check how required and optional arguments handle Argument Inputs
  // Should work only with [Try]GetArgument;
  // [Try]GetRepeatedArgument does not handle Argument Inputs
  for (const auto &arg_name : {"required_tensor"s, "default_tensor"s, "no_default_tensor"s}) {
    ArgumentWorkspace ws0;
    auto tv = std::make_shared<TensorVector<CPUBackend>>(2);
    tv->Resize(uniform_list_shape(2, {1}));
    tv->set_type(TypeInfo::Create<int32_t>());
    for (int i = 0; i < 2; i++) {
      tv->tensor_handle(i)->mutable_data<int32_t>()[0] = 42 + i;
    }
    ws0.AddArgumentInput(arg_name, tv);
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2)
        .AddArgumentInput(arg_name, "<not_used>");
    ASSERT_EQ(spec0.GetArgument<int32_t>(arg_name, &ws0, 0), 42);
    ASSERT_EQ(spec0.GetArgument<int32_t>(arg_name, &ws0, 1), 43);
    int result = 0;
    ASSERT_TRUE(spec0.TryGetArgument<int32_t>(result, arg_name, &ws0, 0));
    ASSERT_EQ(result, 42);
    ASSERT_TRUE(spec0.TryGetArgument<int32_t>(result, arg_name, &ws0, 1));
    ASSERT_EQ(result, 43);
    ASSERT_THROW(spec0.GetArgument<float>(arg_name, &ws0, 0), std::runtime_error);
    float tmp = 0.f;
    ASSERT_FALSE(spec0.TryGetArgument<float>(tmp, arg_name, &ws0, 0));

    ArgumentWorkspace ws1;
    auto spec1 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2);
    // If we have a default optional argument, we will just return its value
    if (arg_name != "default_tensor"s) {
      ASSERT_THROW(spec1.GetArgument<int>(arg_name, &ws1, 0), std::runtime_error);
      ASSERT_THROW(spec1.GetArgument<int>(arg_name, &ws1, 1), std::runtime_error);
      int result = 0;
      ASSERT_FALSE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 0));
      ASSERT_FALSE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 1));
    } else {
      ASSERT_EQ(spec1.GetArgument<int>(arg_name, &ws1, 0), 11);
      ASSERT_EQ(spec1.GetArgument<int>(arg_name, &ws1, 1), 11);
      int result = 0;
      ASSERT_TRUE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 0));
      ASSERT_EQ(result, 11);
      result = 0;
      ASSERT_TRUE(spec1.TryGetArgument<int>(result, arg_name, &ws1, 1));
      ASSERT_EQ(result, 11);
    }
  }
}

TEST(OpSpecTest, GetArgumentValue) {
  for (const auto &arg_name : {"required"s, "default"s, "no_default"s,
                               "required_tensor"s, "default_tensor"s, "no_default_tensor"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2)
        .AddArg(arg_name, 42);
    ASSERT_EQ(spec0.GetArgument<int>(arg_name, &ws), 42);
    int result = 0;
    ASSERT_TRUE(spec0.TryGetArgument(result, arg_name, &ws));
    ASSERT_EQ(result, 42);

    ASSERT_THROW(spec0.GetArgument<float>(arg_name, &ws), std::runtime_error);
    float tmp = 0.f;
    ASSERT_FALSE(spec0.TryGetArgument(tmp, arg_name, &ws));
  }

  for (const auto &arg_name : {"required"s, "no_default"s,
                               "required_tensor"s, "no_default_tensor"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2);
    ASSERT_THROW(spec0.GetArgument<int>(arg_name, &ws), std::runtime_error);
    int result = 0;
    ASSERT_FALSE(spec0.TryGetArgument(result, arg_name, &ws));

    ASSERT_THROW(spec0.GetArgument<float>(arg_name, &ws), std::runtime_error);
    float tmp = 0.f;
    ASSERT_FALSE(spec0.TryGetArgument(tmp, arg_name, &ws));
  }

  for (const auto &arg_name : {"default"s, "default_tensor"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2);
    ASSERT_EQ(spec0.GetArgument<int>(arg_name, &ws), 11);

    int result = 0;
    ASSERT_TRUE(spec0.TryGetArgument(result, arg_name, &ws));
    ASSERT_EQ(result, 11);

    ASSERT_THROW(spec0.GetArgument<float>(arg_name, &ws), std::runtime_error);
    float tmp = 0.f;
    ASSERT_FALSE(spec0.TryGetArgument(tmp, arg_name, &ws));
  }
}

TEST(OpSpecTest, GetArgumentVec) {
  for (const auto &arg_name : {"required_vec"s, "default_vec"s, "no_default_vec"s}) {
    ArgumentWorkspace ws;
    auto value = std::vector<int32_t>{42, 43};

    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2)
        .AddArg(arg_name, value);

    ASSERT_EQ(spec0.GetRepeatedArgument<int32_t>(arg_name), value);
    std::vector<int32_t> result;
    ASSERT_TRUE(spec0.TryGetRepeatedArgument(result, arg_name));
    ASSERT_EQ(result, value);
  }

  for (const auto &arg_name : {"required_vec"s, "no_default_vec"s}) {
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2);

    ASSERT_THROW(spec0.GetRepeatedArgument<int32_t>(arg_name), std::runtime_error);
    std::vector<int32_t> result;
    ASSERT_FALSE(spec0.TryGetRepeatedArgument(result, arg_name));

    ASSERT_THROW(spec0.GetRepeatedArgument<float>(arg_name), std::runtime_error);
    std::vector<float> tmp;
    ASSERT_FALSE(spec0.TryGetRepeatedArgument(tmp, arg_name));
  }

  {
    auto arg_name = "default_vec"s;
    ArgumentWorkspace ws;
    auto spec0 = OpSpec("DummyOpForSpecTest")
        .AddArg("batch_size", 2);
    auto default_val = std::vector<int32_t>{0, 1};
    ASSERT_EQ(spec0.GetRepeatedArgument<int32_t>(arg_name), default_val);
  }
}


TEST(OpSpecTest, GetArgumentNonExisting) {
  auto spec0 = OpSpec("DummyOpForSpecTest")
      .AddArg("batch_size", 2);
  ASSERT_THROW(spec0.GetArgument<int>("<no_such_argument>"), std::runtime_error);
  int result = 0;
  ASSERT_FALSE(spec0.TryGetArgument<int>(result, "<no_such_argument>"));


  ASSERT_THROW(spec0.GetRepeatedArgument<int>("<no_such_argument>"), std::runtime_error);
  std::vector<int> result_vec;
  ASSERT_FALSE(spec0.TryGetRepeatedArgument<int>(result_vec, "<no_such_argument>"));
}

class TestArgumentInput_Producer : public Operator<CPUBackend> {
 public:
  explicit TestArgumentInput_Producer(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    output_desc.resize(3);
    output_desc[0] = {uniform_list_shape(batch_size_, {1}), TypeTable::GetTypeInfo(DALI_INT32)};
    // Non-matching shapes
    output_desc[1] = {uniform_list_shape(batch_size_, {1}), TypeTable::GetTypeInfo(DALI_FLOAT)};
    output_desc[2] = {uniform_list_shape(batch_size_, {1, 2}), TypeTable::GetTypeInfo(DALI_INT32)};
    return true;
  }

  void RunImpl(HostWorkspace &ws) override {
    // Initialize all the data with a 0, 1, 2 .... sequence
    auto &out0 = ws.OutputRef<CPUBackend>(0);
    for (int i = 0; i < out0.shape().num_samples(); i++) {
      *out0[i].mutable_data<int>() = i;
    }

    auto &out1 = ws.OutputRef<CPUBackend>(1);
    for (int i = 0; i < out1.shape().num_samples(); i++) {
      *out1[i].mutable_data<float>() = i;
    }

    auto &out2 = ws.OutputRef<CPUBackend>(2);
    for (int i = 0; i < out2.shape().num_samples(); i++) {
      for (int j = 0; j < 2; j++) {
        out2[i].mutable_data<int>()[j] = i;
      }
    }
  }
};

DALI_REGISTER_OPERATOR(TestArgumentInput_Producer, TestArgumentInput_Producer, CPU);

DALI_SCHEMA(TestArgumentInput_Producer)
    .DocStr("TestArgumentInput_Producer")
    .NumInput(0)
    .NumOutput(3);

class TestArgumentInput_Consumer : public Operator<CPUBackend> {
 public:
  explicit TestArgumentInput_Consumer(const OpSpec &spec) : Operator<CPUBackend>(spec) {}

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    output_desc.resize(1);
    output_desc[0] = {uniform_list_shape(batch_size_, {1}), TypeInfo::Create<int>()};
    return true;
  }

  void RunImpl(HostWorkspace &ws) override {
    for (int i = 0; i < batch_size_; i++) {
      EXPECT_EQ(spec_.GetArgument<int>("arg0", &ws, i), i);
    }
    // Non-matching shapes (differnet than 1 scalar value per sample) should not work with
    // OpSpec::GetArgument()
    ASSERT_THROW(auto z = spec_.GetArgument<float>("arg2", &ws, 0), std::runtime_error);

    // They can be accessed as proper ArgumentInputs
    auto &ref_1 = ws.ArgumentInput("arg1");
    ASSERT_EQ(ref_1.shape().num_samples(), batch_size_);
    ASSERT_TRUE(is_uniform(ref_1.shape()));
    ASSERT_EQ(ref_1.shape()[0], TensorShape<>(1));
    for (int i = 0; i < ref_1.shape().num_samples(); i++) {
      EXPECT_EQ(ref_1[i].data<float>()[0], i);
    }

    auto &ref_2 = ws.ArgumentInput("arg2");
    ASSERT_EQ(ref_2.shape().num_samples(), batch_size_);
    ASSERT_TRUE(is_uniform(ref_2.shape()));
    ASSERT_EQ(ref_2.shape()[0], TensorShape<>(1, 2));
    for (int i = 0; i < ref_2.shape().num_samples(); i++) {
      for (int j = 0; j < 2; j++) {
        EXPECT_EQ(ref_2[i].data<int>()[j], i);
      }
    }
  }
};

DALI_REGISTER_OPERATOR(TestArgumentInput_Consumer, TestArgumentInput_Consumer, CPU);

DALI_SCHEMA(TestArgumentInput_Consumer)
    .DocStr("TestArgumentInput_Consumer")
    .NumInput(0)
    .NumOutput(1)
    .AddOptionalArg("arg0", "no-doc", 42, true)
    .AddOptionalArg("arg1", "no-doc", 42.f, true)
    .AddOptionalArg("arg2", "no-doc", 42, true)
    .AddOptionalArg("arg3", "no-doc", 42, true);

/*
 * This test is based on test operators implemented specifically for the purpose of testing
 * the access to argument inputs.
 *
 * The EXPECT_* and ASSERT_* macros are actually placed in the RunImpl of operator
 * accessing the data (TestArgumentInput_Consumer), and the different (valid and invalid)
 * arguments inputs are provided by a Operator: TestArgumentInput_Producer.
 */
TEST(ArgumentInputTest, OpSpecAccess) {
  Pipeline pipe(10, 4, 0);
  pipe.AddOperator(OpSpec("TestArgumentInput_Producer")
                       .AddArg("device", "cpu")
                       .AddOutput("support_arg0", "cpu")
                       .AddOutput("support_arg1", "cpu")
                       .AddOutput("support_arg2", "cpu"));

  pipe.AddOperator(OpSpec("TestArgumentInput_Consumer")
                       .AddArg("device", "cpu")
                       .AddArgumentInput("arg0", "support_arg0")
                       .AddArgumentInput("arg1", "support_arg1")
                       .AddArgumentInput("arg2", "support_arg2")
                       .AddOutput("I need to specify something", "cpu")
                       .AddArg("preserve", true));

  vector<std::pair<string, string>> outputs = {{"I need to specify something", "cpu"}};
  pipe.Build(outputs);

  pipe.RunCPU();
  pipe.RunGPU();

  DeviceWorkspace ws;
  pipe.Outputs(&ws);
}

}  // namespace dali
