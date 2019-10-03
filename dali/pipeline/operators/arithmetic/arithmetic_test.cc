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
// #include <dbg.h>

#include "dali/core/static_switch.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/arithmetic/arithmetic.h"
#include "dali/pipeline/operators/arithmetic/arithmetic_meta.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_operator_test.h"

namespace dali {

TEST(ArithmeticOps, ExpressionTree) {
  std::string expr = "mul(&0 &1)";
  auto result = ParseExpressionString(expr);
  auto &result_ref = *result;
  ASSERT_EQ(result_ref.GetNodeType(), NodeType::Function);
  ASSERT_EQ(result_ref.GetSubexpressionCount(), 2);
  ASSERT_EQ(result_ref.GetOp(), "mul");
  ASSERT_EQ(result_ref[0].GetNodeType(), NodeType::Tensor);
  ASSERT_EQ(result_ref[1].GetNodeType(), NodeType::Tensor);
  ASSERT_EQ(dynamic_cast<ExprTensor&>(result_ref[0]).GetMappedInput(), 0);
  ASSERT_EQ(dynamic_cast<ExprTensor&>(result_ref[1]).GetMappedInput(), 1);
}

TEST(ArithmeticOps, ExpressionTreeComplex) {
  std::string expr = "div(sub(&42 &2) &1)";
  auto result = ParseExpressionString(expr);
  auto &result_ref = *result;
  ASSERT_EQ(result_ref.GetNodeType(), NodeType::Function);
  ASSERT_EQ(result_ref.GetSubexpressionCount(), 2);
  EXPECT_EQ(result_ref.GetOp(), "div");
  ASSERT_EQ(result_ref[0].GetNodeType(), NodeType::Function);
  ASSERT_EQ(result_ref[0].GetSubexpressionCount(), 2);
  EXPECT_EQ(result_ref[0].GetOp(), "sub");
  ASSERT_EQ(result_ref[0][0].GetNodeType(), NodeType::Tensor);
  EXPECT_EQ(dynamic_cast<ExprTensor&>(result_ref[0][0]).GetMappedInput(), 42);
  ASSERT_EQ(result_ref[0][1].GetNodeType(), NodeType::Tensor);
  EXPECT_EQ(dynamic_cast<ExprTensor&>(result_ref[0][1]).GetMappedInput(), 2);
  ASSERT_EQ(result_ref[1].GetNodeType(), NodeType::Tensor);
  EXPECT_EQ(dynamic_cast<ExprTensor&>(result_ref[1]).GetMappedInput(), 1);
}

TEST(ArithmeticOps, TreePropagation) {
  std::string expr_str = "div(sub(&0 &1) $2:int8)";
  auto expr = ParseExpressionString(expr_str);
  auto &expr_ref = *expr;
  HostWorkspace ws;
  std::shared_ptr<TensorVector<CPUBackend>> in[3];
  for (auto &ptr : in) {
    ptr = std::make_shared<TensorVector<CPUBackend>>();
    ptr->Resize({{1}});
  }
  in[0]->set_type(TypeInfo::Create<uint8_t>());
  in[1]->set_type(TypeInfo::Create<int16_t>());
  in[2]->set_type(TypeInfo::Create<int32_t>());
  ws.AddInput(in[0]);
  ws.AddInput(in[1]);
  ws.AddInput(in[2]);

  auto result_type = PropagateTypes<CPUBackend>(expr_ref, ws);
  auto result_shape = PropagateShapes<CPUBackend>(expr_ref, ws);
  // dbg(result_type);
  // dbg(result_shape);
  // dbg(expr_ref.GetNodeDesc());
  // dbg(expr_ref.GetOutputDesc());
  // dbg(expr_ref.GetNodeDesc());
  // dbg(expr_ref.GetOutputDesc());
  // dbg(expr_ref[0].GetNodeDesc());
  // dbg(expr_ref[0].GetOutputDesc());
  // dbg(expr_ref[1].GetNodeDesc());
  // dbg(expr_ref[1].GetOutputDesc());
  // auto &result_ref = *result;
  // ASSERT_EQ(result_ref.GetNodeType(), NodeType::Function);
  // ASSERT_EQ(result_ref.GetSubexpressionCount(), 2);
  // EXPECT_EQ(result_ref.GetOp(), "div");
  // ASSERT_EQ(result_ref[0].GetNodeType(), NodeType::Function);
  // ASSERT_EQ(result_ref[0].GetSubexpressionCount(), 2);
  // EXPECT_EQ(result_ref[0].GetOp(), "sub");
  // ASSERT_EQ(result_ref[0][0].GetNodeType(), NodeType::Tensor);
  // EXPECT_EQ(dynamic_cast<ExprTensor&>(result_ref[0][0]).GetMappedInput(), 42);
  // ASSERT_EQ(result_ref[0][1].GetNodeType(), NodeType::Tensor);
  // EXPECT_EQ(dynamic_cast<ExprTensor&>(result_ref[0][1]).GetMappedInput(), 2);
  // ASSERT_EQ(result_ref[1].GetNodeType(), NodeType::Tensor);
  // EXPECT_EQ(dynamic_cast<ExprTensor&>(result_ref[1]).GetMappedInput(), 1);
}



// namespace {
//   bool operator==(const SampleExtent &l, const SampleExtent &r) {
//     return l.extent_idx == r.extent_idx && l.extent_size == l.extent_size &&
//            l.sample_idx == r.sample_idx && l.task_idx == r.task_idx;
//   }
// }  // namespace

TEST(ArithmeticOps, GetTiledCover) {
  kernels::TensorListShape<> shape0({{150}, {50}, {150}, {30}});
  auto result0 = GetTiledCover(shape0, 50, 2);
  std::vector<TileDesc> cover0 = {{0, 0, 0, 50}, {0, 1, 0, 50}, {0, 2, 0, 50}, {1, 0, 0, 50},
                                      {2, 0, 1, 50}, {2, 1, 1, 50}, {2, 2, 1, 50}, {3, 0, 1, 30}};
  // dbg(std::get<0>(result0));
  // EXPECT_EQ(cover0, result0);

  kernels::TensorListShape<> shape1({{42}, {75}, {42}, {121}});
  auto result1 = GetTiledCover(shape1, 50, 2);
  std::vector<TileDesc> cover1 = {{0, 0, 0, 42}, {1, 0, 0, 50}, {1, 1, 0, 25}, {2, 0, 0, 42},
                                      {3, 0, 1, 50}, {3, 1, 1, 50}, {3, 2, 1, 21}};
  // dbg(std::get<0>(result1));
  // EXPECT_EQ(cover1, result1);

  auto result2 = GetCover(shape0, 2);
  std::vector<TileDesc> cover2 = {{0, 0, 0, 150}, {1, 0, 0, 50}, {2, 0, 1, 150}, {3, 0, 1, 30}};
  // dbg(std::get<0>(result2));
  //  EXPECT_EQ(cover2, result2);

  auto result3 = GetCover(shape1, 2);
  std::vector<TileDesc> cover3 = {{0, 0, 0, 42}, {1, 0, 0, 75}, {2, 0, 1, 42}, {3, 0, 1, 121}};
  // dbg(std::get<0>(result3));
  //  EXPECT_EQ(cover3, result3);
}

template <typename T>
using bin_op_pointer = T (*)(T, T);

template <typename Backend, typename T>
class BinaryArithmeticOpTest
    : public ::testing::TestWithParam<std::tuple<std::string, bin_op_pointer<T>>> {
 protected:
  static constexpr int batch_size = 16;
  static constexpr int num_threads = 4;
  static constexpr int tensor_elements = 16;

  void TestFunction() {
    auto backend = testing::detail::BackendStringName<Backend>();

    auto param = this->GetParam();
    auto expression_desc = std::get<0>(param) + "(&0 &1)";
    auto result_fun = std::get<1>(param);
    Pipeline pipe(batch_size, num_threads, 0);

    pipe.AddExternalInput("data0");
    pipe.AddExternalInput("data1");

    pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                         .AddArg("device", backend)
                         .AddArg("expression_desc", expression_desc)
                         .AddInput("data0", backend)
                         .AddInput("data1", backend)
                         .AddOutput("result", backend),
                     std::get<0>(param));

    vector<std::pair<string, string>> outputs = {{"result", backend}};
    pipe.Build(outputs);

    TensorList<CPUBackend> batch[2];
    kernels::TensorListShape<> shape{{32000}, {2345}, {212}, {1}, {100}, {6400}, {8000}, {323}, {32000}, {2345}, {212}, {1}, {100}, {6400}, {8000}, {323}};
    for (auto &b : batch) {
      // b.Resize(kernels::uniform_list_shape(batch_size, {tensor_elements}));
      b.Resize(shape);
      b.set_type(TypeInfo::Create<T>());
      for (int i = 0; i < shape.num_samples(); i++) {
        auto *t = b.template mutable_tensor<T>(i);
        for (int j = 0; j < shape[i].num_elements(); j++) {
          t[j] = GenerateData<T>(i, j);
        }
      }
    }

    pipe.SetExternalInput("data0", batch[0]);
    pipe.SetExternalInput("data1", batch[1]);

    pipe.RunCPU();
    pipe.RunGPU();
    DeviceWorkspace ws;
    pipe.Outputs(&ws);
    auto *result = ws.OutputRef<Backend>(0).template data<T>();
    vector<T> result_cpu(shape.num_elements());
    MemCopy(result_cpu.data(), result, shape.num_elements() * sizeof(T));

    // for (int i = 0; i < shape.num_elements(); i++) {
    //   EXPECT_EQ(result_cpu[i],
    //             result_fun(batch[0].template data<T>()[i], batch[1].template data<T>()[i])) << " difference at position " << i;
    // }

    int64_t offset = 0;
    for (int i = 0; i < shape.num_samples(); i++) {
      for (int j = 0; j < shape[i].num_elements(); j++) {
        ASSERT_EQ(result_cpu[offset + j],
                  result_fun(batch[0].template tensor<T>(i)[j], batch[1].template tensor<T>(i)[j]))
                   << " difference at sample: " << i << ", element: " << j;
      }
      offset += shape[i].num_elements();
    }
  }

  template <typename S>
  std::enable_if_t<std::is_integral<S>::value, S> GenerateData(int sample, int element) {
    static std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(-1024, 1024);
    auto result = dis(gen);
    return result == 0 ? 1 : result;
  }

  template <typename S>
  std::enable_if_t<!std::is_integral<S>::value, S> GenerateData(int sample, int element) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1024, 1024);
    auto result = dis(gen);
    return result == 0.f ? 1.f : result;
  }
};

template <typename T>
std::vector<std::tuple<std::string, bin_op_pointer<T>>> getOpNameRef() {
  return std::vector<std::tuple<std::string, bin_op_pointer<T>>>{
      std::make_tuple("add", [](T l, T r) -> T { return l + r; }),
      std::make_tuple("sub", [](T l, T r) -> T { return l - r; }),
      std::make_tuple("mul", [](T l, T r) -> T { return l * r; }),
      std::make_tuple("div", [](T l, T r) -> T { return l / r; }),
      std::make_tuple("mod",
                      std::is_integral<T>::value
                      ? [](T l, T r) -> T { return std::fmod(l, r); }
                      : [](T l, T r) -> T { return std::remainder(l, r); }),
  };
}

// We need to pass a type name to macro TEST_P, so we use an alias
// We will test for both backends with int32_t and float.
using BinaryArithmeticOpCPUint32 = BinaryArithmeticOpTest<CPUBackend, int32_t>;
using BinaryArithmeticOpCPUfloat = BinaryArithmeticOpTest<CPUBackend, float>;
using BinaryArithmeticOpGPUint32 = BinaryArithmeticOpTest<GPUBackend, int32_t>;
using BinaryArithmeticOpGPUfloat = BinaryArithmeticOpTest<GPUBackend, float>;

// Create the tests
TEST_P(BinaryArithmeticOpCPUint32, SimplePipeline) {
  TestFunction();
}
TEST_P(BinaryArithmeticOpCPUfloat, SimplePipeline) {
  TestFunction();
}
TEST_P(BinaryArithmeticOpGPUint32, SimplePipeline) {
  TestFunction();
}
TEST_P(BinaryArithmeticOpGPUfloat, SimplePipeline) {
  TestFunction();
}

// Pass the values to tests suites
INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteCPUint32, BinaryArithmeticOpCPUint32,
                         ::testing::ValuesIn(getOpNameRef<int32_t>()));

INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteCPUfloat, BinaryArithmeticOpCPUfloat,
                         ::testing::ValuesIn(getOpNameRef<float>()));

INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteGPUint32, BinaryArithmeticOpGPUint32,
                         ::testing::ValuesIn(getOpNameRef<int32_t>()));

INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteGPUfloat, BinaryArithmeticOpGPUfloat,
                         ::testing::ValuesIn(getOpNameRef<float>()));

TEST(ArithmeticOps, GenericPipeline) {
  constexpr int batch_size = 16;
  constexpr int num_threads = 4;
  constexpr int tensor_elements = 16;
  Pipeline pipe(batch_size, num_threads, 0);

  pipe.AddExternalInput("data0");
  pipe.AddExternalInput("data1");

  pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                       .AddArg("device", "cpu")
                       .AddArg("expression_desc", "add(&0 &1)")
                       .AddInput("data0", "cpu")
                       .AddInput("data1", "cpu")
                       .AddOutput("result", "cpu"),
                   "arithm_cpu");

  pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                       .AddArg("device", "gpu")
                       .AddArg("expression_desc", "mul(&0 &1)")
                       .AddInput("result", "gpu")
                       .AddInput("data1", "gpu")
                       .AddOutput("result2", "gpu"),
                   "arithm_gpu");

  vector<std::pair<string, string>> outputs = {{"result", "cpu"}, {"result2", "gpu"}};

  pipe.Build(outputs);
  pipe.SaveGraphToDotFile("test.dot");

  TensorList<CPUBackend> batch;
  batch.Resize(kernels::uniform_list_shape(batch_size, {tensor_elements}));
  batch.set_type(TypeInfo::Create<int32_t>());
  for (int i = 0; i < batch_size; i++) {
    auto *t = batch.mutable_tensor<int32_t>(i);
    for (int j = 0; j < tensor_elements; j++) {
      t[j] = i * tensor_elements + j;
    }
  }

  pipe.SetExternalInput("data0", batch);
  pipe.SetExternalInput("data1", batch);
  pipe.RunCPU();
  pipe.RunGPU();
  DeviceWorkspace ws;
  pipe.Outputs(&ws);
  auto *result = ws.OutputRef<CPUBackend>(0).data<int32_t>();
  auto *result2 = ws.OutputRef<GPUBackend>(1).data<int32_t>();
  vector<int32_t> result2_cpu(batch_size * tensor_elements);

  MemCopy(result2_cpu.data(), result2, batch_size * tensor_elements * sizeof(int));
  for (int i = 0; i < batch_size * tensor_elements; i++) {
    EXPECT_EQ(result[i], i + i);
    EXPECT_EQ(result2_cpu[i], i * (i + i));
  }
}

}  // namespace dali
