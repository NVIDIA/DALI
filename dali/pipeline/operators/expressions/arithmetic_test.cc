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
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operators/expressions/arithmetic.h"
#include "dali/pipeline/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_operator_test.h"

namespace dali {

TEST(ArithmeticOpsTest, TreePropagation) {
  std::string expr_str = "div(sub(&0 &1) $2:int32)";
  auto expr = ParseExpressionString(expr_str);
  auto &expr_ref = *expr;
  HostWorkspace ws;
  std::shared_ptr<TensorVector<CPUBackend>> in[3];
  for (auto &ptr : in) {
    ptr = std::make_shared<TensorVector<CPUBackend>>();
    ptr->Resize({{1}, {2}});
  }
  in[0]->set_type(TypeInfo::Create<uint8_t>());
  in[1]->set_type(TypeInfo::Create<int16_t>());
  in[2]->set_type(TypeInfo::Create<int32_t>());
  in[0]->SetLayout(TensorLayout());
  in[1]->SetLayout(TensorLayout("HW"));
  ws.AddInput(in[0]);
  ws.AddInput(in[1]);

  auto result_type = PropagateTypes<CPUBackend>(expr_ref, ws);
  auto result_shape = PropagateShapes<CPUBackend>(expr_ref, ws);
  auto result_layout = GetCommonLayout<CPUBackend>(expr_ref, ws);
  auto expected_shpe = kernels::TensorListShape<>{{1}, {2}};
  EXPECT_EQ(result_type, DALIDataType::DALI_INT32);
  EXPECT_EQ(result_shape, expected_shpe);
  EXPECT_EQ(result_layout, "HW");
  EXPECT_EQ(expr_ref.GetNodeDesc(), "div:T:int32(FT:int16 CC:int32)");
  EXPECT_EQ(expr_ref.GetOutputDesc(), "FT:int32");
  EXPECT_EQ(expr_ref.GetNodeDesc(), "div:T:int32(FT:int16 CC:int32)");
  EXPECT_EQ(expr_ref.GetOutputDesc(), "FT:int32");
  auto &func = dynamic_cast<ExprFunc&>(expr_ref);
  EXPECT_EQ(func[0].GetNodeDesc(), "sub:T:int16(TT:uint8 TT:int16)");
  EXPECT_EQ(func[0].GetOutputDesc(), "FT:int16");
  EXPECT_EQ(func[1].GetNodeDesc(), "CC:int32");
  EXPECT_EQ(func[1].GetOutputDesc(), "CC:int32");
}

TEST(ArithmeticOpsTest, TreePropagationError) {
  std::string expr_str = "div(sub(&0 &1) &2)";
  auto expr = ParseExpressionString(expr_str);
  auto &expr_ref = *expr;
  HostWorkspace ws;
  std::shared_ptr<TensorVector<CPUBackend>> in[3];
  for (auto &ptr : in) {
    ptr = std::make_shared<TensorVector<CPUBackend>>();
    ptr->Resize({{1}, {2}});
  }
  in[0]->SetLayout(TensorLayout());
  in[1]->SetLayout(TensorLayout("HW"));
  in[2]->SetLayout(TensorLayout("DHW"));
  in[2]->Resize({{10}, {2}});
  ws.AddInput(in[0]);
  ws.AddInput(in[1]);
  ws.AddInput(in[2]);

  ASSERT_THROW(PropagateShapes<CPUBackend>(expr_ref, ws), std::runtime_error);
  ASSERT_THROW(GetCommonLayout<CPUBackend>(expr_ref, ws), std::runtime_error);
}


// namespace {

inline bool operator==(const TileDesc &l, const TileDesc &r) {
  return l.sample_idx == r.sample_idx && l.extent_idx == r.extent_idx &&
         l.extent_size == r.extent_size && l.tile_size == r.tile_size;
}

inline bool operator==(const TileRange &l, const TileRange &r) {
  return l.begin == r.begin && l.end == r.end;
}

// }  // namespace

TEST(ArithmeticOpsTest, GetTiledCover) {
  kernels::TensorListShape<> shape0({{150}, {50}, {150}, {30}});
  auto result0 = GetTiledCover(shape0, 50, 4);
  std::vector<TileDesc> cover0 = {{0, 0, 50, 50}, {0, 1, 50, 50}, {0, 2, 50, 50},
                                  {1, 0, 50, 50}, {2, 0, 50, 50}, {2, 1, 50, 50},
                                  {2, 2, 50, 50}, {3, 0, 30, 50}};
  std::vector<TileRange> range0 = {{0, 4}, {4, 8}};
  EXPECT_EQ(std::get<0>(result0), cover0);
  EXPECT_EQ(std::get<1>(result0), range0);

  kernels::TensorListShape<> shape1({{42}, {75}, {42}, {121}});
  auto result1 = GetTiledCover(shape1, 50, 4);
  std::vector<TileDesc> cover1 = {{0, 0, 42, 50}, {1, 0, 50, 50}, {1, 1, 25, 50},
                                  {2, 0, 42, 50}, {3, 0, 50, 50}, {3, 1, 50, 50},
                                  {3, 2, 21, 50}};
  std::vector<TileRange> range1 = {{0, 4}, {4, 7}};
  EXPECT_EQ(std::get<0>(result1), cover1);
  EXPECT_EQ(std::get<1>(result1), range1);
}

template <typename T>
using bin_op_pointer = T (*)(T, T);

template <typename Backend, typename T>
class BinaryArithmeticOpsTest
    : public ::testing::TestWithParam<std::tuple<std::string, bin_op_pointer<T>>> {
 protected:
  static constexpr int num_threads = 4;

  void TestFunction(const kernels::TensorListShape<> &shape) {
    auto backend = testing::detail::BackendStringName<Backend>();

    auto param = this->GetParam();
    auto expression_desc = std::get<0>(param) + "(&0 &1)";
    auto result_fun = std::get<1>(param);

    Pipeline pipe(shape.num_samples(), num_threads, 0);

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
    for (auto &b : batch) {
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

  void TestFunction() {
    kernels::TensorListShape<> shape0{{32000}, {2345}, {212}, {1}, {100}, {6400}, {8000}, {323},
                                      {32000}, {2345}, {212}, {1}, {100}, {6400}, {8000}, {323}};

    kernels::TensorListShape<> shape1{{1024, 768}, {4096, 1440}, {2435, 33},
                                      {17, 696},   {42, 42},     {1, 1}};
    TestFunction(shape0);
    TestFunction(shape1);
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
using BinaryArithmeticOpCPUint32Test = BinaryArithmeticOpsTest<CPUBackend, int32_t>;
using BinaryArithmeticOpCPUfloatTest = BinaryArithmeticOpsTest<CPUBackend, float>;
using BinaryArithmeticOpGPUint32Test = BinaryArithmeticOpsTest<GPUBackend, int32_t>;
using BinaryArithmeticOpGPUfloatTest = BinaryArithmeticOpsTest<GPUBackend, float>;

// Create the tests
TEST_P(BinaryArithmeticOpCPUint32Test, SimplePipeline) {
  TestFunction();
}
TEST_P(BinaryArithmeticOpCPUfloatTest, SimplePipeline) {
  TestFunction();
}
TEST_P(BinaryArithmeticOpGPUint32Test, SimplePipeline) {
  TestFunction();
}
TEST_P(BinaryArithmeticOpGPUfloatTest, SimplePipeline) {
  TestFunction();
}

// Pass the values to tests suites
INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteCPUint32, BinaryArithmeticOpCPUint32Test,
                         ::testing::ValuesIn(getOpNameRef<int32_t>()));

INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteCPUfloat, BinaryArithmeticOpCPUfloatTest,
                         ::testing::ValuesIn(getOpNameRef<float>()));

INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteGPUint32, BinaryArithmeticOpGPUint32Test,
                         ::testing::ValuesIn(getOpNameRef<int32_t>()));

INSTANTIATE_TEST_SUITE_P(BinaryArithmeticOpsSuiteGPUfloat, BinaryArithmeticOpGPUfloatTest,
                         ::testing::ValuesIn(getOpNameRef<float>()));

TEST(ArithmeticOpsTest, GenericPipeline) {
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
