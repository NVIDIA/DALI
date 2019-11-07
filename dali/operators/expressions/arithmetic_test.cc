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
#include "dali/operators/expressions/arithmetic.h"
#include "dali/operators/expressions/arithmetic_meta.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_operator_test.h"
#include "dali/test/tensor_test_utils.h"

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
  auto result_shape = PropagateShapes<CPUBackend>(expr_ref, ws, 2);
  auto result_layout = GetCommonLayout<CPUBackend>(expr_ref, ws);
  auto expected_shpe = TensorListShape<>{{1}, {2}};
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

TEST(ArithmeticOpsTest, PropagateScalarLike) {
  std::string expr_str = "sub(&0 $1:int32))";
  auto expr = ParseExpressionString(expr_str);
  auto &expr_ref = *expr;
  HostWorkspace ws;
  std::shared_ptr<TensorVector<CPUBackend>> in[1];
  for (auto &ptr : in) {
    ptr = std::make_shared<TensorVector<CPUBackend>>();
    ptr->Resize({{1}, {1}});
    ptr->SetLayout(TensorLayout("HW"));
  }
  ws.AddInput(in[0]);

  auto result_shape = PropagateShapes<CPUBackend>(expr_ref, ws, 2);
  auto expected_shpe = TensorListShape<>{{1}, {1}};
  EXPECT_EQ(result_shape, expected_shpe);
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

  ASSERT_THROW(PropagateShapes<CPUBackend>(expr_ref, ws, 2), std::runtime_error);
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
  TensorListShape<> shape0({{150}, {50}, {150}, {30}});
  auto result0 = GetTiledCover(shape0, 50, 4);
  std::vector<TileDesc> cover0 = {{0, 0, 50, 50}, {0, 1, 50, 50}, {0, 2, 50, 50},
                                  {1, 0, 50, 50}, {2, 0, 50, 50}, {2, 1, 50, 50},
                                  {2, 2, 50, 50}, {3, 0, 30, 50}};
  std::vector<TileRange> range0 = {{0, 4}, {4, 8}};
  EXPECT_EQ(std::get<0>(result0), cover0);
  EXPECT_EQ(std::get<1>(result0), range0);

  TensorListShape<> shape1({{42}, {75}, {42}, {121}});
  auto result1 = GetTiledCover(shape1, 50, 4);
  std::vector<TileDesc> cover1 = {{0, 0, 42, 50}, {1, 0, 50, 50}, {1, 1, 25, 50},
                                  {2, 0, 42, 50}, {3, 0, 50, 50}, {3, 1, 50, 50},
                                  {3, 2, 21, 50}};
  std::vector<TileRange> range1 = {{0, 4}, {4, 7}};
  EXPECT_EQ(std::get<0>(result1), cover1);
  EXPECT_EQ(std::get<1>(result1), range1);
}

namespace {

template <typename T>
T GenerateData(int sample, int element) {
  static std::mt19937 gen(42);
  auto result = uniform_distribution<T>(-1024, 1024)(gen);
  return result == 0 ? 1 : result;  // we do not want to divide by 0 so we discard those results
}

template <typename T>
void FillBatch(TensorList<CPUBackend> &batch, const TensorListShape<> shape) {
  batch.Resize(shape);
  batch.set_type(TypeInfo::Create<T>());
  for (int i = 0; i < shape.num_samples(); i++) {
    auto *t = batch.template mutable_tensor<T>(i);
    for (int j = 0; j < shape[i].num_elements(); j++) {
      t[j] = GenerateData<T>(i, j);
    }
  }
}

}  // namespace

template <typename T>
using bin_op_pointer = T (*)(T, T);

template <typename Backend, typename T>
class BinaryArithmeticOpsTest
    : public ::testing::TestWithParam<std::tuple<std::string, bin_op_pointer<T>>> {
 protected:
  static constexpr int num_threads = 4;

  void TestFunction(const TensorListShape<> &shape) {
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
      FillBatch<T>(b, shape);
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
    TensorListShape<> shape0{{32000}, {2345}, {212}, {1}, {100}, {6400}, {8000}, {323},
                             {32000}, {2345}, {212}, {1}, {100}, {6400}, {8000}, {323}};

    TensorListShape<> shape1{{1024, 768}, {4096, 1440}, {2435, 33}, {17, 696}, {42, 42}, {1, 1}};
    TestFunction(shape0);
    TestFunction(shape1);
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
  FillBatch<int>(batch, uniform_list_shape(batch_size, {tensor_elements}));

  pipe.SetExternalInput("data0", batch);
  pipe.SetExternalInput("data1", batch);
  pipe.RunCPU();
  pipe.RunGPU();
  DeviceWorkspace ws;
  pipe.Outputs(&ws);

  const auto *data = batch.data<int>();

  auto *result = ws.OutputRef<CPUBackend>(0).data<int32_t>();
  auto *result2 = ws.OutputRef<GPUBackend>(1).data<int32_t>();
  vector<int32_t> result2_cpu(batch_size * tensor_elements);

  MemCopy(result2_cpu.data(), result2, batch_size * tensor_elements * sizeof(int));
  for (int i = 0; i < batch_size * tensor_elements; i++) {
    EXPECT_EQ(result[i], data[i] + data[i]);
    EXPECT_EQ(result2_cpu[i], data[i] * (data[i] + data[i]));
  }
}

TEST(ArithmeticOpsTest, FdivPipeline) {
  constexpr int batch_size = 16;
  constexpr int num_threads = 4;
  constexpr int tensor_elements = 16;
  Pipeline pipe(batch_size, num_threads, 0);

  pipe.AddExternalInput("data0");
  pipe.AddExternalInput("data1");

  pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                       .AddArg("device", "cpu")
                       .AddArg("expression_desc", "fdiv(&0 &1)")
                       .AddInput("data0", "cpu")
                       .AddInput("data1", "cpu")
                       .AddOutput("result0", "cpu"),
                   "arithm_cpu");

  pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                       .AddArg("device", "gpu")
                       .AddArg("expression_desc", "fdiv(&0 &1)")
                       .AddInput("data0", "gpu")
                       .AddInput("data1", "gpu")
                       .AddOutput("result1", "gpu"),
                   "arithm_gpu");

  vector<std::pair<string, string>> outputs = {{"result0", "cpu"}, {"result1", "gpu"}};

  pipe.Build(outputs);

  TensorList<CPUBackend> batch[2];
  for (auto &b : batch) {
    FillBatch<int>(b, uniform_list_shape(batch_size, {tensor_elements}));
  }

  pipe.SetExternalInput("data0", batch[0]);
  pipe.SetExternalInput("data1", batch[1]);
  pipe.RunCPU();
  pipe.RunGPU();
  DeviceWorkspace ws;
  pipe.Outputs(&ws);
  ASSERT_EQ(ws.OutputRef<CPUBackend>(0).type(), TypeInfo::Create<float>());
  ASSERT_EQ(ws.OutputRef<GPUBackend>(1).type(), TypeInfo::Create<float>());

  const auto *data0 = batch[0].data<int>();
  const auto *data1 = batch[1].data<int>();

  auto *result0 = ws.OutputRef<CPUBackend>(0).data<float>();
  auto *result1 = ws.OutputRef<GPUBackend>(1).data<float>();
  vector<float> result1_cpu(batch_size * tensor_elements);

  MemCopy(result1_cpu.data(), result1, batch_size * tensor_elements * sizeof(float));
  for (int i = 0; i < batch_size * tensor_elements; i++) {
    EXPECT_EQ(result0[i], static_cast<float>(data0[i]) / data1[i]);
    EXPECT_EQ(result1_cpu[i], static_cast<float>(data0[i]) / data1[i]);
  }
}

TEST(ArithmeticOpsTest, ConstantsPipeline) {
  constexpr int magic_int = 42;
  constexpr int magic_float = 42.f;
  constexpr int batch_size = 16;
  constexpr int num_threads = 4;
  constexpr int tensor_elements = 16;
  Pipeline pipe(batch_size, num_threads, 0);

  pipe.AddExternalInput("data0");

  pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                       .AddArg("device", "cpu")
                       .AddArg("expression_desc", "add(&0 $0:int32)")
                       .AddArg("integer_constants", std::vector<int>{magic_int})
                       .AddInput("data0", "cpu")
                       .AddOutput("result0", "cpu"),
                   "arithm_cpu_add");

  pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                       .AddArg("device", "cpu")
                       .AddArg("expression_desc", "mul(&0 $0:float32)")
                       .AddArg("real_constants", std::vector<float>{magic_float})
                       .AddInput("data0", "cpu")
                       .AddOutput("result1", "cpu"),
                   "arithm_cpu_mul");

  vector<std::pair<string, string>> outputs = {{"result0", "cpu"}, {"result1", "cpu"}};

  pipe.Build(outputs);

  TensorList<CPUBackend> batch;
  FillBatch<int>(batch, uniform_list_shape(batch_size, {tensor_elements}));

  pipe.SetExternalInput("data0", batch);
  pipe.RunCPU();
  pipe.RunGPU();
  DeviceWorkspace ws;
  pipe.Outputs(&ws);

  const auto *data = batch.data<int>();
  auto *result0 = ws.OutputRef<CPUBackend>(0).data<int32_t>();
  auto *result1 = ws.OutputRef<CPUBackend>(1).data<float>();

  for (int i = 0; i < batch_size * tensor_elements; i++) {
    EXPECT_EQ(result0[i], data[i] + magic_int);
    EXPECT_EQ(result1[i], data[i] * magic_float);
  }
}

using shape_sequence = std::vector<std::array<TensorListShape<>, 3>>;

int GetBatchSize(const shape_sequence &seq) {
  return seq[0][0].num_samples();
}

class ArithmeticOpsScalarTest :  public ::testing::TestWithParam<shape_sequence> {
 public:
  void Run() {
    constexpr int num_threads = 4;
    auto shape_seq = GetParam();
    int batch_size = GetBatchSize(shape_seq);
    Pipeline pipe(batch_size, num_threads, 0);

    pipe.AddExternalInput("data0");
    pipe.AddExternalInput("data1");

    pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                        .AddArg("device", "cpu")
                        .AddArg("expression_desc", "add(&0 &1)")
                        .AddInput("data0", "cpu")
                        .AddInput("data1", "cpu")
                        .AddOutput("result0", "cpu"),
                    "arithm_cpu");

    pipe.AddOperator(OpSpec("ArithmeticGenericOp")
                        .AddArg("device", "gpu")
                        .AddArg("expression_desc", "add(&0 &1)")
                        .AddInput("data0", "gpu")
                        .AddInput("data1", "gpu")
                        .AddOutput("result1", "gpu"),
                    "arithm_gpu");

    vector<std::pair<string, string>> outputs = {{"result0", "cpu"}, {"result1", "gpu"}};

    pipe.Build(outputs);

    for (const auto &s : shape_seq) {
      const auto &result_shape = s[2];

      TensorList<CPUBackend> batch[2];
      for (int i = 0; i < 2; i++) {
        FillBatch<int>(batch[i], s[i]);
      }

      pipe.SetExternalInput("data0", batch[0]);
      pipe.SetExternalInput("data1", batch[1]);
      pipe.RunCPU();
      pipe.RunGPU();
      DeviceWorkspace ws;
      pipe.Outputs(&ws);


      const auto *data0 = batch[0].data<int>();
      const auto *data1 = batch[1].data<int>();

      const auto *result0 = ws.OutputRef<CPUBackend>(0).data<int>();
      const auto *result1 = ws.OutputRef<GPUBackend>(1).data<int>();

      ASSERT_EQ(ws.OutputRef<CPUBackend>(0).shape(), result_shape);
      ASSERT_EQ(ws.OutputRef<GPUBackend>(1).shape(), result_shape);

      vector<int> result1_cpu(result_shape.num_elements());

      MemCopy(result1_cpu.data(), result1, result_shape.num_elements() * sizeof(int));

      int64_t offset_out = 0;
      int64_t offset_in[2] = {0, 0};
      for (int tensor_idx = 0; tensor_idx < result_shape.num_samples(); tensor_idx++) {
        for (int j = 0; j < result_shape[tensor_idx].num_elements(); j++) {
          auto is_scalar = [] (auto &shape, int tensor_idx) {
            return shape[tensor_idx] == TensorShape<>{1};
          };
          int expected = data0[offset_in[0] + (is_scalar(s[0], tensor_idx) ? 0 : j)] +
                         data1[offset_in[1] + (is_scalar(s[1], tensor_idx) ? 0 : j)];

          ASSERT_EQ(result0[offset_out + j], expected)
              << " difference at sample: " << tensor_idx << ", element: " << j;
          ASSERT_EQ(result1_cpu[offset_out + j], expected)
              << " difference at sample: " << tensor_idx << ", element: " << j;
        }
        offset_out += result_shape[tensor_idx].num_elements();
        for (int in_idx = 0; in_idx < 2; in_idx++) {
          offset_in[in_idx] += s[in_idx][tensor_idx].num_elements();
        }
      }
    }
  }
};

TEST_P(ArithmeticOpsScalarTest, TensorScalarMix) {
  this->Run();
}

namespace {

std::array<TensorListShape<>, 3> GetShapesForSequence(int batch_size, int left_elems,
                                                      int right_elems) {
  return {uniform_list_shape(batch_size, {left_elems}),
          uniform_list_shape(batch_size, {right_elems}),
          uniform_list_shape(batch_size, {std::max(left_elems, right_elems)})};
}

/**
 * @brief Return a vector of shape_sequences
 */
std::vector<shape_sequence> GetTensorScalarTensorSequences(int batch_size) {
  auto seq0 = shape_sequence{
    GetShapesForSequence(batch_size, 5, 5),
    GetShapesForSequence(batch_size, 1, 1),
    GetShapesForSequence(batch_size, 6, 6),
  };

  auto seq1 = shape_sequence{
    GetShapesForSequence(batch_size, 5, 5),
    GetShapesForSequence(batch_size, 1, 5),
    GetShapesForSequence(batch_size, 5, 1),
    GetShapesForSequence(batch_size, 1, 1),
    GetShapesForSequence(batch_size, 6, 6),
  };

  auto seq2 = shape_sequence{
    GetShapesForSequence(batch_size, 5, 5),
    GetShapesForSequence(batch_size, 1, 5),
    GetShapesForSequence(batch_size, 5, 1),
    GetShapesForSequence(batch_size, 1, 1),
    GetShapesForSequence(batch_size, 6, 6),
  };

  auto seq3 = shape_sequence{
    GetShapesForSequence(batch_size, 5, 5),
    GetShapesForSequence(batch_size, 1, 5),
    GetShapesForSequence(batch_size, 1, 1),
    GetShapesForSequence(batch_size, 5, 1),
    GetShapesForSequence(batch_size, 6, 6),
  };

  auto seq4 = shape_sequence{
    GetShapesForSequence(batch_size, 5, 5),
    GetShapesForSequence(batch_size, 5, 1),
    GetShapesForSequence(batch_size, 1, 1),
    GetShapesForSequence(batch_size, 1, 5),
    GetShapesForSequence(batch_size, 6, 6),
  };
  return {seq0, seq1, seq2, seq3, seq4};
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(TensorScalarTensorBatch5,
                         ArithmeticOpsScalarTest,
                         ::testing::ValuesIn(GetTensorScalarTensorSequences(5)));

INSTANTIATE_TEST_SUITE_P(TensorScalarTensorBatch1,
                         ArithmeticOpsScalarTest,
                         ::testing::ValuesIn(GetTensorScalarTensorSequences(1)));

}  // namespace dali

