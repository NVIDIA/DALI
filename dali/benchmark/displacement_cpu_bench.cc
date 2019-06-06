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

#include <benchmark/benchmark.h>

#include "dali/benchmark/dali_bench.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/pipeline/operators/displacement/displacement_filter_impl_cpu.h"
#include "dali/pipeline/operators/displacement/rotate.h"
#include "dali/pipeline/operators/displacement/sphere.h"
#include "dali/pipeline/operators/displacement/water.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

namespace {

/**
 * @brief Used to store OpSpec to be passed into DisplacementBench for given operator type
 *
 * @tparam OpType Type of operator
 */
template <typename OpType>
struct OpInfo {
  static OpSpec op;
};
template <typename OpType>
OpSpec OpInfo<OpType>::op = OpSpec("OperatorName");

}  // namespace

// Register OpSpec OP_SPEC for given operator type OP_TYPE
#define DALI_DECLARE_OP_INFO(OP_TYPE, OP_SPEC) \
namespace {                                    \
template<>                                     \
OpSpec OpInfo<OP_TYPE>::op = OP_SPEC;          \
}

/**
 * @brief Benchmark function for Displacement filters
 *
 * @tparam DisplacementFilterType Displacement operator type CPUBackend,
 *         for example DisplacementFilter<CPUBackend>
 * @tparam T Underlying input data
 * @param st
 */
template <typename DisplacementFilterType, typename T>
void DisplacementBench(benchmark::State& st) {//NOLINT
  OpSpec op = OpInfo<DisplacementFilterType>::op;
  int interp_type = st.range(0);
  op.AddArg("interp_type", interp_type);

  // batch_size and num_threads are checked by OperatorBase, and are used later to create
  // appropriate executor by the Pipeline. We have to specify them,
  // but they are not used in this benchmark
  const int dummy_batch_size = 128;
  const int dummy_num_thread = 1;
  op.AddArg("batch_size", dummy_batch_size).AddArg("num_threads", dummy_num_thread);

  // Create Operator and pass its arguments via OpSpec
  // We get partially-specified OpSpec, added remaining arguments,
  // and we use it with DisplacementFilterType to instantiate operator
  DisplacementFilterType df(op);

  static constexpr int W = 1920, H = 1080, C = 3;
  static constexpr int N = W * H * C;

  // The inputs and outputs to CPUBackend are: shared_ptr<Tensor<CPUBackend>>;
  // create input and output, initialize input
  shared_ptr<Tensor<CPUBackend>> tensor_in(new Tensor<CPUBackend>());
  shared_ptr<Tensor<CPUBackend>> tensor_out(new Tensor<CPUBackend>());
  // If we want to specify input, we can share data
  // tensor_in->ShareData(img, N * sizeof(T));
  // Here we let underlying buffer allocate it by itself. We have to specify size and type
  tensor_in->set_type(TypeInfo::Create<T>());
  tensor_in->Resize({W, H, C});
  // tensor out is resized by operator itself in DisplacementFilter::DataDependentSetup()

  // TODO(klecki) Accomodate to use different inputs from test data
  auto *ptr = tensor_in->mutable_data<T>();
  for (int i = 0; i < N; i++) {
    ptr[i] = i;
  }

  // Create workspace and set input and output
  SampleWorkspace s_ws;
  s_ws.AddInput(tensor_in);
  s_ws.AddOutput(tensor_out);

  // Run once so output is allocated
  df.Run(&s_ws);

  for (auto _ : st) {
    df.Run(&s_ws);
  }
}

// Register displacement benchmarks for given type OP_TYPE.
// It registers two instantiations of template DisplacementBench function for
// given OP_TYPE with uint8_t and float data input types and sets appropraite input ranges
// and other parameters.
#define DALI_BENCHMARK_DISPLACEMENT(OP_TYPE)            \
BENCHMARK_TEMPLATE(DisplacementBench, OP_TYPE, uint8_t) \
->Range(DALI_INTERP_NN, DALI_INTERP_LINEAR)             \
->Unit(benchmark::kMillisecond)                         \
->UseRealTime();                                        \
BENCHMARK_TEMPLATE(DisplacementBench, OP_TYPE, float)   \
->Range(DALI_INTERP_NN, DALI_INTERP_LINEAR)             \
->Unit(benchmark::kMillisecond)                         \
->UseRealTime();

// Register and instantiate benchmark functions and specify OpSpec for given OpType
#define DALI_BENCHMARK_DISPLACEMENT_CASE(OP_TYPE, OP_SPEC) \
DALI_DECLARE_OP_INFO(OP_TYPE, OP_SPEC)                     \
DALI_BENCHMARK_DISPLACEMENT(OP_TYPE)

DALI_BENCHMARK_DISPLACEMENT_CASE(DisplacementFilter<CPUBackend>, OpSpec("DisplacementFilter"));
DALI_BENCHMARK_DISPLACEMENT_CASE(Rotate<CPUBackend>, OpSpec("Rotate").AddArg("angle", 42.f));
namespace {
std::vector<float> affine_mat = { 1.0f, 0.8f, 0.0f, 0.0f, 1.2f, 0.0f };
}  // namespace
DALI_BENCHMARK_DISPLACEMENT_CASE(WarpAffine<CPUBackend>,
                                 OpSpec("WarpAffine").AddArg("matrix", affine_mat));
DALI_BENCHMARK_DISPLACEMENT_CASE(Water<CPUBackend>, OpSpec("Water"));

}  // namespace dali
