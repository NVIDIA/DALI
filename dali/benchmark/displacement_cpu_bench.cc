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
#include "dali/operators/image/remap/displacement_filter_impl_cpu.h"
#include "dali/operators/image/remap/sphere.h"
#include "dali/operators/image/remap/water.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

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
  op.AddArg("max_batch_size", dummy_batch_size).AddArg("num_threads", dummy_num_thread);

  // Create Operator and pass its arguments via OpSpec
  // We get partially-specified OpSpec, added remaining arguments,
  // and we use it with DisplacementFilterType to instantiate operator
  DisplacementFilterType df(op);

  static constexpr int W = 1920, H = 1080, C = 3;
  static constexpr int N = W * H * C;

  // The inputs and outputs to CPUBackend are: shared_ptr<Tensor<CPUBackend>>;
  // create input and output, initialize input
  auto tensor_in = std::make_shared<TensorVector<CPUBackend>>(1);
  auto tensor_out = std::make_shared<TensorVector<CPUBackend>>(1);
  // If we want to specify input, we can share data
  // tensor_in->ShareData(img, N * sizeof(T));
  // Here we let underlying buffer allocate it by itself. We have to specify size and type
  tensor_in->set_type(TypeInfo::Create<T>());
  tensor_in->Resize({{W, H, C}});
  // tensor out is resized by operator itself in DisplacementFilter::DataDependentSetup()

  // TODO(klecki) Accomodate to use different inputs from test data
  auto *ptr = (*tensor_in)[0].template mutable_data<T>();
  for (int i = 0; i < N; i++) {
    ptr[i] = i;
  }

  // We need a thread pool
  ThreadPool tp(4, 0, false);

  // Create workspace and set input and output
  HostWorkspace ws;
  ws.AddInput(tensor_in);
  ws.AddOutput(tensor_out);

  ws.SetThreadPool(&tp);

  // Run once so output is allocated
  df.Run(ws);

  for (auto _ : st) {
    df.Run(ws);
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
namespace {
std::vector<float> affine_mat = { 1.0f, 0.8f, 0.0f, 0.0f, 1.2f, 0.0f };
}  // namespace

DALI_BENCHMARK_DISPLACEMENT_CASE(Water<CPUBackend>, OpSpec("Water"));

}  // namespace dali
