// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include "dali/dali.h"
#include "dali/dali_cpp_wrappers.h"
#include "dali/c_api_2/pipeline.h"
#include "dali/c_api_2/pipeline_test_utils.h"
#include "dali/c_api_2/data_objects.h"
#include "dali/pipeline/executor/executor2/exec2_ops_for_test.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/pipeline_params.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali::c_api::test {

namespace {

constexpr int kBatchSize  = 4;
constexpr int kNumThreads = 4;

// ----- Helpers for building reference (C++) pipelines -----

// Counter + two CPU TestOps: ctr -> op1(+1000), ctr -> op2(+2000); outputs: op1, op2
std::unique_ptr<Pipeline> BuildRefCPUPipeline(std::optional<int> device_id) {
  auto p = std::make_unique<Pipeline>(MakePipelineParams(
      kBatchSize, kNumThreads, device_id.value_or(CPU_ONLY_DEVICE_ID)));

  p->AddOperator(
    OpSpec(exec2::test::kCounterOpName)
      .AddArg("ctr", std::string("ctr"))
      .AddOutput("ctr", StorageDevice::CPU),
    "ctr");
  p->AddOperator(
    OpSpec(exec2::test::kTestOpName)
      .AddArg("name", std::string("op1"))
      .AddArg("device", std::string("cpu"))
      .AddInput("ctr", StorageDevice::CPU)
      .AddOutput("op1", StorageDevice::CPU)
      .AddArg("addend", 1000),
    "op1");
  p->AddOperator(
    OpSpec(exec2::test::kTestOpName)
      .AddArg("name", std::string("op2"))
      .AddArg("device", std::string("cpu"))
      .AddInput("ctr", StorageDevice::CPU)
      .AddOutput("op2", StorageDevice::CPU)
      .AddArg("addend", 2000),
    "op2");
  p->SetOutputDescs({ {"op1", "cpu"}, {"op2", "cpu"} });
  p->Build();
  return p;
}

// ExternalSource (cpu) -> output "ext" directly; used for feed-input comparison
std::unique_ptr<Pipeline> BuildRefExtSrcPipeline(std::optional<int> device_id) {
  auto p = std::make_unique<Pipeline>(MakePipelineParams(
      kBatchSize, kNumThreads, device_id.value_or(CPU_ONLY_DEVICE_ID)));

  p->AddExternalInput("ext", "cpu");
  p->SetOutputDescs({ {"ext", "cpu"} });
  p->Build();
  return p;
}

// ----- Helpers for building C API pipelines -----

daliPipelineParams_t MakeCApiParams(std::optional<int> device_id) {
  daliPipelineParams_t params{};
  params.max_batch_size_present = true;
  params.max_batch_size = kBatchSize;
  params.num_threads_present = true;
  params.num_threads = kNumThreads;
  params.device_id_present = device_id.has_value();
  params.device_id = device_id.value_or(-1);
  return params;
}

// Counter + two CPU TestOps built via C API, matching BuildRefCPUPipeline
PipelineHandle BuildCApiCPUPipeline(std::optional<int> device_id) {
  auto params = MakeCApiParams(device_id);
  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineCreate(&h, &params));
  PipelineHandle pipe(h);

  // CounterOp "ctr": schema arg "ctr"="ctr", output "ctr" (CPU)
  daliArgDesc_t ctr_args[1];
  ctr_args[0].name  = "ctr";
  ctr_args[0].dtype = DALI_STRING;
  ctr_args[0].str   = "ctr";

  daliIODesc_t ctr_out[1];
  ctr_out[0].name        = "ctr";
  ctr_out[0].device_type = DALI_STORAGE_CPU;

  daliOperatorDesc_t ctr_op{};
  ctr_op.schema_name   = exec2::test::kCounterOpName;
  ctr_op.instance_name = "ctr";
  ctr_op.backend       = DALI_BACKEND_CPU;
  ctr_op.num_outputs   = 1;
  ctr_op.num_args      = 1;
  ctr_op.outputs       = ctr_out;
  ctr_op.args          = ctr_args;
  CHECK_DALI(daliPipelineAddOperator(h, &ctr_op));

  // TestOp "op1": input "ctr" (CPU), output "op1" (CPU), addend=1000
  daliArgDesc_t op1_args[2];
  op1_args[0].name    = "name";
  op1_args[0].dtype   = DALI_STRING;
  op1_args[0].str     = "op1";
  op1_args[1].name    = "addend";
  op1_args[1].dtype   = DALI_INT32;
  op1_args[1].ivalue  = 1000;

  daliIODesc_t op1_in[1];
  op1_in[0].name        = "ctr";
  op1_in[0].device_type = DALI_STORAGE_CPU;

  daliIODesc_t op1_out[1];
  op1_out[0].name        = "op1";
  op1_out[0].device_type = DALI_STORAGE_CPU;

  daliOperatorDesc_t op1_op{};
  op1_op.schema_name   = exec2::test::kTestOpName;
  op1_op.instance_name = "op1";
  op1_op.backend       = DALI_BACKEND_CPU;
  op1_op.num_inputs    = 1;
  op1_op.num_outputs   = 1;
  op1_op.num_args      = 2;
  op1_op.inputs        = op1_in;
  op1_op.outputs       = op1_out;
  op1_op.args          = op1_args;
  CHECK_DALI(daliPipelineAddOperator(h, &op1_op));

  // TestOp "op2": input "ctr" (CPU), output "op2" (CPU), addend=2000
  daliArgDesc_t op2_args[2];
  op2_args[0].name    = "name";
  op2_args[0].dtype   = DALI_STRING;
  op2_args[0].str     = "op2";
  op2_args[1].name    = "addend";
  op2_args[1].dtype   = DALI_INT32;
  op2_args[1].ivalue  = 2000;

  daliIODesc_t op2_in[1];
  op2_in[0].name        = "ctr";
  op2_in[0].device_type = DALI_STORAGE_CPU;

  daliIODesc_t op2_out[1];
  op2_out[0].name        = "op2";
  op2_out[0].device_type = DALI_STORAGE_CPU;

  daliOperatorDesc_t op2_op{};
  op2_op.schema_name   = exec2::test::kTestOpName;
  op2_op.instance_name = "op2";
  op2_op.backend       = DALI_BACKEND_CPU;
  op2_op.num_inputs    = 1;
  op2_op.num_outputs   = 1;
  op2_op.num_args      = 2;
  op2_op.inputs        = op2_in;
  op2_op.outputs       = op2_out;
  op2_op.args          = op2_args;
  CHECK_DALI(daliPipelineAddOperator(h, &op2_op));

  // Set outputs: op1 (cpu), op2 (cpu)
  daliPipelineIODesc_t out_descs[2];
  out_descs[0] = {};
  out_descs[0].name   = "op1";
  out_descs[0].device = DALI_STORAGE_CPU;
  out_descs[1] = {};
  out_descs[1].name   = "op2";
  out_descs[1].device = DALI_STORAGE_CPU;
  CHECK_DALI(daliPipelineSetOutputs(h, 2, out_descs));

  CHECK_DALI(daliPipelineBuild(h));
  return pipe;
}

// ExternalSource "ext" (CPU) built via C API, matching BuildRefExtSrcPipeline
PipelineHandle BuildCApiExtSrcPipeline(std::optional<int> device_id) {
  auto params = MakeCApiParams(device_id);
  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineCreate(&h, &params));
  PipelineHandle pipe(h);

  daliPipelineIODesc_t ext_desc{};
  ext_desc.name   = "ext";
  ext_desc.device = DALI_STORAGE_CPU;
  CHECK_DALI(daliPipelineAddExternalInput(h, &ext_desc));

  daliPipelineIODesc_t out_desc{};
  out_desc.name   = "ext";
  out_desc.device = DALI_STORAGE_CPU;
  CHECK_DALI(daliPipelineSetOutputs(h, 1, &out_desc));

  CHECK_DALI(daliPipelineBuild(h));
  return pipe;
}

}  // namespace

// --- Test 1: Operator pipeline (counter + two CPU TestOps) ---
// Build both C++ and C API pipelines, run 5 iterations, compare outputs.

TEST(CAPI2_PipelineBuilderTest, AddOperator_CPUOnly) {
  auto ref  = BuildRefCPUPipeline(std::nullopt);
  auto test = BuildCApiCPUPipeline(std::nullopt);
  ComparePipelineOutputs(*ref, test, /*iters=*/5, /*prefetch_on_first_iter=*/true);
}

// --- Test 2: ExternalSource pipeline ---
// Build both C++ (AddExternalInput) and C API (daliPipelineAddExternalInput) pipelines.
// Feed identical data to both, run, compare outputs.

TEST(CAPI2_PipelineBuilderTest, AddExternalInput) {
  auto ref  = BuildRefExtSrcPipeline(0);
  auto test = BuildCApiExtSrcPipeline(0);

  // Determine how many times we need to feed before prefetching
  int feed_count = ref->InputFeedCount("ext");

  // Create and feed identical scalar TensorLists
  for (int i = 0; i < feed_count; i++) {
    auto cpp_tl = std::make_shared<TensorList<CPUBackend>>();
    cpp_tl->Resize(uniform_list_shape(kBatchSize, TensorShape<>{}), DALI_INT32);
    for (int s = 0; s < kBatchSize; s++)
      (*cpp_tl)[s].mutable_data<int>()[0] = i * kBatchSize + s;

    // Feed to reference C++ pipeline
    ref->SetExternalInput("ext", *cpp_tl);

    // Feed to test C API pipeline
    auto tl_handle = Wrap(cpp_tl);
    CHECK_DALI(daliPipelineFeedInput(test, "ext", tl_handle.get(), nullptr, {}, nullptr));
  }

  // Prefetch both, then compare outputs for each prefetched batch
  ref->Prefetch();
  CHECK_DALI(daliPipelinePrefetch(test));

  for (int i = 0; i < feed_count; i++)
    ComparePipelineOutput(*ref, test);
}

}  // namespace dali::c_api::test
