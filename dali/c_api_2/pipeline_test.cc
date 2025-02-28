// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/c_api_2/pipeline.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/executor/executor2/exec2_ops_for_test.h"

namespace dali {

OpSpec CounterOp(const std::string &name) {
  return OpSpec(exec2::test::kCounterOpName)
    .AddArg(name, string(name))
    .AddOutput(name, StorageDevice::CPU);

}

OpSpec TestOp(std::string_view name, std::string_view device) {
  return OpSpec(exec2::test::kTestOpName)
      .AddArg("name", name)
      .AddArg("device", device);
}

std::string GetCPUOnlyPipelineProto(int max_batch_size, int num_threads, int device_id) {
  Pipeline p(max_batch_size, num_threads, device_id);

  OpSpec op1 = TestOp("op1", "cpu")
    .AddInput("ctr", StorageDevice::CPU)
    .AddOutput("op1", StorageDevice::CPU)
    .AddArg("addend", 1000);

  OpSpec op2 = TestOp("op2", "cpu")
    .AddInput("ctr", StorageDevice::CPU)
    .AddOutput("op2", StorageDevice::CPU)
    .AddArg("addend", 2000);

  p.AddOperator(CounterOp("ctr"), "ctr");
  p.AddOperator(op1, "op1");
  p.AddOperator(op2, "op2");
  p.SetOutputDescs({ {"op1", "cpu" }, {"op2", "cpu"} });

  return p.SerializeToProtobuf();
}

std::string GetCPU2GPUPipelineProto(int max_batch_size, int num_threads, int device_id) {
  Pipeline p(max_batch_size, num_threads, device_id);
  OpSpec op1 = TestOp("op1", "cpu")
    .AddInput("ctr", StorageDevice::CPU)
    .AddOutput("op1", StorageDevice::CPU)
    .AddArg("addend", 1000);

  OpSpec op2 = TestOp("op2", "gpu")
    .AddInput("ctr", StorageDevice::GPU)
    .AddOutput("op2", StorageDevice::GPU)
    .AddArg("addend", 2000);

  p.AddOperator(CounterOp("ctr"), "ctr");
  p.AddOperator(op1, "op1");
  p.AddOperator(op2, "op2");
  p.SetOutputDescs({ {"op1", "cpu" }, {"op2", "cpu"} });

  return p.SerializeToProtobuf();
}

namespace c_api {
namespace test {

inline Pipeline *GetPipeline(daliPipeline_h h) {
  return h ? static_cast<PipelineWrapper *>(h)->Unwrap() : nullptr;
}

TEST(CAPI2_PipelineTest, Deserialize) {
  std::string proto = GetCPUOnlyPipelineProto(4, 4, 0);

  daliPipelineParams_t params{};
  daliPipeline_h h = nullptr;
  EXPECT_EQ(daliPipelineDeserialize(&h, proto.c_str(), proto.length(), &params), DALI_SUCCESS)
    << daliGetLastErrorMessage();
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(daliPipelineBuild(h), DALI_SUCCESS) << daliGetLastErrorMessage();
  EXPECT_EQ(GetPipeline(h)->output_descs().size(), 2);
  EXPECT_EQ(daliPipelineDestroy(h), DALI_SUCCESS);
}

void CheckSequence(daliTensorList_h tl, int start, int stride) {
  // TODO
}

TEST(CAPI2_PipelineTest, Run) {
  std::string proto = GetCPUOnlyPipelineProto(4, 4, 0);

  daliPipelineParams_t params{};
  params.prefetch_queue_depth_present = true;
  params.prefetch_queue_depth = 2;
  params.enable_checkpointing_present = true;
  params.enable_checkpointing = 2;
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;

  daliPipeline_h raw_h = nullptr;
  EXPECT_EQ(daliPipelineDeserialize(&raw_h, proto.c_str(), proto.length(), &params), DALI_SUCCESS)
    << daliGetLastErrorMessage();
  ASSERT_NE(raw_h, nullptr);
  PipelineHandle h(raw_h);

  EXPECT_EQ(daliPipelineBuild(h), DALI_SUCCESS) << daliGetLastErrorMessage();
  EXPECT_EQ(daliPipelinePrefetch(h), DALI_SUCCESS) << daliGetLastErrorMessage();
  daliPipelineOutputs_h raw_out_h;
  EXPECT_EQ(daliPipelinePopOutputs(h, &raw_out_h), DALI_SUCCESS);
  ASSERT_NE(raw_out_h, nullptr);
  PipelineOutputsHandle out_h(raw_out_h);
  EXPECT_EQ(daliPipelineOutputsDestroy(out_h.release()), DALI_SUCCESS);
}

}  // namespace test
}  // namespace c_api
}  // namespace dali
