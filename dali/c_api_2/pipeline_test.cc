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
#include "dali/c_api_2/pipeline_test_utils.h"

namespace dali {

namespace {

template <typename Backend>
class TestOpWithTrace : public Operator<Backend> {
 public:
  explicit TestOpWithTrace(const OpSpec &spec) : Operator<Backend>(spec) {
    prefix_ = spec.GetArgument<std::string>("prefix") + "_";
  }

  bool SetupImpl(std::vector<OutputDesc> &outs, const Workspace &ws) override {
    outs.resize(1);
    outs[0].shape = ws.GetInputShape(0);
    outs[0].type = ws.GetInputDataType(0);
    return true;
  }

  void RunImpl(Workspace &ws) override {
    ws.Output<Backend>(0).Copy(ws.Input<Backend>(0));
    ws.SetOperatorTrace("trace", make_string(prefix_, iter_++));
  }

 private:
  int iter_ = 0;
  std::string prefix_;
};

DALI_SCHEMA(TestOpWithTrace)
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("prefix", "trace prefix", "value");

DALI_REGISTER_OPERATOR(TestOpWithTrace, TestOpWithTrace<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(TestOpWithTrace, TestOpWithTrace<GPUBackend>, GPU);

}  // namespace

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

OpSpec TraceOp(std::string_view name, std::string_view device, std::string_view prefix) {
  return OpSpec("TestOpWithTrace")
      .AddArg("name", name)
      .AddArg("device", device)
      .AddArg("prefix", prefix);
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
  p.SetOutputDescs({ {"op1", "cpu" }, {"op2", "gpu"} });

  return p.SerializeToProtobuf();
}

std::string GetGPU2CPUPipelineProto(int max_batch_size, int num_threads, int device_id) {
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

std::string GetPipelineWithTraces(int max_batch_size, int num_threads, int device_id) {
  Pipeline p(max_batch_size, num_threads, device_id);
  OpSpec op1 = TraceOp("op1", "cpu", "t1")
    .AddInput("ctr", StorageDevice::CPU)
    .AddOutput("op1", StorageDevice::CPU);

  OpSpec op2 = TraceOp("op1", "gpu", "t2")
    .AddInput("ctr", StorageDevice::GPU)
    .AddOutput("op2", StorageDevice::GPU);

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
  CHECK_DALI(daliPipelineDeserialize(&h, proto.c_str(), proto.length(), &params));
  ASSERT_NE(h, nullptr);
  CHECK_DALI(daliPipelineBuild(h));
  EXPECT_EQ(GetPipeline(h)->output_descs().size(), 2);
  CHECK_DALI(daliPipelineDestroy(h));
}

template <typename T = int>
void CheckScalarSequence(daliTensorList_h tl, int expected_batch_size, int start, int stride) {
  int num_samples = 0, ndim = -1;
  CHECK_DALI(daliTensorListGetShape(tl, &num_samples, &ndim, nullptr));
  ASSERT_EQ(ndim, 0);
  EXPECT_EQ(num_samples, expected_batch_size);
  daliBufferPlacement_t placement{};
  CHECK_DALI(daliTensorListGetBufferPlacement(tl, &placement));

  cudaStream_t stream = 0;
  if (auto result = daliTensorListGetStream(tl, &stream)) {
    ASSERT_EQ(result, DALI_NO_DATA) << daliGetLastErrorMessage();
  }

  for (int i = 0; i < num_samples; i++) {
    daliTensorDesc_t desc{};
    CHECK_DALI(daliTensorListGetTensorDesc(tl, &desc, i));
    ASSERT_EQ(desc.dtype, type2id<T>::value);
    T value;
    if (placement.device_type == DALI_STORAGE_CPU) {
      value = *static_cast<const T *>(desc.data);
    } else {
      ASSERT_EQ(placement.device_type, DALI_STORAGE_GPU);
      CUDA_CALL(cudaMemcpyAsync(&value, desc.data, sizeof(T), cudaMemcpyDeviceToHost, stream));
      AccessOrder::host().wait(stream);
    }
    EXPECT_EQ(value, static_cast<T>(start + i * stride)) << " in sample " << i;
  }
}

enum PipelineType {
  CPUOnly,
  CPU2GPU,
  GPU2CPU
};

void TestPipelineRun(PipelineType p) {
  std::string proto;
  switch (p) {
    case CPUOnly:
      proto = GetCPUOnlyPipelineProto(1, 4, CPU_ONLY_DEVICE_ID);
      break;
      case CPU2GPU:
      proto = GetCPU2GPUPipelineProto(1, 4, 0);
      break;
    case GPU2CPU:
      proto = GetGPU2CPUPipelineProto(1, 4, 0);
      break;
    default:
      FAIL() << "Invalid pipeline type.";
  }

  daliPipelineParams_t params{};
  params.max_batch_size_present = true;
  params.max_batch_size = 4;
  params.prefetch_queue_depth_present = true;
  params.prefetch_queue_depth = 3;
  params.enable_checkpointing_present = true;
  params.enable_checkpointing = 2;
  if (p == GPU2CPU) {
    params.exec_type_present = true;
    params.exec_type = DALI_EXEC_DYNAMIC;
  }

  PipelineHandle h = Deserialize(proto, params);
  ASSERT_NE(h, nullptr);

  CHECK_DALI(daliPipelineBuild(h));
  for (int iter = 0; iter < 5; iter++) {
    if (iter == 0) {
      CHECK_DALI(daliPipelinePrefetch(h));
    } else {
      CHECK_DALI(daliPipelineRun(h));
    }

    auto out_h = PopOutputs(h);
    ASSERT_NE(out_h, nullptr);
    auto o1 = GetOutput(out_h, 0);
    auto o2 = GetOutput(out_h, 1);
    CheckScalarSequence(o1, params.max_batch_size, 1000 + iter * params.max_batch_size, 2);
    CheckScalarSequence(o2, params.max_batch_size, 2000 + iter * params.max_batch_size, 2);
    CHECK_DALI(daliPipelineOutputsDestroy(out_h.release()));
  }
}

TEST(CAPI2_PipelineTest, RunCPUOnly) {
  TestPipelineRun(CPUOnly);
}

TEST(CAPI2_PipelineTest, RunCPU2GPU) {
  TestPipelineRun(GPU2CPU);
}

TEST(CAPI2_PipelineTest, RunGPU2CPU) {
  TestPipelineRun(CPU2GPU);
}

TEST(CAPI2_PipelineTest, IncompatibleExec) {
  auto proto = GetGPU2CPUPipelineProto(1, 4, 0);
  daliPipelineParams_t params{};
  params.max_batch_size_present = true;
  params.max_batch_size = 4;
  params.prefetch_queue_depth_present = true;
  params.prefetch_queue_depth = 3;
  params.enable_checkpointing_present = true;
  params.enable_checkpointing = 2;
  params.exec_type_present = true;

  params.exec_type = DALI_EXEC_ASYNC_PIPELINED;
  auto deserialize_and_build = [&]() {
    daliClearLastError();
    auto h = Deserialize(proto, params);
    if (h) {
      return daliPipelineBuild(h);
    }
    return daliGetLastError();
  };
  EXPECT_EQ(deserialize_and_build(), DALI_ERROR_INVALID_OPERATION);
  params.exec_type = DALI_EXEC_SIMPLE;
  EXPECT_EQ(deserialize_and_build(), DALI_ERROR_INVALID_OPERATION);
  params.exec_type = DALI_EXEC_DYNAMIC;
  EXPECT_EQ(deserialize_and_build(), DALI_SUCCESS);
}

TEST(CAPI2_PipelineTest, Traces) {
  auto proto = GetPipelineWithTraces(4, 4, 0);
  daliPipelineParams_t params{};
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;

  auto h = Deserialize(proto, params);
  ASSERT_NE(h, nullptr);
  CHECK_DALI(daliPipelineBuild(h));
  for (int i = 0; i < 3; i++) {
    CHECK_DALI(daliPipelineRun(h));
    auto out_h = PopOutputs(h);
    ASSERT_NE(out_h, nullptr);

    // Check individual trace queries
    const char *t = nullptr;
    CHECK_DALI(daliPipelineOutputsGetTrace(out_h, &t, "op1", "trace"));
    auto ref_t1 = make_string("t1_", i);
    auto ref_t2 = make_string("t2_", i);
    EXPECT_EQ(t, ref_t1);
    CHECK_DALI(daliPipelineOutputsGetTrace(out_h, &t, "op2", "trace"));
    EXPECT_EQ(t, ref_t2);
    const daliOperatorTrace_t *traces = nullptr;
    int n;
    // Check bulk trace query
    CHECK_DALI(daliPipelineOutputsGetTraces(out_h, &traces, &n));
    ASSERT_EQ(n, 2);
    EXPECT_STREQ(traces[0].operator_name, "op1");
    EXPECT_STREQ(traces[0].trace, "trace");
    EXPECT_EQ(traces[0].value, ref_t1);
    EXPECT_STREQ(traces[1].operator_name, "op2");
    EXPECT_STREQ(traces[1].trace, "trace");
    EXPECT_EQ(traces[1].value, ref_t2);
  }
}

}  // namespace test
}  // namespace c_api
}  // namespace dali
