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
#include <limits>
#include <random>
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
  Pipeline p(MakePipelineParams(max_batch_size, num_threads, device_id));
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
  Pipeline p(MakePipelineParams(max_batch_size, num_threads, device_id));
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

std::string GetPipelineWithExternalSource(
      StorageDevice dev_type,
      int max_batch_size,
      int num_threads,
      int device_id,
      bool extended_input_desc = false) {
  Pipeline p(max_batch_size, num_threads, device_id);
  OpSpec src = OpSpec("ExternalSource")
    .AddOutput("out", dev_type)
    .AddArg("device", dev_type == StorageDevice::CPU ? "cpu" : "gpu")
    .AddArg("name", "ext")
    .AddArg("batch_size", max_batch_size);
  if (extended_input_desc)
    src.AddArg("ndim", 3).AddArg("layout", "HWC").AddArg("dtype", DALI_UINT8);
  p.AddOperator(src, "ext");
  p.SetOutputDescs({ {"out", to_string(dev_type)} });
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

void TestPipelineRun(PipelineType ptype) {
  std::string proto;
  switch (ptype) {
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
  params.prefetch_queue_depths_present = true;
  params.prefetch_queue_depths = { 3, 3 };
  params.enable_checkpointing_present = true;
  params.enable_checkpointing = false;
  if (ptype == GPU2CPU) {
    params.exec_type_present = true;
    params.exec_type = DALI_EXEC_DYNAMIC;
  }

  PipelineHandle h = Deserialize(proto, params);
  ASSERT_NE(h, nullptr);

  EXPECT_EQ(daliPipelineRun(h), DALI_ERROR_INVALID_OPERATION);
  daliClearLastError();
  CHECK_DALI(daliPipelineBuild(h));

  int count;
  CHECK_DALI(daliPipelineGetOutputCount(h, &count));
  ASSERT_EQ(count, 2);
  daliPipelineIODesc_t desc{};
  EXPECT_EQ(daliPipelineGetOutputDesc(h, &desc, -1), DALI_ERROR_OUT_OF_RANGE);
  EXPECT_EQ(daliPipelineGetOutputDesc(h, &desc, count), DALI_ERROR_OUT_OF_RANGE);
  daliClearLastError();
  CHECK_DALI(daliPipelineGetOutputDesc(h, &desc, 0));
  EXPECT_STREQ(desc.name, "op1");
  EXPECT_EQ(desc.device, DALI_STORAGE_CPU);
  CHECK_DALI(daliPipelineGetOutputDesc(h, &desc, 1));
  EXPECT_STREQ(desc.name, "op2");
  EXPECT_EQ(desc.device, ptype == CPU2GPU ? DALI_STORAGE_GPU : DALI_STORAGE_CPU);

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
  // Skip this test when DALI_USE_EXEC2 is set
  const char *env = getenv("DALI_USE_EXEC2");
  if (env && atoi(env)) {
    GTEST_SKIP() << "This test cannot work when the use of Dynamic Executor is forced.";
  }

  auto proto = GetGPU2CPUPipelineProto(1, 4, 0);
  daliPipelineParams_t params{};
  params.max_batch_size_present = true;
  params.max_batch_size = 4;
  params.prefetch_queue_depths_present = true;
  params.prefetch_queue_depths = { 3, 3 };

  params.enable_checkpointing_present = true;
  params.enable_checkpointing = false;
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

TensorShape<> MakeRandomShape(
      std::mt19937_64 &rng,
      const TensorShape<> &min_shape,
      const TensorShape<> &max_shape) {
  assert(min_shape.size() == max_shape.size());
  TensorShape<> shape;
  shape.resize(min_shape.size());
  for (int i = 0; i < min_shape.size(); i++) {
    shape[i] = std::uniform_int_distribution<int>(min_shape[i], max_shape[i])(rng);
  }
  return shape;
}

TensorListShape<> MakeRandomShape(
      std::mt19937_64 &rng,
      const TensorShape<> &min_shape,
      const TensorShape<> &max_shape,
      int batch_size) {
  TensorListShape<> shape;
  shape.resize(batch_size, min_shape.size());
  for (int i = 0; i < batch_size; i++) {
    shape.set_tensor_shape(i, MakeRandomShape(rng, min_shape, max_shape));
  }
  return shape;
}

template <typename T>
void FillTensorList(TensorList<CPUBackend> &tl, std::mt19937_64 &rng) {
  auto dist = std::uniform_int_distribution<T>(0, std::numeric_limits<T>::max());
  for (int i = 0; i < tl.num_samples(); i++) {
    auto sample = tl[i];
    int64_t n = volume(sample.shape());
    auto buf = sample.mutable_data<T>();
    for (int64_t j = 0; j < n; j++) {
      buf[j] = dist(rng);
    }
  }
}

template <typename T, typename Backend>
void FillRandomTensorList(
      TensorList<Backend> &tl,
      std::mt19937_64 &rng,
      const TensorShape<> &min_shape,
      const TensorShape<> &max_shape,
      int batch_size) {
  auto shape = MakeRandomShape(rng, min_shape, max_shape, batch_size);
  if constexpr (std::is_same_v<Backend, CPUBackend>) {
    tl.Resize(shape, DALI_UINT8);
    FillTensorList<T>(tl, rng);
  } else {
    TensorList<CPUBackend> cpu_tl;
    cpu_tl.set_order(tl.order());
    cpu_tl.set_pinned(true);
    cpu_tl.Resize(shape, DALI_UINT8);
    FillTensorList<T>(cpu_tl, rng);
    tl.Copy(cpu_tl, tl.order());
  }
}

template <typename Backend>
void TestFeedInput(daliFeedInputFlags_t flags) {
  StorageDevice storage_backend = std::is_same_v<Backend, CPUBackend>
      ? StorageDevice::CPU : StorageDevice::GPU;
  auto proto = GetPipelineWithExternalSource(storage_backend, 4, 4, 0);
  daliPipelineParams_t params{};
  params.max_batch_size_present = true;
  params.max_batch_size = 8;
  params.prefetch_queue_depths_present = true;
  params.prefetch_queue_depths = { 3, 3 };
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;
  auto h = Deserialize(proto, params);
  ASSERT_NE(h, nullptr);
  CHECK_DALI(daliPipelineBuild(h));
  int count = 0;
  EXPECT_EQ(daliPipelineGetFeedCount(h, &count, "nonexistent"), DALI_ERROR_INVALID_KEY);
  daliClearLastError();
  CHECK_DALI(daliPipelineGetFeedCount(h, &count, "ext"));
  ASSERT_EQ(count, params.prefetch_queue_depths.cpu)
    << "Feed count should be equal to prefetch queue depth.";

  std::mt19937_64 rng(1234);
  std::uniform_int_distribution<int> bs_dist(1, params.max_batch_size);

  CUDAStreamLease stream1 = CUDAStreamPool::instance().Get();
  CUDAStreamLease stream2;
  if (flags & DALI_FEED_INPUT_SYNC)
    stream2 = CUDAStreamPool::instance().Get();

  std::vector<std::shared_ptr<TensorList<Backend>>> cpp_tls;
  for (int i = 0; i < count; i++) {
    std::shared_ptr<TensorList<Backend>> cpp_tl = std::make_shared<TensorList<Backend>>();
    cpp_tls.push_back(cpp_tl);
    cpp_tl->set_order(stream1.get());

    int bs = bs_dist(rng);
    FillRandomTensorList<uint8_t>(*cpp_tl, rng, { 240, 320, 1 }, { 1080, 1920, 3 }, bs);

    auto tl = Wrap(cpp_tl);
    daliTensorList_h tl_h = tl.get();
    cudaStream_t s = stream2 ? stream2.get() : stream1.get();
    EXPECT_EQ(daliPipelineFeedInput(h, "nonexistent", tl_h, "", flags, &s), DALI_ERROR_INVALID_KEY);
    daliClearLastError();
    CHECK_DALI(daliPipelineFeedInput(h, "ext", tl_h, "data", flags, &s));
  }
  CUDA_CALL(cudaDeviceSynchronize());
  CHECK_DALI(daliPipelinePrefetch(h));
  for (int i = 0; i < count; i++) {
    auto outs = PopOutputsAsync(h, stream1.get());
    ASSERT_NE(outs, nullptr);
    auto out_tl = GetOutput(outs, 0);
    CompareTensorLists(*cpp_tls[i], *Unwrap<Backend>(out_tl));
  }
}

TEST(CAPI2_PipelineTest, FeedInputCPU) {
  TestFeedInput<CPUBackend>({});
}

TEST(CAPI2_PipelineTest, FeedInputGPUSync) {
  TestFeedInput<GPUBackend>(DALI_FEED_INPUT_SYNC);
}

TEST(CAPI2_PipelineTest, FeedInputGPUAsync) {
  TestFeedInput<GPUBackend>({});
}

TEST(CAPI2_PipelineTest, InputDescSimple) {
  auto proto = GetPipelineWithExternalSource(dali::StorageDevice::GPU, 4, 4, 0, false);
  daliPipelineParams_t params{};
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;

  auto h = Deserialize(proto, params);
  CHECK_DALI(daliPipelineBuild(h));
  int count = 0;
  CHECK_DALI(daliPipelineGetInputCount(h, &count));
  ASSERT_EQ(count, 1);
  daliPipelineIODesc_t desc{};
  CHECK_DALI(daliPipelineGetInputDescByIdx(h, &desc, 0));
  EXPECT_EQ(desc.device, DALI_STORAGE_GPU);
  EXPECT_STREQ(desc.name, "ext");
  EXPECT_FALSE(desc.ndim_present);
  EXPECT_FALSE(desc.dtype_present);
  EXPECT_EQ(desc.layout, nullptr);
}

TEST(CAPI2_PipelineTest, InputDescExtended) {
  auto proto = GetPipelineWithExternalSource(dali::StorageDevice::CPU, 4, 4, 0, true);
  daliPipelineParams_t params{};
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;

  auto h = Deserialize(proto, params);
  CHECK_DALI(daliPipelineBuild(h));
  int count = 0;
  CHECK_DALI(daliPipelineGetInputCount(h, &count));
  ASSERT_EQ(count, 1);
  daliPipelineIODesc_t desc{};
  CHECK_DALI(daliPipelineGetInputDescByIdx(h, &desc, 0));
  EXPECT_EQ(desc.device, DALI_STORAGE_CPU);
  EXPECT_STREQ(desc.name, "ext");
  EXPECT_TRUE(desc.ndim_present);
  EXPECT_EQ(desc.ndim, 3);
  EXPECT_TRUE(desc.dtype_present);
  EXPECT_EQ(desc.dtype, DALI_UINT8);
  EXPECT_STREQ(desc.layout, "HWC");
}

}  // namespace test
}  // namespace c_api
}  // namespace dali
