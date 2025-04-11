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

#ifndef DALI_C_API_2_PIPELINE_TEST_UTILS_H_
#define DALI_C_API_2_PIPELINE_TEST_UTILS_H_

#include "dali/c_api_2/test_utils.h"
#include "dali/c_api_2/pipeline.h"
#include "dali/c_api_2/managed_handle.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/pipeline.h"

namespace dali::c_api::test {

inline PipelineHandle Deserialize(std::string_view s, const daliPipelineParams_t &params) {
  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineDeserialize(&h, s.data(), s.length(), &params));
  return PipelineHandle(h);
}

inline TensorListHandle GetOutput(daliPipelineOutputs_h h, int idx) {
  daliTensorList_h tl = nullptr;
  CHECK_DALI(daliPipelineOutputsGet(h, &tl, idx));
  return TensorListHandle(tl);
}

inline PipelineOutputsHandle PopOutputs(daliPipeline_h h) {
  daliPipelineOutputs_h raw_out_h;
  CHECK_DALI(daliPipelinePopOutputs(h, &raw_out_h));
  return PipelineOutputsHandle(raw_out_h);
}

inline PipelineOutputsHandle PopOutputsAsync(daliPipeline_h h, cudaStream_t stream) {
  daliPipelineOutputs_h raw_out_h;
  CHECK_DALI(daliPipelinePopOutputsAsync(h, &raw_out_h, stream));
  return PipelineOutputsHandle(raw_out_h);
}

template <typename Backend>
auto &Unwrap(daliTensorList_h h) {
  return static_cast<ITensorList *>(h)->Unwrap<Backend>();
}

void CompareTensorLists(const TensorList<CPUBackend> &a, const TensorList<CPUBackend> &b) {
  ASSERT_EQ(a.type(), b.type());
  ASSERT_EQ(a.sample_dim(), b.sample_dim());
  ASSERT_EQ(a.num_samples(), b.num_samples());
  TYPE_SWITCH(a.type(), type2id, T,
    (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double),
    (
      Check(view<const T>(a), view<const T>(b));
    ), (GTEST_FAIL() << "Unsupported type " << a.type();));  // NOLINT
}

void CompareTensorLists(const TensorList<GPUBackend> &a, const TensorList<GPUBackend> &b) {
  TensorList<CPUBackend> a_cpu, b_cpu;
  a_cpu.set_order(AccessOrder::host());
  b_cpu.set_order(AccessOrder::host());
  a_cpu.Copy(a);
  b_cpu.Copy(b);
  CompareTensorLists(a_cpu, b_cpu);
}

void ComparePipelineOutput(Pipeline &ref, daliPipeline_h test) {
  Workspace ws;
  ws.set_output_order(AccessOrder::host());
  ref.Outputs(&ws);
  auto outs = PopOutputs(test);
  int out_count;
  ASSERT_EQ(daliPipelineGetOutputCount(test, &out_count), DALI_SUCCESS);
  ASSERT_EQ(ws.NumOutput(), out_count) << "The pipelines have a different number of outputs.";
  for (int i = 0; i < out_count; i++) {
    auto test_tl = GetOutput(outs, i);
    daliBufferPlacement_t placement{};
    CHECK_DALI(daliTensorListGetBufferPlacement(test_tl, &placement));
    if (ws.OutputIsType<CPUBackend>(i)) {
      ASSERT_EQ(placement.device_type, DALI_STORAGE_CPU);
      CompareTensorLists(ws.Output<CPUBackend>(i), *Unwrap<CPUBackend>(test_tl));
    } else if (ws.OutputIsType<GPUBackend>(i)) {
      ASSERT_EQ(placement.device_type, DALI_STORAGE_GPU);
      CompareTensorLists(ws.Output<GPUBackend>(i), *Unwrap<GPUBackend>(test_tl));
    }
  }
  ref.ReleaseOutputs();
}

void ComparePipelineOutputs(
      Pipeline &ref,
      daliPipeline_h test,
      int iters = 5,
      bool prefetch_on_first_iter = false) {
  for (int iter = 0; iter < iters; iter++) {
    if (iter == 0 && prefetch_on_first_iter) {
      ref.Prefetch();
      CHECK_DALI(daliPipelinePrefetch(test));
    } else {
      ref.Run();
      CHECK_DALI(daliPipelineRun(test));
    }
    ComparePipelineOutput(ref, test);
  }
}

}  // namespace dali::c_api::test

#endif  // DALI_C_API_2_PIPELINE_TEST_UTILS_H_
