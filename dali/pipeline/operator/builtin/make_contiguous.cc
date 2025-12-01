// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/nvtx.h"
#include "dali/pipeline/operator/builtin/make_contiguous.h"

namespace dali {

void MakeContiguousCPU::RunImpl(Workspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  DomainTimeRange tr("[DALI][MakeContiguousCPU] H2H", DomainTimeRange::kBlue);
  if (pass_through_) {
    output.ShareData(input);
  } else {
    int batch_size = input.num_samples();
    output.SetLayout(input.GetLayout());
    auto shapes = input.shape();

    auto &thread_pool = ws.GetThreadPool();
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      thread_pool.AddWork(
          [sample_id, &input, &output](int tid) {
            output.CopySample(sample_id, input, sample_id, AccessOrder::host());
          },
          shapes.tensor_size(sample_id));
    }
    thread_pool.RunAll();
  }
}

void MarkPassThrough(OperatorBase &op) {
  auto *make_contiguous_cpu = dynamic_cast<MakeContiguousBase<CPUBackend> *>(&op);
  if (make_contiguous_cpu) {
    make_contiguous_cpu->MarkPassThrough();
    return;
  }
  auto *make_contiguous_mixed = dynamic_cast<MakeContiguousBase<MixedBackend> *>(&op);
  if (make_contiguous_mixed) {
    make_contiguous_mixed->MarkPassThrough();
    return;
  }
  auto *make_contiguous_gpu = dynamic_cast<MakeContiguousBase<GPUBackend> *>(&op);
  if (make_contiguous_gpu) {
    make_contiguous_gpu->MarkPassThrough();
    return;
  }
  DALI_FAIL("This operation should be called only on MakeContiguous Operators.");
}

bool IsPassThrough(const OperatorBase &op) {
  const auto *make_contiguous_cpu = dynamic_cast<const MakeContiguousBase<CPUBackend> *>(&op);
  if (make_contiguous_cpu) {
    return make_contiguous_cpu->IsPassThrough();
  }
  const auto *make_contiguous_mixed = dynamic_cast<const MakeContiguousBase<MixedBackend> *>(&op);
  if (make_contiguous_mixed) {
    return make_contiguous_mixed->IsPassThrough();
  }
  const auto *make_contiguous_gpu = dynamic_cast<const MakeContiguousBase<GPUBackend> *>(&op);
  if (make_contiguous_gpu) {
    return make_contiguous_gpu->IsPassThrough();
  }
  DALI_FAIL("This operation should be called only on MakeContiguous Operators.");
}

bool SetMakeContiguousMode(OperatorBase &op, MakeContiguousMode mode) {
  if (auto *make_contiguous_cpu = dynamic_cast<MakeContiguousBase<CPUBackend> *>(&op)) {
    make_contiguous_cpu->SetMode(mode);
  } else if (auto *make_contiguous_mixed = dynamic_cast<MakeContiguousBase<MixedBackend> *>(&op)) {
    make_contiguous_mixed->SetMode(mode);
  } else if (auto *make_contiguous_gpu = dynamic_cast<MakeContiguousBase<GPUBackend> *>(&op)) {
    make_contiguous_gpu->SetMode(mode);
  } else {
    return false;
  }
  return true;
}

DALI_SCHEMA(MakeContiguous)
  .DocStr(R"code(Move input batch to a contiguous representation, more suitable for execution on the GPU)code")
  .NumInput(1)
  .InputDevice(0, InputDevice::MatchBackendOrCPU)
  .NumOutput(1)
  .MakeInternal();

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousCPU, CPU);

}  // namespace dali
