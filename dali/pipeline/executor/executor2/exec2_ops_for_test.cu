// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/executor/executor2/exec2_ops_for_test.h"
#include <vector>
#include "dali/pipeline/operator/op_schema.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/common/scatter_gather.h"
#include "dali/core/span.h"

namespace dali {

DALI_SCHEMA(Exec2TestOp)  // DALI_SCHEMA can't take a macro :(
  .NumInput(0, 99)
  .NumOutput(1)
  .AddOptionalArg("delay", "[CPU-only] in milliseconds, to wait inside the operator's Run", 1.0f)
  .AddArg("addend", "a value added to the sum of inputs", DALI_INT32, true);

// DALI_REGISTER_OPERATOR can't take a macro for the name
DALI_REGISTER_OPERATOR(Exec2TestOp, exec2::test::DummyOpCPU, CPU);
DALI_REGISTER_OPERATOR(Exec2TestOp, exec2::test::DummyOpGPU, GPU);

DALI_SCHEMA(Exec2Counter)
  .NumInput(0)
  .NumOutput(1);

DALI_REGISTER_OPERATOR(Exec2Counter, exec2::test::CounterOp, CPU);

DALI_SCHEMA(Exec2Sink)
  .NumInput(0, 99)
  .NumOutput(0)
  .NoPrune();

DALI_REGISTER_OPERATOR(Exec2Sink, exec2::test::SinkOp, CPU);

namespace exec2 {
namespace test {

__global__ void Sum(int *out, const int **ins, int nbuf, int buf_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= buf_size)
    return;
  out[x] = x;
  for (int i = 0; i < nbuf; i++)
    out[x] += ins[i][x];
}

void DummyOpGPU::RunImpl(Workspace &ws) {
  kernels::DynamicScratchpad scratch(ws.stream());
  int N = ws.GetRequestedBatchSize(0);
  addend_.Acquire(spec_, ws, N);
  scratch.Allocate<mm::memory_kind::device, int>(N);
  auto addend_cpu = addend_.get();
  std::vector<const int *> pointers;

  std::vector<int> addend_cpu_cont(N);
  for (int i = 0; i < N; i++)
    addend_cpu_cont[i] = *addend_cpu[i].data;

  // This should go early - if we clobber some memory with that, we'll see it
  pointers.push_back(
    scratch.ToGPU(ws.stream(), make_cspan(addend_cpu_cont)));

  kernels::ScatterGatherGPU sg;
  for (int i = 0; i < ws.NumInput(); i++) {
    auto &inp = ws.Input<GPUBackend>(i);
    if (!inp.IsContiguousInMemory()) {
      int *cont = scratch.AllocateGPU<int>(N);
      for (int s = 0; s < N; s++)
        sg.AddCopy(cont + s, inp[s].data<int>(), sizeof(int));
      pointers.push_back(cont);
    } else {
      pointers.push_back(inp[0].data<int>());
    }
  }

  sg.Run(ws.stream());

  // The goal of this part is to introduce a delay in the GPU - and make the results come late
  size_t junk_size = 4 << 20;
  char *junk = scratch.Allocate<mm::memory_kind::device, char>(junk_size);
  for (int i = 0; i < 256; i++)
    CUDA_CALL(cudaMemsetAsync(junk, i, junk_size, ws.stream()));

  // After the delay, we can finally run the test body
  auto &out = ws.Output<GPUBackend>(0);
  if (!out.IsContiguousInMemory()) {
    out.Reset();
    out.SetContiguity(dali::BatchContiguity::Contiguous);
    out.Resize(uniform_list_shape(N, TensorShape<0>()), DALI_INT32);
    assert(out.IsContiguousInMemory());
  }
  Sum<<<div_ceil(N, 256), 256, 0, ws.stream()>>>(
    out[0].mutable_data<int>(),
    scratch.ToGPU(ws.stream(), pointers),
    ws.NumInput() + 1,
    N);
  CUDA_CALL(cudaGetLastError());
}


}  // namespace test
}  // namespace exec2
}  // namespace dali
