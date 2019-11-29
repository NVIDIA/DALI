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

#include "dali/core/convert.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/operators/util/cast.h"

namespace dali {

template <typename OType, typename IType>
__global__ void
BatchedCastKernel(OType * output, const IType * in, size_t N) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    output[tid] = ConvertSat<OType>(in[tid]);
  }
}

template <typename OType, typename IType>
DALIError_t BatchedCast(OType * output,
                        const IType * input,
                        size_t N,
                        cudaStream_t stream) {
  DALI_ASSERT(output != nullptr);
  DALI_ASSERT(input != nullptr);
  const int threads = 512;
  const int blocks = (N + threads - 1)/threads;
  BatchedCastKernel<<<blocks, threads, 0, stream>>>(output, input, N);
  return DALISuccess;
}

template<>
void Cast<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  DALIDataType itype = input.type().id();
  TYPE_SWITCH(output_type_, type2id, OType, CAST_ALLOWED_TYPES, (
    output.mutable_data<OType>();
    output.ResizeLike(input);
    TYPE_SWITCH(itype, type2id, IType, CAST_ALLOWED_TYPES, (
      BatchedCast(output.mutable_data<OType>(), input.data<IType>(), input.size(), ws.stream());
    ), DALI_FAIL("Invalid input type"););  // NOLINT(whitespace/parens)
  ), DALI_FAIL("Invalid output type"););  // NOLINT(whitespace/parens)
}

DALI_REGISTER_OPERATOR(Cast, Cast<GPUBackend>, GPU);

}  // namespace dali
