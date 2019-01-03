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

#include "dali/pipeline/operators/util/cast.h"
#include "dali/error_handling.h"
#include "dali/util/cuda_utils.h"

namespace dali {

template <typename IType, typename OType>
__global__ void
BatchedCastKernel(OType * output, const IType * in, size_t N) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    if (std::is_same<IType, dali::float16>::value) {
      output[tid] = static_cast<OType>(static_cast<float>(in[tid]));
    } else {
      output[tid] = static_cast<OType>(in[tid]);
    }
  }
}

template <typename IType, typename OType>
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
void Cast<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto *output = ws->Output<GPUBackend>(idx);

  DALIDataType itype = input.type().id();

  DALI_TYPE_SWITCH(output_type_, OType,
      output->mutable_data<OType>();
      output->ResizeLike(input);
      DALI_TYPE_SWITCH(itype, IType,
        DALI_CALL(BatchedCast(
            output->mutable_data<OType>(),
            input.data<IType>(),
            input.size(),
            ws->stream()));););
}

DALI_REGISTER_OPERATOR(Cast, Cast<GPUBackend>, GPU);

}  // namespace dali
