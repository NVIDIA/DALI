// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/cast.h"
#include "ndll/error_handling.h"

namespace ndll {

template <typename IType, typename OType>
__global__ void
BatchedCastKernel(OType * output, const IType * in, size_t N) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < N) {
    if (std::is_same<IType, ndll::float16>::value) {
      output[tid] = static_cast<OType>(static_cast<float>(in[tid]));
    } else {
      output[tid] = static_cast<OType>(in[tid]);
    }
  }
}

template <typename IType, typename OType>
NDLLError_t BatchedCast(OType * output,
                        const IType * input,
                        size_t N,
                        cudaStream_t stream) {
  NDLL_ASSERT(output != nullptr);
  NDLL_ASSERT(input != nullptr);
  const int threads = 512;
  const int blocks = (N + threads - 1)/threads;
  BatchedCastKernel<<<blocks, threads, 0, stream>>>(output, input, N);
  return NDLLSuccess;
}

template <typename Backend>
void Cast<Backend>::RunBatchedGPU(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto *output = ws->Output<GPUBackend>(idx);

  NDLLDataType itype = input.type().id();

  NDLL_TYPE_SWITCH(output_type_, OType,
    output->mutable_data<OType>();
    output->ResizeLike(input);
    NDLL_TYPE_SWITCH(itype, IType,
      NDLL_CALL(BatchedCast(
          output->mutable_data<OType>(),
          input.data<IType>(),
          input.size(),
          ws->stream()));););
}

template
void Cast<GPUBackend>::RunBatchedGPU(DeviceWorkspace *ws, const int idx);
template
void Cast<CPUBackend>::RunBatchedGPU(DeviceWorkspace *ws, const int idx);

}  // namespace ndll
