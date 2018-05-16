// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/color_twist.h"

namespace ndll {

NDLL_REGISTER_OPERATOR_FOR_DEVICE(ColorIntensity, GPU);
NDLL_REGISTER_OPERATOR_FOR_DEVICE(ColorOffset, GPU);
NDLL_REGISTER_OPERATOR_FOR_DEVICE(HueSaturation, GPU);
NDLL_REGISTER_OPERATOR_FOR_DEVICE(ColorContrast, GPU);

NDLLError_t BatchedColorTwist(const uint8 **in_batch, const NDLLSize *sizes,
                              uint8 **out_batch, int N, int C, colorTwistFunc func,
                              const Npp32f aTwist[][4], cudaStream_t s) {
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT(C == 1 || C == 3);
  NDLL_ASSERT(sizes != nullptr);

  if (!func)            // If transformation function is not defined,
    aTwist = NULL;      // we will not use twist matrix

  for (int i = 0; i < N; ++i) {
    NDLL_ASSERT(in_batch[i] != nullptr);
    NDLL_ASSERT(out_batch[i] != nullptr);

    const int nStep = sizes[i].width * C;
    if (aTwist)
      NDLL_CHECK_NPP(func(in_batch[i], nStep, out_batch[i], nStep, sizes[i], aTwist));
    else
      CUDA_CALL(cudaMemcpyAsync(out_batch[i], in_batch[i],
                           nStep * sizes[i].height, cudaMemcpyDefault, s));
  }

  return NDLLSuccess;
}

template <>
void ColorTwist<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  DataDependentSetupGPU(input, output, batch_size_, reshapeBatch(),
                        &input_ptrs_, &output_ptrs_, &sizes_);

  cudaStream_t old_stream = nppGetStream();
  cudaStream_t s = ws->stream();
  nppSetStream(s);
  float matr[4][4];
  NDLL_CALL(BatchedColorTwist((const uint8 **) input_ptrs_.data(), sizes_.data(),
                              output_ptrs_.data(), reshapeBatch() ? 1 : batch_size_, C_,
                              twistFunc_, twistMatr(matr) ? matr : NULL, s));
  nppSetStream(old_stream);
}

}  // namespace ndll
