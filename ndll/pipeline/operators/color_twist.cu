// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/color_twist.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(ColorIntensity, ColorIntensity<GPUBackend>, GPU);
NDLL_REGISTER_OPERATOR(ColorOffset, ColorOffset<GPUBackend>, GPU);
NDLL_REGISTER_OPERATOR(HueSaturation, HueSaturation<GPUBackend>, GPU);
NDLL_REGISTER_OPERATOR(ColorContrast, ColorContrast<GPUBackend>, GPU);

template <>
void ColorTwist<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  DataDependentSetupGPU(input, output, batch_size_, reshapeBatch(),
                        &input_ptrs_, &output_ptrs_, &sizes_);

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws->stream());
  float matr[4][4];
  NDLL_CALL(BatchedColorTwist((const uint8 **) input_ptrs_.data(), sizes_.data(),
                              output_ptrs_.data(), reshapeBatch() ? 1 : batch_size_, C_,
                              twistFunc_, twistMatr(matr) ? matr : NULL));
  nppSetStream(old_stream);
}

}  // namespace ndll
