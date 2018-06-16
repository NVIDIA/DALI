// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include <cuda_runtime_api.h>
#include "dali/pipeline/operators/util/copy.h"

namespace dali {

template<>
void Copy<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);
  output->set_type(input.type());
  output->ResizeLike(input);
  CUDA_CALL(cudaMemcpyAsync(
          output->raw_mutable_data(),
          input.raw_data(),
          input.nbytes(),
          cudaMemcpyDeviceToDevice,
          ws->stream()));
}

DALI_REGISTER_OPERATOR(Copy, Copy<GPUBackend>, GPU);

}  // namespace dali

