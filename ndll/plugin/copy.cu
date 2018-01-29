// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <cuda_runtime.h>

#include "ndll/plugin/copy.h"
#include "ndll/error_handling.h"
#include "ndll/util/user_stream.h"

namespace ndll {

void CopyToExternalTensor(const TensorList<CPUBackend>& tl, void* ptr) {
  NDLL_ENFORCE(tl.IsTensor(),
      "All tensors in the TensorList must have the same shape to copy to external tensor.");
  NDLL_ENFORCE(tl.ntensor() > 0,
      "Can't copy empty TensorList.");
  std::memcpy(ptr,
              tl.raw_tensor(0),
              tl.ntensor() * Product(tl.tensor_shape(0)) * sizeof(tl.type().size()));
}

void CopyToExternalTensor(const TensorList<GPUBackend>& tl, void* ptr) {
  NDLL_ENFORCE(tl.IsTensor(),
      "All tensors in the TensorList must have the same shape to copy to external tensor.");
  NDLL_ENFORCE(tl.ntensor() > 0,
      "Can't copy empty TensorList.");
  cudaStream_t stream = UserStream::Get()->GetStream(tl);
  CUDA_CALL(cudaMemcpyAsync(ptr,
                            tl.raw_tensor(0),
                            tl.ntensor() * Product(tl.tensor_shape(0)) * sizeof(tl.type().size()),
                            cudaMemcpyDeviceToDevice,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
}

}  // namespace ndll
