// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <cuda_runtime.h>

#include "ndll/plugin/copy.h"
#include "ndll/error_handling.h"
#include "ndll/util/user_stream.h"

namespace ndll {

void CopyToExternalTensor(const Tensor<CPUBackend>& t, void* ptr) {
  NDLL_ENFORCE(t.ndim() > 0, "Can't copy empty Tensor!");
  std::memcpy(ptr,
              t.raw_data(),
              Product(t.shape()) * t.type().size());
}

void CopyToExternalTensor(const Tensor<GPUBackend>& t, void* ptr) {
  NDLL_ENFORCE(t.ndim() > 0, "Can't copy empty Tensor!");
  cudaStream_t stream = UserStream::Get()->GetStream(t);
  std::cout << ptr << std::endl;
  std::cout << t.raw_data() << std::endl;
  std::cout << Product(t.shape()) * t.type().size() << std::endl;
  CUDA_CALL(cudaMemcpyAsync(ptr,
                            t.raw_data(),
                            Product(t.shape()) * t.type().size(),
                            cudaMemcpyDeviceToDevice,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
}

void CopyToExternalTensor(TensorList<CPUBackend>& tl, void* ptr) {
  Tensor<CPUBackend> t;
  t.ShareData(&tl);
  CopyToExternalTensor(t, ptr);
}

void CopyToExternalTensor(TensorList<GPUBackend>& tl, void* ptr) {
  Tensor<GPUBackend> t;
  t.ShareData(&tl);
  CopyToExternalTensor(t, ptr);
}

}  // namespace ndll
