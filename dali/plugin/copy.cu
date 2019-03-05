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


#include <cuda_runtime.h>

#include "dali/plugin/copy.h"
#include "dali/error_handling.h"
#include "dali/util/user_stream.h"
#include "dali/util/device_guard.h"

namespace dali {

template <typename T>
static void CopyToExternalTensorHelper(const dali::Buffer<T> &src, void *dst,
                                       device_type_t dst_type, size_t num) {}

template <>
void CopyToExternalTensorHelper<CPUBackend>(const dali::Buffer<CPUBackend> &src, void *dst,
                                            device_type_t dst_type, size_t num) {
  if (dst_type == CPU) {
    std::memcpy(dst, src.raw_data(), num);
  } else {
    DALI_FAIL("Coping from CPUBackend to device type " + to_string(dst_type));
  }
}

template <>
void CopyToExternalTensorHelper<GPUBackend>(const dali::Buffer<GPUBackend> &src, void *dst,
                                            device_type_t dst_type, size_t num) {
  DeviceGuard d(src.device_id());
  cudaMemcpyKind direction;
  cudaStream_t stream = UserStream::Get()->GetStream(src);
  if (dst_type == GPU) {
    direction = cudaMemcpyDeviceToDevice;
  } else if (dst_type == CPU) {
    direction = cudaMemcpyDeviceToHost;
  } else {
    DALI_FAIL("Coping from GPUBackend to device type " + to_string(dst_type));
  }
  CUDA_CALL(cudaMemcpyAsync(dst,
                            src.raw_data(),
                            num,
                            direction,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
}

template <typename T>
static void CopyToExternalTensorListHelper(TensorList<T>* tl, void* ptr,
                                           device_type_t dst_type) {
  if (tl->IsDenseTensor()) {
    Tensor<T> t;
    t.ShareData(tl);
    CopyToExternalTensor(t, ptr, dst_type);
  } else {
    CopyToExternalTensorHelper<T>(*tl, ptr, dst_type, tl->nbytes());
  }
}

void CopyToExternalTensor(const Tensor<CPUBackend>& t, void* ptr,
                          device_type_t dst_type) {
  DALI_ENFORCE(t.ndim() > 0, "Can't copy empty Tensor!");
  CopyToExternalTensorHelper<CPUBackend>(t, ptr, dst_type,
                                         volume(t.shape()) * t.type().size());
}

void CopyToExternalTensor(const Tensor<GPUBackend>& t, void* ptr,
                          device_type_t dst_type) {
  DALI_ENFORCE(t.ndim() > 0, "Can't copy empty Tensor!");
  CopyToExternalTensorHelper<GPUBackend>(t, ptr, dst_type,
                                         volume(t.shape()) * t.type().size());
}
void CopyToExternalTensor(TensorList<CPUBackend>* tl, void* ptr,
                          device_type_t dst_type) {
  CopyToExternalTensorListHelper<CPUBackend>(tl, ptr, dst_type);
}

void CopyToExternalTensor(TensorList<GPUBackend>* tl, void* ptr,
                          device_type_t dst_type) {
  CopyToExternalTensorListHelper<GPUBackend>(tl, ptr, dst_type);
}

}  // namespace dali
