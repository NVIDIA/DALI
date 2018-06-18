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
#include "dali/pipeline/util/device_guard.h"

namespace dali {

void CopyToExternalTensor(const Tensor<CPUBackend>& t, void* ptr) {
  DALI_ENFORCE(t.ndim() > 0, "Can't copy empty Tensor!");
  std::memcpy(ptr,
              t.raw_data(),
              Product(t.shape()) * t.type().size());
}

void CopyToExternalTensor(const Tensor<GPUBackend>& t, void* ptr) {
  DALI_ENFORCE(t.ndim() > 0, "Can't copy empty Tensor!");
  DeviceGuard d(t.device_id());
  cudaStream_t stream = UserStream::Get()->GetStream(t);
  CUDA_CALL(cudaMemcpyAsync(ptr,
                            t.raw_data(),
                            Product(t.shape()) * t.type().size(),
                            cudaMemcpyDeviceToDevice,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
}

void CopyToExternalTensor(TensorList<CPUBackend>* tl, void* ptr) {
  Tensor<CPUBackend> t;
  t.ShareData(tl);
  CopyToExternalTensor(t, ptr);
}

void CopyToExternalTensor(TensorList<GPUBackend>* tl, void* ptr) {
  Tensor<GPUBackend> t;
  t.ShareData(tl);
  CopyToExternalTensor(t, ptr);
}

}  // namespace dali
