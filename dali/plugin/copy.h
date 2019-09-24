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

#ifndef DALI_PLUGIN_COPY_H_
#define DALI_PLUGIN_COPY_H_

#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

// this is copy from c_api.h
enum device_type_t {
  CPU = 0,
  GPU = 1
};

DLL_PUBLIC void CopyToExternalTensor(TensorList<CPUBackend>* tl,
                                     void* ptr,
                                     device_type_t dst_type,
                                     cudaStream_t stream = 0,
                                     bool non_blocking = false);
DLL_PUBLIC void CopyToExternalTensor(TensorList<GPUBackend>* tl,
                                     void* ptr,
                                     device_type_t dst_type,
                                     cudaStream_t stream = 0,
                                     bool non_blocking = false);
DLL_PUBLIC void CopyToExternalTensor(const Tensor<CPUBackend>& tl,
                                     void* ptr,
                                     device_type_t dst_type,
                                     cudaStream_t stream = 0,
                                     bool non_blocking = false);
DLL_PUBLIC void CopyToExternalTensor(const Tensor<GPUBackend>& tl,
                                     void* ptr,
                                     device_type_t dst_type,
                                     cudaStream_t stream = 0,
                                     bool non_blocking = false);

}  // namespace dali

#endif  // DALI_PLUGIN_COPY_H_
