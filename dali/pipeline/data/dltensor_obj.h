// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_DLTENSOR_OBJ_H_
#define DALI_PIPELINE_DATA_DLTENSOR_OBJ_H_

#include <pybind11/pybind11.h>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include "third_party/dlpack/include/dlpack/dlpack.h"

#include "dali/core/common.h"
#include "dali/core/cuda_shared_event.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/dltensor.h"

namespace dali {

/**
 * @brief Stream-aware wrapper for DLManagedTensor.
 *
 * Provides additional layer over usual `py::capsule` that implements
 * `__dlpack__` and `__dlpack_device__` methods, for stream-aware ownership
 * transfer.
 *
 */
class DLL_PUBLIC DLTensorObj {
 public:
  DLL_PUBLIC DLTensorObj(DLMTensorPtr ptr, std::optional<cudaStream_t> producer_stream)
      : dlm_ptr_{std::move(ptr)} {
    DALI_ENFORCE(dlm_ptr_, "Expected non-null pointer for managed DLTensor");
    device_id_ = dlm_ptr_->dl_tensor.device.device_id;
    device_type_ = dlm_ptr_->dl_tensor.device.device_type;
    DALI_ENFORCE(device_type_ == kDLCPU || device_type_ == kDLCUDAHost || device_type_ == kDLCUDA,
                 "Currently only DLCPU and DLGPU device types are supported");
    if (producer_stream) {
      DALI_ENFORCE(device_type_ == kDLCUDA || device_type_ == kDLCUDAHost,
                   "Stream-aware DLTensorObj supports only CUDA and CUDA host device type.");
      auto &pool = CUDAEventPool::instance();
      data_ready_ = CUDASharedEvent::GetFromPool(device_id_);
      CUDA_CALL(cudaEventRecord(data_ready_, *producer_stream));
    }
  }

  DLL_PUBLIC DLTensorObj(DLMTensorPtr ptr, CUDASharedEvent event)
      : dlm_ptr_{std::move(ptr)} {
    DALI_ENFORCE(dlm_ptr_, "Expected non-null pointer for managed DLTensor");
    device_id_ = dlm_ptr_->dl_tensor.device.device_id;
    device_type_ = dlm_ptr_->dl_tensor.device.device_type;
    DALI_ENFORCE(device_type_ == kDLCPU || device_type_ == kDLCUDAHost || device_type_ == kDLCUDA,
                 "Currently only DLCPU and DLGPU device types are supported");
    if (event) {
      DALI_ENFORCE(device_type_ == kDLCUDA || device_type_ == kDLCUDAHost,
                   "Stream-aware DLTensorObj supports only CUDA and CUDA host device type.");
    }
    data_ready_ = std::move(event);
  }

  DLTensorObj(DLTensorObj &) = delete;
  DLL_PUBLIC DLTensorObj(DLTensorObj &&) = default;

  DLL_PUBLIC std::tuple<int, int> dlpack_device() {
    return {device_type_, device_id_};
  }

  DLL_PUBLIC DLManagedTensor *dlpack(std::optional<cudaStream_t> consumer_stream) {
    DALI_ENFORCE(dlm_ptr_, "The dlpack object was already consumed");
    if (consumer_stream) {
      CUDA_CALL(cudaStreamWaitEvent(*consumer_stream, data_ready_, 0));
    }
    return dlm_ptr_.release();
  }

 private:
  DLMTensorPtr dlm_ptr_;
  int device_id_;
  DLDeviceType device_type_;
  CUDASharedEvent data_ready_;
};


}  // namespace dali
#endif  // DALI_PIPELINE_DATA_DLTENSOR_OBJ_H_
