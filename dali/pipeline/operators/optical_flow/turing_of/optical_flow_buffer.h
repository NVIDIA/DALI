// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_BUFFER_H_
#define DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_BUFFER_H_

#include <sstream>
#include "dali/error_handling.h"
#include "dali/pipeline/operators/optical_flow/turing_of/utils.h"
#include <nvOpticalFlowCuda.h>

namespace dali {
namespace optical_flow {


/**
 * Wrapper class for TuringOF API Buffer
 */
class OpticalFlowBuffer {
 public:
  OpticalFlowBuffer(NvOFHandle &of_handle, size_t width, size_t height,
                    NV_OF_CUDA_API_FUNCTION_LIST function_list,
                    NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT format) :
          turing_of_(function_list),
          descriptor_(GenerateBufferDescriptor(width, height, format, usage)) {
    // Buffer alloc
    TURING_OF_API_CALL(turing_of_.nvOFCreateGPUBufferCuda(of_handle, &descriptor_,
                                                          NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                                                          &handle_));
    ptr_ = turing_of_.nvOFGPUBufferGetCUdeviceptr(handle_);
    DALI_ENFORCE(ptr_ != 0, "Invalid pointer");

    // Assigning stride
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride_info;
    TURING_OF_API_CALL(turing_of_.nvOFGPUBufferGetStrideInfo(handle_, &stride_info));
    stride_ = {stride_info.strideInfo[0].strideXInBytes, stride_info.strideInfo[0].strideYInBytes};
  }


  OpticalFlowBuffer(const OpticalFlowBuffer &) = delete;

  void operator=(const OpticalFlowBuffer &) = delete;


  ~OpticalFlowBuffer() {
    auto err = turing_of_.nvOFDestroyGPUBufferCuda(handle_);
    if (err != NV_OF_SUCCESS) {
      // Failing to destroy GPU CUDA buffer leads to significant memory leak,
      // thus we'll rather terminate, than live with that memleak.
      std::terminate();
    }
  }


  NV_OF_BUFFER_DESCRIPTOR GetDescriptor() {
    return descriptor_;
  }


  NvOFGPUBufferHandle GetHandle() {
    return handle_;
  }


  CUdeviceptr GetPtr() {
    return ptr_;
  }


  struct Stride {
    size_t x, y;
  };


  Stride GetStride() {
    return stride_;
  }


 private:
  NV_OF_BUFFER_DESCRIPTOR
  GenerateBufferDescriptor(size_t width, size_t height, NV_OF_BUFFER_FORMAT format,
                           NV_OF_BUFFER_USAGE usage) {
    NV_OF_BUFFER_DESCRIPTOR ret;
    ret.height = static_cast<uint32_t>(height);
    ret.width = static_cast<uint32_t>(width);
    ret.bufferFormat = format;
    ret.bufferUsage = usage;
    return ret;
  }


  NV_OF_CUDA_API_FUNCTION_LIST turing_of_;
  NV_OF_BUFFER_DESCRIPTOR descriptor_;
  NvOFGPUBufferHandle handle_;
  CUdeviceptr ptr_;
  Stride stride_;
};

}  // namespace optical_flow
}  // namespace dali
#endif  // DALI_PIPELINE_OPERATORS_OPTICAL_FLOW_TURING_OF_OPTICAL_FLOW_BUFFER_H_
