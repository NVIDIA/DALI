// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_BASE_H_
#define DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_BASE_H_

#include <vector>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/pipeline/operators/op_spec.h"

namespace dali {

class ResizeBase {
 public:
  DLL_PUBLIC static void GetResamplingFilters(
      kernels::FilterDesc &min_filter,
      kernels::FilterDesc &mag_filter,
      const OpSpec &spec);

  struct KernelData {
    kernels::KernelContext context;
    kernels::KernelRequirements requirements;
    kernels::ScratchpadAllocator scratch_alloc;
  };
  std::vector<KernelData> kernel_data_;

  inline void Initialize(int num_threads = 1) {
    kernel_data_.resize(num_threads);
    out_shape_.resize(num_threads);
    resample_params_.resize(num_threads);
  }
  inline void InitializeGPU() {
    kernel_data_.resize(1);
  }

  inline KernelData &GetKernelData() {
    DALI_ENFORCE(!kernel_data_.empty(), "Resize kernel data not initialized");
    return kernel_data_.front();
  }
  inline KernelData &GetKernelData(int thread_idx) {
    DALI_ENFORCE(thread_idx >= 0 && static_cast<size_t>(thread_idx) < kernel_data_.size(),
                 "Thread index out of range");
    return kernel_data_[thread_idx];
  }

  DLL_PUBLIC void RunGPU(TensorList<GPUBackend> &output,
                         const TensorList<GPUBackend> &input,
                         cudaStream_t stream);

  DLL_PUBLIC void RunCPU(Tensor<CPUBackend> &output,
                         const Tensor<CPUBackend> &input,
                         int thread_idx);

  std::vector<kernels::ResamplingParams2D> resample_params_;
  std::vector<Dims> out_shape_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_BASE_H_
