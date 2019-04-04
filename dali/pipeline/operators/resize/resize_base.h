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

class DLL_PUBLIC ResamplingFilterAttr {
 public:
  DLL_PUBLIC ResamplingFilterAttr(const OpSpec &spec);

  /// Filter used when downscaling
  kernels::FilterDesc min_filter_{ kernels::ResamplingFilterType::Triangular, 0 };
  /// Filter used when upscaling
  kernels::FilterDesc mag_filter_{ kernels::ResamplingFilterType::Linear, 0 };
  /// Initial size, in bytes, of a temporary buffer for resampling.
  size_t temp_buffer_hint_ = 0;
};

class DLL_PUBLIC ResizeBase : public ResamplingFilterAttr {
 public:
  explicit inline ResizeBase(const OpSpec &spec) : ResamplingFilterAttr(spec) {}

  DLL_PUBLIC void Initialize(int num_threads = 1);
  DLL_PUBLIC void InitializeGPU(int batch_size, int minibatch_size);

  DLL_PUBLIC void RunGPU(TensorList<GPUBackend> &output,
                         const TensorList<GPUBackend> &input,
                         cudaStream_t stream);

  DLL_PUBLIC void RunCPU(Tensor<CPUBackend> &output,
                         const Tensor<CPUBackend> &input,
                         int thread_idx);

  std::vector<kernels::ResamplingParams2D> resample_params_;
  std::vector<Dims> out_shape_;

 private:
  struct KernelData {
    kernels::KernelContext context;
    kernels::KernelRequirements requirements;
    kernels::ScratchpadAllocator scratch_alloc;
  };

  std::vector<KernelData> kernel_data_;
  struct MiniBatch {
    int start, count;
    std::vector<Dims> out_shape;
    kernels::InListGPU<uint8_t, 3> input;
    kernels::OutListGPU<uint8_t, 3> output;
  };
  std::vector<MiniBatch> minibatches_;

  void SubdivideInput(const kernels::InListGPU<uint8_t, 3> &in);
  void SubdivideOutput(const kernels::OutListGPU<uint8_t, 3> &out);

  inline KernelData &GetKernelData(int thread_idx) {
    DALI_ENFORCE(thread_idx >= 0 && static_cast<size_t>(thread_idx) < kernel_data_.size(),
                 "Thread index out of range");
    return kernel_data_[thread_idx];
  }

  inline kernels::ScratchpadAllocator &GetGPUScratchAlloc() {
    DALI_ENFORCE(!kernel_data_.empty(), "Resize kernel data not initialized");
    return kernel_data_[0].scratch_alloc;
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_BASE_H_
