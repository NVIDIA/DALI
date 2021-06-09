// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_NUMPY_READER_GPU_OP_H_
#define DALI_OPERATORS_READER_NUMPY_READER_GPU_OP_H_

#include <utility>
#include <string>
#include <vector>

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/numpy_loader_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/transpose/transpose_gpu.h"

namespace dali {

class NumpyReaderGPU : public DataReader<GPUBackend, ImageFileWrapperGPU> {
 public:
  explicit NumpyReaderGPU(const OpSpec& spec) :
    DataReader<GPUBackend, ImageFileWrapperGPU>(spec),
    thread_pool_(num_threads_, spec.GetArgument<int>("device_id"), false) {
    if (spec.ArgumentDefined("roi_start") || spec.ArgumentDefined("rel_roi_start") ||
        spec.ArgumentDefined("roi_end") || spec.ArgumentDefined("rel_roi_end") ||
        spec.ArgumentDefined("roi_shape") || spec.ArgumentDefined("rel_roi_shape") ||
        spec.ArgumentDefined("roi_axes")) {
      DALI_FAIL(
          "NumpyReader: Region-of-intereset reading is not yet supported for the GPU backend.");
    }

    prefetched_batch_tensors_.resize(prefetch_queue_depth_);

    // set a device guard
    DeviceGuard g(device_id_);

    // init loader
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<NumpyLoaderGPU>(spec, std::vector<string>(),
                                         shuffle_after_epoch);

    kmgr_.Resize<TransposeKernel>(1, 1);
  }

  ~NumpyReaderGPU() override {
    /*
     * Stop the prefetch thread as it uses the thread pool from this class. So before we can
     * destroy the thread pool make sure no one is using it anymore.
     */

    DataReader<GPUBackend, ImageFileWrapperGPU>::StopPrefetchThread();
  }

  // override prefetching here
  void Prefetch() override;

 protected:
  // we need to do the threading manually because gpu workspaces
  // do not have a thread pool
  ThreadPool thread_pool_;

  vector<TensorList<GPUBackend>> prefetched_batch_tensors_;

  // helpers for sample types and shapes
  TensorShape<> GetSampleShape(int sample_idx) {
    const auto& imfile = GetSample(sample_idx);
    return imfile.shape;
  }

  TypeInfo GetSampleType(int sample_idx) {
    const auto& imfile = GetSample(sample_idx);
    return imfile.type_info;
  }

  const void* GetSampleRawData(int sample_idx) {
    return prefetched_batch_tensors_[curr_batch_consumer_].raw_mutable_tensor(sample_idx);
  }

  bool CanInferOutputs() const override { return false; }

  void RunImpl(DeviceWorkspace &ws) override;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, ImageFileWrapperGPU);

  using TransposeKernel = kernels::TransposeGPU;
  kernels::KernelManager kmgr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_GPU_OP_H_
