// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <string>
#include <vector>

#include "dali/core/cuda_event_pool.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/common/scatter_gather.h"
#include "dali/util/cufile_helper.h"
#include "dali/kernels/transpose/transpose_gpu.h"
#include "dali/operators/reader/loader/numpy_loader_gpu.h"
#include "dali/operators/reader/numpy_reader_op.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

namespace gds {
class GDSLazyInit {
 protected:
  void InitDriverScope() {
    driver_scope_ = std::make_unique<cufile::CUFileDriverScope>();
  }
  std::unique_ptr<cufile::CUFileDriverScope> driver_scope_;
};
}  // namespace gds

// GDSLazyInit needs to be a base class, so that it destroyed last, after the loader
class NumpyReaderGPU : gds::GDSLazyInit, public NumpyReader<GPUBackend, NumpyFileWrapperGPU> {
 public:
  explicit NumpyReaderGPU(const OpSpec& spec);

  ~NumpyReaderGPU() override;

  // override prefetching here
  void Prefetch() override;

 protected:
  // we need to do the threading manually because gpu workspaces
  // do not have a thread pool
  ThreadPool thread_pool_;

  vector<TensorList<GPUBackend>> prefetched_batch_tensors_;

  template <typename T, int Dims>
  TensorListView<StorageGPU, const T, Dims> GetCurrBatchView() {
    return view<const T, Dims>(prefetched_batch_tensors_[curr_batch_consumer_]);
  }

  template <typename T, int Dims>
  void RunImplTyped(Workspace &ws);
  void RunImpl(Workspace &ws) override;
  using Operator<GPUBackend>::RunImpl;


  USE_READER_OPERATOR_MEMBERS(GPUBackend, NumpyFileWrapperGPU, NumpyFileWrapperGPU, true);

 private:
  using TransposeKernel = kernels::TransposeGPU;
  kernels::ScatterGatherGPU sg_;
  kernels::KernelManager kmgr_transpose_;
  kernels::KernelManager kmgr_slice_;
  TensorListShape<> tmp_buf_sh_;
  TensorList<GPUBackend> tmp_buf_;

  void ScheduleChunkedRead(SampleView<GPUBackend> &out_sample, NumpyFileWrapperGPU &target);

  size_t chunk_size_ = gds::GetGDSChunkSize();
  detail::NumpyHeaderCache header_cache_;
  gds::GDSStagingEngine staging_;
  CUDAStreamLease staging_stream_;
  CUDAEvent staging_ready_;
  std::vector<int> source_data_index_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_GPU_OP_H_
