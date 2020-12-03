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

#ifndef DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_OP_CUH_
#define DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_OP_CUH_

#include <vector>
#include <utility>
#include "dali/kernels/alloc.h"
#include "dali/operators/random/normal_distribution_op.h"
#include "dali/operators/util/randomizer.cuh"

namespace dali {

namespace detail {
DLL_PUBLIC std::pair<std::vector<int>, int> DistributeBlocksPerSample(
  const TensorListShape<> &shape, int block_size, int max_blocks);
}

namespace mem = kernels::memory;

class NormalDistributionGpu : public NormalDistribution<GPUBackend> {
 public:
  struct BlockDesc {
    void *sample;
    int64_t start, end;
    float mean, std;
  };

  explicit NormalDistributionGpu(const OpSpec &spec);

  ~NormalDistributionGpu() override = default;

 protected:
  void RunImpl(workspace_t<GPUBackend> &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(NormalDistributionGpu);

 private:
  int SetupSingleValueDescs(TensorList<GPUBackend> &output, cudaStream_t stream);

  int SetupBlockDescs(TensorList<GPUBackend> &output, cudaStream_t stream);

  int SetupDescs(TensorList<GPUBackend> &output, cudaStream_t stream);

  void LaunchKernel(int blocks_num, int64_t elements, cudaStream_t stream);

  static constexpr int block_size_ = 256;
  static constexpr int max_blocks_ = 1024;
  mem::KernelUniquePtr<BlockDesc> block_descs_gpu_;
  mem::KernelUniquePtr<BlockDesc> block_descs_cpu_;
  curand_states randomizer_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_OP_CUH_
