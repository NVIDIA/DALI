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

#include <vector>
#include <utility>
#include "dali/core/convert.h"
#include "dali/kernels/alloc.h"
#include "dali/operators/random/rng_base_gpu.cuh"
#include "dali/operators/random/normal_distribution.h"

namespace dali {

class NormalDistributionGPU : public NormalDistribution<GPUBackend, NormalDistributionGPU> {
 public:
  template <typename T>
  struct Dist {
    using FloatType =
      typename std::conditional<
          ((std::is_integral<T>::value && sizeof(T) >= 4) || sizeof(T) > 4),
          double, float>::type;
    using type = curand_normal_dist<FloatType>;
    static constexpr bool has_state = true;
  };

  explicit NormalDistributionGPU(const OpSpec &spec)
      : NormalDistribution<GPUBackend, NormalDistributionGPU>(spec) {
    assert(max_batch_size_ < backend_data_.max_blocks_);
    dists_gpu_ = kernels::memory::alloc_unique<uint8_t>(
        kernels::AllocType::GPU, kDistMaxSize * max_batch_size_);
    dists_cpu_ = kernels::memory::alloc_unique<uint8_t>(
        kernels::AllocType::Pinned, kDistMaxSize * max_batch_size_);
  }

  ~NormalDistributionGPU() override = default;

  template <typename Dist>
  Dist* SetupDists(int nsamples, cudaStream_t stream) {
    assert(sizeof(Dist) * nsamples <= kDistMaxSize * max_batch_size_);
    auto *dists_cpu = reinterpret_cast<Dist*>(dists_cpu_.get());
    auto *dists_gpu = reinterpret_cast<Dist*>(dists_gpu_.get());
    for (int s = 0; s < nsamples; s++) {
      dists_cpu[s] = {mean_[s].data[0], stddev_[s].data[0]};
    }
    cudaMemcpyAsync(dists_gpu, dists_cpu,
      sizeof(Dist) * nsamples, cudaMemcpyHostToDevice, stream);
    return dists_gpu;
  }

 private:
  kernels::memory::KernelUniquePtr<uint8_t> dists_cpu_;
  kernels::memory::KernelUniquePtr<uint8_t> dists_gpu_;
  static constexpr size_t kDistMaxSize = sizeof(curand_normal_dist<double>);
};


DALI_REGISTER_OPERATOR(random__NormalDistribution, NormalDistributionGPU, GPU);
DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionGPU, GPU);

}  // namespace dali
