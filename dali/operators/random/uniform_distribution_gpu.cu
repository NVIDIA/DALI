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
#include "dali/kernels/common/scatter_gather.h"
#include "dali/core/tensor_shape.h"
#include "dali/operators/random/rng_base_gpu.cuh"
#include "dali/operators/random/uniform_distribution.h"

namespace dali {

class UniformDistributionGPU : public UniformDistribution<GPUBackend, UniformDistributionGPU> {
 public:
  template <typename T>
  struct Dist {
    using FloatType =
      typename std::conditional<
          ((std::is_integral<T>::value && sizeof(T) > 3) || sizeof(T) > 4),
          double, float>::type;
    using type = curand_uniform_dist<FloatType>;
    static constexpr bool has_state = true;
  };

  using DistDiscrete = curand_uniform_int_values_dist<float>;

  explicit UniformDistributionGPU(const OpSpec &spec)
      : UniformDistribution<GPUBackend, UniformDistributionGPU>(spec) {
    assert(max_batch_size_ < backend_specific_.max_blocks_);
    dists_gpu_ = kernels::memory::alloc_unique<uint8_t>(
        kernels::AllocType::GPU, kDistMaxSize * max_batch_size_);
    dists_cpu_ = kernels::memory::alloc_unique<uint8_t>(
        kernels::AllocType::Pinned, kDistMaxSize * max_batch_size_);

    if (values_.IsDefined())
      sg_ = kernels::ScatterGatherGPU(1<<18, spec.GetArgument<int>("max_batch_size"));
  }

  ~UniformDistributionGPU() override = default;

  template <typename Dist>
  Dist* SetupDists(int nsamples, cudaStream_t stream) {
    assert(sizeof(Dist) * nsamples <= kDistMaxSize * max_batch_size_);
    auto *dists_cpu = reinterpret_cast<Dist*>(dists_cpu_.get());
    auto *dists_gpu = reinterpret_cast<Dist*>(dists_gpu_.get());
    for (int s = 0; s < nsamples; s++) {
      dists_cpu[s] = {range_[s].data[0], range_[s].data[1]};
    }
    cudaMemcpyAsync(dists_gpu, dists_cpu,
      sizeof(Dist) * nsamples, cudaMemcpyHostToDevice, stream);
    return dists_gpu;
  }

 private:
  kernels::memory::KernelUniquePtr<uint8_t> dists_cpu_;
  kernels::memory::KernelUniquePtr<uint8_t> dists_gpu_;
  TensorList<GPUBackend> values_gpu_;

  static constexpr size_t kSzC = sizeof(curand_uniform_dist<double>);  // max continuous dist size
  static constexpr size_t kSzD = sizeof(curand_uniform_int_values_dist<float>);  // max discrete dist size
  static constexpr size_t kDistMaxSize = std::max(kSzC, kSzD);
  kernels::ScatterGatherGPU sg_;
};

template <>
UniformDistributionGPU::DistDiscrete*
UniformDistributionGPU::SetupDists<UniformDistributionGPU::DistDiscrete>(int nsamples, cudaStream_t stream) {
  using Dist = typename UniformDistributionGPU::DistDiscrete;
  const auto &values_view = values_.get();
  values_gpu_.Resize(values_view.shape, TypeTable::GetTypeInfo(DALI_FLOAT));

  assert(sizeof(Dist) * nsamples <= kDistMaxSize * max_batch_size_);
  auto *dists_cpu = reinterpret_cast<Dist*>(dists_cpu_.get());
  auto *dists_gpu = reinterpret_cast<Dist*>(dists_gpu_.get());
  for (int s = 0; s < nsamples; s++) {
    int64_t nvalues = values_view.shape.tensor_size(s);
    auto values_gpu_ptr = reinterpret_cast<float*>(values_gpu_.raw_mutable_tensor(s));
    sg_.AddCopy(values_gpu_ptr, values_view.data[s], nvalues * sizeof(float));
    dists_cpu[s] = {values_gpu_ptr, nvalues};
  }
  sg_.Run(stream);
  cudaMemcpyAsync(dists_gpu, dists_cpu,
    sizeof(Dist) * nsamples, cudaMemcpyHostToDevice, stream);
  return dists_gpu;
}


DALI_REGISTER_OPERATOR(random__UniformDistribution, UniformDistributionGPU, GPU);

}  // namespace dali
