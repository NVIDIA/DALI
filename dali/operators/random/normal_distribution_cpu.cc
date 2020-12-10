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

#include "dali/operators/random/rng_base_cpu.h"
#include "dali/operators/random/normal_distribution.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

DALI_SCHEMA(random__NormalDistribution)
    .DocStr(R"code(Generates random numbers following a normal distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single value per
sample is generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg<float>("mean",
      R"code(Mean of the distribution.)code",
      0.f, true)
    .AddOptionalArg<float>("stddev",
      R"code(Standard deviation of the distribution.)code",
      1.f, true)
    .AddParent("RNGAttr");

class NormalDistributionCPU : public NormalDistribution<CPUBackend, NormalDistributionCPU> {
 public:
  template <typename T>
  struct Dist {
    using FloatType =
      typename std::conditional<
          ((std::is_integral<T>::value && sizeof(T) >= 4) || sizeof(T) > 4),
          double, float>::type;
    using type = std::normal_distribution<FloatType>;
  };

  explicit NormalDistributionCPU(const OpSpec &spec)
      : NormalDistribution<CPUBackend, NormalDistributionCPU>(spec) {
    dist_data_.resize(max_batch_size_ * kDistMaxSize);
  }
  ~NormalDistributionCPU() override = default;

  template <typename Dist>
  Dist* SetupDists(int nsamples) {
    assert(sizeof(Dist) * nsamples <= dist_data_.size());
    auto dists = reinterpret_cast<Dist*>(dist_data_.data());
    for (int s = 0; s < nsamples; s++) {
      dists[s] = Dist(mean_[s].data[0], stddev_[s].data[0]);
    }
    return dists;
  }

 private:
  using Operator<CPUBackend>::max_batch_size_;
  using NormalDistribution<CPUBackend, NormalDistributionCPU>::mean_;
  using NormalDistribution<CPUBackend, NormalDistributionCPU>::stddev_;
  std::vector<uint8_t> dist_data_;
  static constexpr size_t kDistMaxSize = sizeof(std::normal_distribution<double>);
};


DALI_REGISTER_OPERATOR(random__NormalDistribution, NormalDistributionCPU, CPU);

// Deprecated alias
DALI_SCHEMA(NormalDistribution)
    .DocStr(R"code(Generates random numbers following a normal distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single value per
sample is generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg<float>("mean",
      R"code(Mean of the distribution.)code",
      0.f, true)
    .AddOptionalArg<float>("stddev",
      R"code(Standard deviation of the distribution.)code",
      1.f, true)
    .AddParent("random__NormalDistribution")
    .Deprecate("random.NormalDistribution");  // Deprecated in 0.30


DALI_REGISTER_OPERATOR(NormalDistribution, NormalDistributionCPU, CPU);

}  // namespace dali
