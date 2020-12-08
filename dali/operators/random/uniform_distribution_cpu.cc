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
#include "dali/operators/random/uniform_distribution.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {

namespace {

constexpr float kDefRange[] = {-1.0f, 1.0f};

}  // namespace

DALI_SCHEMA(random__UniformDistribution)
    .DocStr(R"code(Generates random numbers following a uniform distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single number is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("range",
      R"code(Range [min, max) of a continuous uniform distribution.

This argument is mutually exclusive with ``values``.)code",
      std::vector<float>{-1.0f, 1.0f}, true)
    .AddOptionalArg<std::vector<float>>("values",
      R"code(The discrete values produced by a discrete uniform distribution.

This argument is mutually exclusive with ``range``.)code",
      nullptr, true)
    .AddParent("RNGAttr");

class UniformDistributionCPU : public UniformDistribution<CPUBackend, UniformDistributionCPU> {
 public:
  template <typename T>
  struct Dist {
    using FloatType =
      typename std::conditional<
          ((std::is_integral<T>::value && sizeof(T) > 3) || sizeof(T) > 4),
          double, float>::type;
    using type = std::uniform_real_distribution<FloatType>;
  };

  struct DistDiscrete {
    explicit DistDiscrete(span<const float> values) 
        : dist_(0, values.size() - 1), values_(values) {}

    template <typename Generator>
    float operator()(Generator& g) {
      return values_[dist_(g)];
    }

    std::uniform_int_distribution<int> dist_;
    span<const float> values_;
  };

  explicit UniformDistributionCPU(const OpSpec &spec)
      : UniformDistribution<CPUBackend, UniformDistributionCPU>(spec) {
    dist_data_.resize(max_batch_size_ * kDistMaxSize);
  }
  ~UniformDistributionCPU() override = default;

  template <typename Dist>
  Dist* SetupDists(int nsamples) {
    assert(sizeof(Dist) * nsamples <= dist_data_.size());
    auto dists = reinterpret_cast<Dist*>(dist_data_.data());
    for (int s = 0; s < nsamples; s++) {
      dists[s] = Dist(range_[s].data[0], range_[s].data[1]);
    }
    return dists;
  }

 private:
  using Operator<CPUBackend>::max_batch_size_;
  using UniformDistribution<CPUBackend, UniformDistributionCPU>::range_;
  using UniformDistribution<CPUBackend, UniformDistributionCPU>::values_;
  std::vector<uint8_t> dist_data_;
  static constexpr size_t kSzC = sizeof(std::uniform_real_distribution<double>);  // max continuous dist size
  static constexpr size_t kSzD = sizeof(DistDiscrete);  // max discrete dist size
  static constexpr size_t kDistMaxSize = std::max(kSzC, kSzD);
};

template <>
UniformDistributionCPU::DistDiscrete*
UniformDistributionCPU::SetupDists<UniformDistributionCPU::DistDiscrete>(int nsamples) {
  using Dist = typename UniformDistributionCPU::DistDiscrete;
  assert(sizeof(Dist) * nsamples <= dist_data_.size());
  auto dists = reinterpret_cast<Dist*>(dist_data_.data());
  for (int s = 0; s < nsamples; s++) {
    dists[s] = Dist(span<const float>(values_[s].data, volume(values_[s].shape)));
  }
  return dists;
}

DALI_REGISTER_OPERATOR(random__UniformDistribution, UniformDistributionCPU, CPU);

// Deprecated alias
DALI_SCHEMA(Uniform)
    .DocStr(R"code(Generates random numbers following a uniform distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single number is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("range",
      R"code(Range [min, max) of a continuous uniform distribution.

This argument is mutually exclusive with ``values``.)code",
      std::vector<float>{-1.0f, 1.0f}, true)
    .AddOptionalArg<std::vector<float>>("values",
      R"code(The discrete values produced by a discrete uniform distribution.

This argument is mutually exclusive with ``range``.)code",
      nullptr, true)
    .AddParent("RNGAttr")
    .Deprecate("random.UniformDistribution");  // Deprecated in 0.30


DALI_REGISTER_OPERATOR(Uniform, UniformDistributionCPU, CPU);

}  // namespace dali
