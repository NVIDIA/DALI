// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_H_
#define DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_H_

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_NORMDIST_TYPES \
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double

namespace dali {

template <typename Backend, typename T>
struct NormalDistImpl {
  using FloatType =
      typename std::conditional<((std::is_integral<T>::value && sizeof(T) >= 4) || sizeof(T) > 4),
                                double, float>::type;
  using DistType = typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                               curand_normal_dist<FloatType>,
                                               std::normal_distribution<FloatType>>;

  DALI_HOST_DEV NormalDistImpl() {}

  DALI_HOST_DEV explicit NormalDistImpl(FloatType mean, FloatType stddev)
    : dist_{mean, stddev} {}

  template <typename Generator>
  DALI_HOST_DEV FloatType Generate(Generator &st) {
    return dist_(st);
  }

  DistType dist_;
};

template <typename Backend>
class NormalDistribution : public rng::RNGBase<Backend, NormalDistribution<Backend>, false> {
 public:
  using BaseImpl = rng::RNGBase<Backend, NormalDistribution<Backend>, false>;

  template <typename T>
  using Impl = NormalDistImpl<Backend, T>;

  explicit NormalDistribution(const OpSpec &spec)
      : BaseImpl(spec),
        mean_("mean", spec),
        stddev_("stddev", spec) {
  }

  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {
    mean_.Acquire(spec, ws, nsamples);
    stddev_.Acquire(spec, ws, nsamples);
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

  template <typename T>
  bool SetupDists(Impl<T>* dists_data, int nsamples) {
    if (!mean_.HasExplicitValue() && !stddev_.HasExplicitValue()) {
      return false;
    }
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = Impl<T>(mean_[s].data[0], stddev_[s].data[0]);
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(Workspace &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_NORMDIST_TYPES), (
      using ImplT = Impl<T>;
      BaseImpl::template RunImplTyped<T, ImplT>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_NORMDIST_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float> mean_;
  ArgValue<float> stddev_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_H_
