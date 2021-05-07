// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_NOISE_GAUSSIAN_NOISE_H_
#define DALI_OPERATORS_RANDOM_NOISE_GAUSSIAN_NOISE_H_

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_GAUSSIAN_NOISE_TYPES \
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float

namespace dali {

template <typename Backend, typename T>
struct GaussianNoiseImpl {
  using FloatType =
      typename std::conditional<((std::is_integral<T>::value && sizeof(T) >= 4) || sizeof(T) > 4),
                                double, float>::type;
  using DistType = typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                               curand_normal_dist<FloatType>,
                                               std::normal_distribution<FloatType>>;

  DALI_HOST_DEV explicit GaussianNoiseImpl(FloatType mean = 0, FloatType stddev = 1)
    : dist_{mean, stddev} {}

  template <typename Generator>
  DALI_HOST_DEV FloatType Generate(T input, Generator &st) {
    (void) input;  // noise generation doesn't depend on the input
    return dist_(st);
  }

  DALI_HOST_DEV void Apply(T& output, T input, FloatType n) {
    output = ConvertSat<T>(input + n);
  }

  DistType dist_;
};

template <typename Backend>
class GaussianNoise : public RNGBase<Backend, GaussianNoise<Backend>, true> {
 public:
  using BaseImpl = RNGBase<Backend, GaussianNoise<Backend>, true>;

  template <typename T>
  using Impl = GaussianNoiseImpl<Backend, T>;

  explicit GaussianNoise(const OpSpec &spec)
      : BaseImpl(spec),
        mean_("mean", spec),
        stddev_("stddev", spec) {
    if (mean_.IsDefined() || stddev_.IsDefined()) {
      backend_data_.ReserveDistsData(sizeof(Impl<double>) * max_batch_size_);
    }
  }

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    mean_.Acquire(spec, ws, nsamples, true);
    stddev_.Acquire(spec, ws, nsamples, true);
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

  template <typename T>
  bool SetupDists(Impl<T>* dists_data, int nsamples) {
    if (!mean_.IsDefined() && !stddev_.IsDefined()) {
      return false;
    }
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = Impl<T>(mean_[s].data[0], stddev_[s].data[0]);
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_GAUSSIAN_NOISE_TYPES), (
      using ImplT = Impl<T>;
      BaseImpl::template RunImplTyped<T, ImplT>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_GAUSSIAN_NOISE_TYPES>()));
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

#endif  // DALI_OPERATORS_RANDOM_NOISE_GAUSSIAN_NOISE_H_
