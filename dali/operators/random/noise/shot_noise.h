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

#ifndef DALI_OPERATORS_RANDOM_NOISE_SHOT_NOISE_H_
#define DALI_OPERATORS_RANDOM_NOISE_SHOT_NOISE_H_

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_SHOT_NOISE_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t)

namespace dali {

template <typename Backend, typename T>
class shot_noise_impl {
 public:
  using Dist =
      typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                  curand_poisson_dist,
                                  std::poisson_distribution<uint32_t>>;

  DALI_HOST_DEV explicit shot_noise_impl(float peak = 12)
      : peak_{peak} {}

  template <typename Generator>
  DALI_HOST_DEV inline T operator()(T in_val, Generator& g) {
    float norm = ConvertNorm<float>(in_val);
    Dist dist(peak_ * norm);
    return ConvertNorm<T>(dist(g) / peak_);
  }

 private:
  float peak_;
};

template <typename Backend>
class ShotNoise : public RNGBase<Backend, ShotNoise<Backend>, true> {
 public:
  using BaseImpl = RNGBase<Backend, ShotNoise<Backend>, true>;
  template <typename T>
  struct Dist {
    using type = shot_noise_impl<Backend, T>;
  };

  explicit ShotNoise(const OpSpec &spec)
      : BaseImpl(spec),
        peak_("peak", spec) {}

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    peak_.Acquire(spec, ws, nsamples, true);
    for (int s = 0; s < nsamples; s++) {
      DALI_ENFORCE(
          peak_[s].data[0] > 0.0f,
          make_string("``peak`` argument must be greater than 0.0. Got: ", peak_[s].data[0]));
    }
  }

  DALIDataType DefaultDataType() const {
    assert(false);  // should not be used
    return {};
  }

  template <typename T>
  bool SetupDists(typename Dist<T>::type* dists_data, int nsamples) {
    if (!peak_.IsDefined()) {
      return false;
    }
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = typename Dist<T>::type{peak_[s].data[0]};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, DALI_SHOT_NOISE_TYPES, (
      using Dist = typename Dist<T>::type;
      this->template RunImplTyped<T, Dist>(ws);
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float> peak_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NOISE_GAUSSIAN_NOISE_H_
