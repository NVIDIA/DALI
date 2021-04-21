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

#ifndef DALI_OPERATORS_RANDOM_NOISE_SALT_AND_PEPPER_NOISE_H_
#define DALI_OPERATORS_RANDOM_NOISE_SALT_AND_PEPPER_NOISE_H_

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_SALT_AND_PEPPER_NOISE_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float)

namespace dali {

template <typename Backend, typename T>
class salt_and_pepper_noise_impl {
 public:
  using Dist = typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                           curand_uniform_dist<float>,
                                           std::uniform_real_distribution<float>>;

  DALI_HOST_DEV explicit salt_and_pepper_noise_impl(float noise_prob = 0.05,
                                                    float salt_to_pepper_prob = 0.5)
      : noise_prob_(clamp<float>(noise_prob, 0, 1)),
        salt_prob_(noise_prob_ * clamp<float>(salt_to_pepper_prob, 0, 1)),
        salt_val_(ConvertNorm<T>(1.0f)),
        pepper_val_(std::is_unsigned<T>::value ? ConvertSatNorm<T>(0.0f) :
                                                 ConvertSatNorm<T>(-1.0f)) {}

  template <typename Generator>
  DALI_HOST_DEV inline T operator()(T in_val, Generator& g) {
    auto val = dist_(g);
    if (val < noise_prob_) {
      if (val < salt_prob_) {
        return salt_val_;
      } else {
        return pepper_val_;
      }
    }
    return in_val;
  }

 private:
  float noise_prob_;
  float salt_prob_;
  T salt_val_;
  T pepper_val_;
  Dist dist_;
};

template <typename Backend>
class SaltAndPepperNoise : public RNGBase<Backend, SaltAndPepperNoise<Backend>, true> {
 public:
  using BaseImpl = RNGBase<Backend, SaltAndPepperNoise<Backend>, true>;
  template <typename T>
  struct Dist {
    using type = salt_and_pepper_noise_impl<Backend, T>;
  };

  explicit SaltAndPepperNoise(const OpSpec &spec)
      : BaseImpl(spec),
        prob_("prob", spec),
        salt_to_pepper_prob_("salt_to_pepper_prob", spec) {
    if (prob_.IsDefined() || salt_to_pepper_prob_.IsDefined()) {
      backend_data_.ReserveDistsData(sizeof(typename Dist<double>::type) * max_batch_size_);
    }
  }

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    prob_.Acquire(spec, ws, nsamples, true);
    salt_to_pepper_prob_.Acquire(spec, ws, nsamples, true);
  }

  DALIDataType DefaultDataType() const {
    assert(false);  // should not be used.
    return {};
  }

  template <typename T>
  bool SetupDists(typename Dist<T>::type* dists_data, int nsamples) {
    if (!prob_.IsDefined() && !salt_to_pepper_prob_.IsDefined()) {
      return false;
    }
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = typename Dist<T>::type{prob_[s].data[0], salt_to_pepper_prob_[s].data[0]};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, DALI_SALT_AND_PEPPER_NOISE_TYPES, (
      using Dist = typename Dist<T>::type;
      this->template RunImplTyped<T, Dist>(ws);
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float> prob_;
  ArgValue<float> salt_to_pepper_prob_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NOISE_SALT_AND_PEPPER_NOISE_H_
