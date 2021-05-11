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
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float

namespace dali {

template <typename Backend, typename T>
class SaltAndPepperNoiseImpl {
 public:
  using DistType = typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                               curand_uniform_dist<float>,
                                               std::uniform_real_distribution<float>>;
  static constexpr T kDefaultSalt =
      std::is_floating_point<T>::value ? T(1) : std::numeric_limits<T>::max();
  static constexpr T kDefaultPepper =
      std::is_floating_point<T>::value ? T(-1) : std::numeric_limits<T>::min();

  DALI_HOST_DEV explicit SaltAndPepperNoiseImpl(float noise_prob = 0.05,
                                                float salt_to_pepper_prob = 0.5,
                                                T salt_val = kDefaultSalt,
                                                T pepper_val = kDefaultPepper)
      : noise_prob_(clamp<float>(noise_prob, 0, 1)),
        salt_prob_(noise_prob_ * clamp<float>(salt_to_pepper_prob, 0, 1)),
        salt_val_(salt_val), pepper_val_(pepper_val) {
  }

  template <typename Generator>
  DALI_HOST_DEV float Generate(T input, Generator &st) {
    (void) input;  // this noise doesn't depend on the input
    return dist_(st);
  }

  DALI_HOST_DEV void Apply(T &output, T input, float n) {
    if (n < noise_prob_) {
      if (n < salt_prob_) {
        output = salt_val_;
      } else {
        output = pepper_val_;
      }
    } else {
      output = input;
    }
  }

 private:
  float noise_prob_;
  float salt_prob_;
  T salt_val_;
  T pepper_val_;
  DistType dist_;
};

template <typename Backend>
class SaltAndPepperNoise : public RNGBase<Backend, SaltAndPepperNoise<Backend>, true> {
 public:
  using BaseImpl = RNGBase<Backend, SaltAndPepperNoise<Backend>, true>;

  template <typename T>
  using Impl = SaltAndPepperNoiseImpl<Backend, T>;

  explicit SaltAndPepperNoise(const OpSpec &spec)
      : BaseImpl(spec),
        prob_("prob", spec),
        salt_to_pepper_prob_("salt_to_pepper_prob", spec),
        salt_val_("salt_val", spec),
        pepper_val_("pepper_val", spec) {
    if (prob_.IsDefined() || salt_to_pepper_prob_.IsDefined() ||
        salt_val_.IsDefined() || pepper_val_.IsDefined()) {
      backend_data_.ReserveDistsData(sizeof(Impl<double>) * max_batch_size_);
    }
  }

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    prob_.Acquire(spec, ws, nsamples, true);
    salt_to_pepper_prob_.Acquire(spec, ws, nsamples, true);
    if (salt_val_.IsDefined())
      salt_val_.Acquire(spec, ws, nsamples, true);
    if (pepper_val_.IsDefined())
      pepper_val_.Acquire(spec, ws, nsamples, true);
  }

  DALIDataType DefaultDataType() const {
    assert(false);  // should not be used.
    return {};
  }

  template <typename T>
  bool SetupDists(Impl<T>* dists_data, int nsamples) {
    if (!prob_.IsDefined() && !salt_to_pepper_prob_.IsDefined()) {
      return false;  // default constructed Impl will be used
    }
    for (int s = 0; s < nsamples; s++) {
      T salt_val = salt_val_.IsDefined() ? salt_val_[s].data[0] : Impl<T>::kDefaultSalt;
      T pepper_val = pepper_val_.IsDefined() ? pepper_val_[s].data[0] : Impl<T>::kDefaultPepper;
      dists_data[s] = Impl<T>{prob_[s].data[0], salt_to_pepper_prob_[s].data[0],
                              salt_val, pepper_val};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_SALT_AND_PEPPER_NOISE_TYPES), (
      using ImplT = Impl<T>;
      BaseImpl::template RunImplTyped<T, ImplT>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ",
                            ListTypeNames<DALI_SALT_AND_PEPPER_NOISE_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float> prob_;
  ArgValue<float> salt_to_pepper_prob_;
  ArgValue<float> salt_val_;
  ArgValue<float> pepper_val_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NOISE_SALT_AND_PEPPER_NOISE_H_
