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
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float

namespace dali {

template <typename Backend, typename T>
class ShotNoiseImpl {
 public:
  using DistType =
      typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                  curand_poisson_dist,
                                  std::poisson_distribution<uint32_t>>;

  DALI_HOST_DEV explicit ShotNoiseImpl(float factor = 12)
      : factor_{factor}, inv_factor_{1.0f / factor_} {}

  template <typename Generator>
  DALI_HOST_DEV uint32_t Generate(T in_val, Generator& g) {
    if (factor_ == 0.0f)
      return in_val;
    DistType dist(cuda_max<float>(0.0f, in_val * inv_factor_));
    return dist(g);
  }

  DALI_HOST_DEV void Apply(T &output, T input, uint32_t n) {
    (void) input;  // applying noise doesn't depend on the input.
    output = ConvertSat<T>(n * factor_);
  }

 private:
  float factor_;
  float inv_factor_;
};

template <typename Backend>
class ShotNoise : public RNGBase<Backend, ShotNoise<Backend>, true> {
 public:
  using BaseImpl = RNGBase<Backend, ShotNoise<Backend>, true>;

  template <typename T>
  using Impl = ShotNoiseImpl<Backend, T>;

  explicit ShotNoise(const OpSpec &spec)
      : BaseImpl(spec),
        factor_("factor", spec) {
    backend_data_.ReserveDistsData(sizeof(Impl<double>) * max_batch_size_);
  }

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    factor_.Acquire(spec, ws, nsamples, true);
  }

  DALIDataType DefaultDataType() const {
    assert(false);  // should not be used
    return {};
  }

  template <typename T>
  bool SetupDists(Impl<T>* dists_data, int nsamples) {
    if (!factor_.IsDefined()) {
      return false;
    }
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = Impl<T>{factor_[s].data[0]};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_SHOT_NOISE_TYPES), (
      using ImplT = Impl<T>;
      BaseImpl::template RunImplTyped<T, ImplT>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_SHOT_NOISE_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float> factor_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NOISE_SHOT_NOISE_H_
