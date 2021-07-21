// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_COIN_FLIP_H_
#define DALI_OPERATORS_RANDOM_COIN_FLIP_H_

#include <random>
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_COINFLIP_TYPES bool, uint8_t, int32_t

namespace dali {

template <typename Backend>
struct CoinFlipImpl {
  using DistType =
    typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
        curand_bernoulli_dist,
        std::bernoulli_distribution>;

  DALI_HOST_DEV explicit CoinFlipImpl() {}

  DALI_HOST_DEV explicit CoinFlipImpl(float prob)
    : dist_(prob) {}

  template <typename Generator>
  DALI_HOST_DEV bool Generate(Generator &st) {
    return dist_(st);
  }

  DistType dist_;
};

template <typename Backend>
class CoinFlip : public RNGBase<Backend, CoinFlip<Backend>, false> {
 public:
  using BaseImpl = RNGBase<Backend, CoinFlip<Backend>, false>;

  using Impl = CoinFlipImpl<Backend>;

  explicit CoinFlip(const OpSpec &spec)
      : RNGBase<Backend, CoinFlip<Backend>, false>(spec),
        probability_("probability", spec) {
    backend_data_.ReserveDistsData(sizeof(Impl) * max_batch_size_);
  }

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    probability_.Acquire(spec, ws, nsamples, true);
  }

  DALIDataType DefaultDataType() const {
    return DALI_INT32;
  }

  template <typename T>
  bool SetupDists(Impl* dists_data, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = Impl{probability_[s].data[0]};
    }
    return true;
  }

  using RNGBase<Backend, CoinFlip<Backend>, false>::RunImpl;
  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_COINFLIP_TYPES), (
      BaseImpl::template RunImplTyped<T, Impl>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_COINFLIP_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using RNGBase<Backend, CoinFlip<Backend>, false>::dtype_;
  using RNGBase<Backend, CoinFlip<Backend>, false>::backend_data_;

  ArgValue<float> probability_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_COIN_FLIP_H_
