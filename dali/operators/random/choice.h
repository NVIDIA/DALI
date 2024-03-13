// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_CHOICE_H_
#define DALI_OPERATORS_RANDOM_CHOICE_H_

#include <random>
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_COINFLIP_TYPES bool, uint8_t, int32_t

namespace dali {

template <typename T>
struct ChoiceSampleDist {
  using DistType = std::discrete_distribution<>;

  DALI_HOST_DEV explicit ChoiceSampleDist() {}

  // TODO(klecki): We need to initialize once or many tiles based on returns_, just
  // handle the copy in the constructor once, accept begin and end here.
  DALI_HOST_DEV ChoiceSampleDist(const T *elements, const float *p_first, const float *p_last,
                                 bool returns)
      : elements_(elements),
        p_dist_(p_first, p_last),
        dist_(p_dist_.begin(), p_dist_.end()),
        returns_(returns) {}

  template <typename Generator>
  DALI_HOST_DEV int Generate(Generator &st) {
    if (!returns_) {
      auto choice_idx = dist_(st);
      p_dist_[choice_idx] = 0;
      dist_ = DistType(p_dist_.begin(), p_dist_.end());
      return choice_idx;
    }
    auto choice_idx = dist_(st);
    return elements_[choice_idx];
  }
  const T *elements_;
  std::vector<float> p_dist_;
  DistType dist_;
  // TODO(klecki): use it only when returns_ == false
  bool returns_;
};

template <typename Backend>
class Choice : public rng::RNGBase<Backend, Choice<Backend>, false> {
 public:
  using BaseImpl = rng::RNGBase<Backend, Choice<Backend>, false>;

  template <typename T>
  using Impl = ChoiceSampleDist<T>;

  explicit Choice(const OpSpec &spec)
      : BaseImpl(spec),
        p_dist_("p", spec) {
  }

  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {
    p_dist_.Acquire(spec, ws, nsamples);

  }

  DALIDataType DefaultDataType() const {
    return DALI_INT32;
  }

  template <typename T>
  bool SetupDists(ChoiceSampleDist<T>* dists_data, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = ChoiceSampleDist<T>{p_dist_[s].data, p_dist_[s].data + p_dist_[s].num_elements()};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(Workspace &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_COINFLIP_TYPES), (
      BaseImpl::template RunImplTyped<T, Impl>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_COINFLIP_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float, 1> p_dist_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_CHOICE_H_
