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

#ifndef DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_
#define DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_

#include <vector>
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/core/dev_buffer.h"

#define DALI_UNIFORM_DIST_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, \
                                 int64_t, float16, float, double)

namespace dali {

template <typename T>
class uniform_int_values_dist {
 public:
  uniform_int_values_dist() : values_(nullptr), nvalues_(0) {}
  uniform_int_values_dist(const T *values, int64_t nvalues)
    : values_(values), nvalues_(nvalues), dist_(0, nvalues-1) {}

  template <typename Generator>
  inline T operator()(Generator& g) {
    int idx = dist_(g);
    return values_[idx];
  }

 private:
  const T *values_ = nullptr;
  int64_t nvalues_ = 0;
  std::uniform_int_distribution<int> dist_;
};

template <typename Backend>
class UniformDistribution : public RNGBase<Backend, UniformDistribution<Backend>> {
 public:
  template <typename T>
  struct DistContinuous {
    using FloatType =
      typename std::conditional<
          ((std::is_integral<T>::value && sizeof(T) >= 4) || sizeof(T) > 4),
          double, float>::type;

    using type =
      typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
          curand_uniform_dist<FloatType>,
          std::uniform_real_distribution<FloatType>>;
  };

  template <typename T>
  struct DistDiscrete {
    // TODO(janton): work directly with T values? (It'll need converting the values)
    using type = typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
        curand_uniform_int_values_dist<float>,
        uniform_int_values_dist<float>>;
  };

  explicit UniformDistribution(const OpSpec &spec)
      : RNGBase<Backend, UniformDistribution<Backend>>(spec),
        values_("values", spec),
        range_("range", spec) {
    int size_dist = values_.IsDefined() ? sizeof(typename DistDiscrete<double>::type)
                                        : sizeof(typename DistContinuous<double>::type);
    backend_data_.ReserveDistsData(size_dist * max_batch_size_);
    per_sample_values_.reserve(max_batch_size_);
    per_sample_nvalues_.reserve(max_batch_size_);
  }

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    if (values_.IsDefined()) {
      values_.Acquire(spec, ws, nsamples, false);
      // read only once for build time arguments
      if (!values_.IsConstant() || per_sample_values_.empty()) {
        per_sample_values_.resize(nsamples);
        per_sample_nvalues_.resize(nsamples);
        if (std::is_same<Backend, GPUBackend>::value) {
          values_cpu_.clear();
          for (int s = 0; s < nsamples; s++) {
            values_cpu_.insert(values_cpu_.end(),
                               values_[s].data,
                               values_[s].data + values_[s].shape[0]);
            per_sample_nvalues_[s] = values_[s].shape[0];
          }
          values_gpu_.from_host(values_cpu_, ws.stream());
          int64_t offset = 0;
          for (int s = 0; s < nsamples; s++) {
            per_sample_values_[s] = values_gpu_.data() + offset;
            offset += per_sample_nvalues_[s];
          }
        } else {
          for (int s = 0; s < nsamples; s++) {
            per_sample_values_[s] = values_[s].data;
            per_sample_nvalues_[s] = values_[s].shape[0];
          }
        }
      }
    } else {
      range_.Acquire(spec, ws, nsamples, TensorShape<1>{2});
    }
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

  template <typename T>
  bool SetupDists(typename DistContinuous<T>::type* dists, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists[s] = typename DistContinuous<T>::type(range_[s].data[0], range_[s].data[1]);
    }
    // note: can't use the default because this operator's default range is different to
    // the default constructed distribution.
    return true;
  }

  template <typename T>
  bool SetupDists(typename DistDiscrete<T>::type* dists, int nsamples) {
    assert(values_.IsDefined());
    for (int s = 0; s < nsamples; s++) {
      dists[s] = typename DistDiscrete<T>::type(per_sample_values_[s], per_sample_nvalues_[s]);
    }
    return true;
  }

  template <typename T>
  void RunImplTyped(workspace_t<Backend> &ws) {
    using Base = RNGBase<Backend, UniformDistribution<Backend>>;
    if (values_.IsDefined()) {
      using Dist = typename DistDiscrete<T>::type;
      Base::template RunImplTyped<T, Dist>(ws);
    } else {
      using Dist = typename DistContinuous<T>::type;
      Base::template RunImplTyped<T, Dist>(ws);
    }
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, DALI_UNIFORM_DIST_TYPES, (
      this->template RunImplTyped<T>(ws);
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using RNGBase<Backend, UniformDistribution<Backend>>::dtype_;
  using RNGBase<Backend, UniformDistribution<Backend>>::backend_data_;

  ArgValue<float, 1> values_;
  ArgValue<float, 1> range_;

  std::vector<float> values_cpu_;
  DeviceBuffer<float> values_gpu_;
  std::vector<const float*> per_sample_values_;
  std::vector<int64_t> per_sample_nvalues_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_
