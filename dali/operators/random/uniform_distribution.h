// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dev_buffer.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"

#define DALI_UNIFORM_DIST_TYPES \
  uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double

namespace dali {

/**
 * @brief Wraps std::uniform_real_distribution<T>> and works around the issue
 *        with end of the range (GCC and LLVM bug)
 */
template <typename T>
class uniform_real_dist {
 public:
  explicit uniform_real_dist(T range_start = -1, T range_end = 1)
      : range_end_(range_end)
      , dist_(range_start, range_end) {
    assert(range_end > range_start);
  }

  template <typename Generator>
  inline T operator()(Generator& g) {
    T val = range_end_;
    while (val >= range_end_)
      val = dist_(g);
    return val;
  }

 private:
  T range_end_ = 1;
  std::uniform_real_distribution<T> dist_;
};


/**
 * @brief Draws values from a discrete uniform distribution
 */
template <typename T>
class uniform_int_values_dist {
 public:
  uniform_int_values_dist() : values_(nullptr), nvalues_(0) {
    // Should not be used. It is just here to make the base
    // RNG operator code easier.
    assert(false);
  }
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

template <typename Backend, typename T>
struct UniformDistributionContinuousImpl {
  using FloatType =
      typename std::conditional<((std::is_integral<T>::value && sizeof(T) >= 4) || sizeof(T) > 4),
                                double, float>::type;
  using DistType =
      typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                  curand_uniform_dist<FloatType>, uniform_real_dist<FloatType>>;

  DALI_HOST_DEV UniformDistributionContinuousImpl()
    : dist_(-1, 1) {}

  DALI_HOST_DEV explicit UniformDistributionContinuousImpl(FloatType range_start,
                                                           FloatType range_end)
      : dist_{range_start, range_end} {}

  template <typename Generator>
  DALI_HOST_DEV FloatType Generate(Generator &st) {
    return dist_(st);
  }

  DistType dist_;
};

template <typename Backend, typename T>
struct UniformDistributionDiscreteImpl {
  using DistType = typename std::conditional_t<std::is_same<Backend, GPUBackend>::value,
                                               curand_uniform_int_values_dist<float>,
                                               uniform_int_values_dist<float>>;

  DALI_HOST_DEV explicit UniformDistributionDiscreteImpl() : dist_(nullptr, 0) {}

  DALI_HOST_DEV explicit UniformDistributionDiscreteImpl(const float *values, int64_t nvalues)
    : dist_(values, nvalues) {}

  template <typename Generator>
  DALI_HOST_DEV float Generate(Generator &st) {
    return dist_(st);
  }

  DistType dist_;
};


template <typename Backend>
class UniformDistribution : public rng::RNGBase<Backend, UniformDistribution<Backend>, false> {
 public:
  using BaseImpl = rng::RNGBase<Backend, UniformDistribution<Backend>, false>;

  template <typename T>
  struct ImplDiscrete {
    using type = UniformDistributionDiscreteImpl<Backend, T>;
  };

  template <typename T>
  struct ImplContinuous {
    using type = UniformDistributionContinuousImpl<Backend, T>;
  };

  explicit UniformDistribution(const OpSpec &spec)
      : BaseImpl(spec),
        values_("values", spec),
        range_("range", spec) {
    int size_dist = values_.HasExplicitValue() ? sizeof(typename ImplDiscrete<double>::type)
                                        : sizeof(typename ImplContinuous<double>::type);
    per_sample_values_.reserve(max_batch_size_);
    per_sample_nvalues_.reserve(max_batch_size_);
  }

  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {
    if (values_.HasExplicitValue()) {
      // read only once for build time arguments
      if (!values_.HasExplicitConstant() || per_sample_values_.empty()) {
        values_.Acquire(spec, ws, values_.HasExplicitConstant() ? max_batch_size_ : nsamples);
        per_sample_values_.resize(values_.size());
        per_sample_nvalues_.resize(values_.size());
        if (std::is_same<Backend, GPUBackend>::value) {
          kernels::DynamicScratchpad scratch({}, ws.stream());
          int64_t nvalues = values_.get().shape.num_elements();

          auto values_cpu =
              make_span(scratch.Allocate<mm::memory_kind::pinned, float>(nvalues), nvalues);
          for (int64_t s = 0, k = 0; s < nsamples; s++) {
            per_sample_nvalues_[s] = values_[s].shape.num_elements();
            for (int64_t v = 0; v < per_sample_nvalues_[s]; v++, k++)
              values_cpu[k] = values_[s].data[v];
          }
          values_gpu_.from_host(values_cpu, ws.stream());

          int64_t offset = 0;
          for (int s = 0; s < nsamples; s++) {
            per_sample_values_[s] = values_gpu_.data() + offset;
            offset += per_sample_nvalues_[s];
          }
        } else {
          for (int s = 0; s < values_.size(); s++) {
            per_sample_values_[s] = values_[s].data;
            per_sample_nvalues_[s] = values_[s].shape[0];
          }
        }
      }
    } else {
      range_.Acquire(spec, ws, nsamples, TensorShape<1>{2});
      for (int s = 0; s < nsamples; s++) {
        float start = range_[s].data[0], end = range_[s].data[1];
        DALI_ENFORCE(end > start, make_string("Invalid range [", start, ", ", end, ")."));
      }
    }
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

  template <typename T>
  bool SetupDists(typename ImplContinuous<T>::type* dists, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists[s] = typename ImplContinuous<T>::type{range_[s].data[0], range_[s].data[1]};
    }
    // note: can't use the default because this operator's default range is different to
    // the default constructed distribution.
    return true;
  }

  template <typename T>
  bool SetupDists(typename ImplDiscrete<T>::type* dists, int nsamples) {
    assert(values_.HasExplicitValue());
    for (int s = 0; s < nsamples; s++) {
      dists[s] = typename ImplDiscrete<T>::type{per_sample_values_[s], per_sample_nvalues_[s]};
    }
    return true;
  }

  template <typename T>
  void RunImplTyped(Workspace &ws) {
    using Base = rng::RNGBase<Backend, UniformDistribution<Backend>, false>;
    if (values_.HasExplicitValue()) {
      using ImplT = typename ImplDiscrete<T>::type;
      Base::template RunImplTyped<T, ImplT>(ws);
    } else {
      using ImplT = typename ImplContinuous<T>::type;
      Base::template RunImplTyped<T, ImplT>(ws);
    }
  }

  void RunImpl(Workspace &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_UNIFORM_DIST_TYPES), (
      this->template RunImplTyped<T>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_UNIFORM_DIST_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float, 1> values_;
  ArgValue<float, 1> range_;

  DeviceBuffer<float> values_gpu_;
  std::vector<const float*> per_sample_values_;
  std::vector<int64_t> per_sample_nvalues_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_
