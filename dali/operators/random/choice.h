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

#include <cstdint>
#include <random>
#include "dali/core/error_handling.h"
#include "dali/operators/random/rng_base.h"
#include "dali/operators/random/rng_base_cpu.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/arg_helper.h"

#define DALI_CHOICE_0D_TYPES \
  uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
#define DALI_CHOICE_1D_TYPES                                                                      \
  bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16, float, \
      double

namespace dali {

/**
 * @brief Variant for 1D input - we sample index and take the corresponding value from input pointer
 */
template <typename T, bool uniform, bool indirect = true>
struct ChoiceSampleDist {
  using DistType =
      std::conditional_t<uniform, std::uniform_int_distribution<>, std::discrete_distribution<>>;

  DALI_HOST_DEV explicit ChoiceSampleDist() {}

  DALI_HOST_DEV ChoiceSampleDist(const T *elements, const float *p_first, const float *p_last)
      : elements_(elements), element_count_(p_last - p_first) {
    if constexpr (!uniform) {
      dist_ = DistType(p_first, p_last);
    } else {
      assert(false);  // Should not be called
    }
  }

  DALI_HOST_DEV ChoiceSampleDist(const T *elements, int64_t element_count)
      : elements_(elements), element_count_(element_count) {
    if constexpr (uniform) {
      dist_ = DistType(0, element_count - 1);
    } else {
      assert(false);  // Should not be called
    }
  }

  template <typename Generator>
  DALI_HOST_DEV T Generate(Generator &st) {
    auto choice_idx = dist_(st);
    return elements_[choice_idx];
  }

  const T *elements_;
  int64_t element_count_;
  DistType dist_;
};

/**
 * @brief Variant for generating indices. In case of 0D input, we treat those indices as the
 * actual sampled values in range [0, element_count).
 */
template <typename T, bool uniform>
struct ChoiceSampleDist<T, uniform, false> {
  using DistType =
      std::conditional_t<uniform, std::uniform_int_distribution<T>, std::discrete_distribution<T>>;

  DALI_HOST_DEV explicit ChoiceSampleDist() {}

  DALI_HOST_DEV ChoiceSampleDist(const float *p_first, const float *p_last)
      : element_count_(p_last - p_first) {
    if constexpr (!uniform) {
      dist_ = DistType(p_first, p_last);
    } else {
      assert(false);  // Should not be called
    }
  }

  DALI_HOST_DEV ChoiceSampleDist(int64_t element_count) : element_count_(element_count) {
    if constexpr (uniform) {
      dist_ = DistType(0, element_count - 1);
    } else {
      assert(false);  // Should not be called
    }
  }

  template <typename Generator>
  DALI_HOST_DEV T Generate(Generator &st) {
    return dist_(st);
  }

  int64_t element_count_;
  DistType dist_;
};

template <typename Backend>
class Choice : public rng::RNGBase<Backend, Choice<Backend>, false> {
 public:
  using BaseImpl = rng::RNGBase<Backend, Choice<Backend>, false>;

  template <typename T, bool uniform, int dim>
  using Impl = ChoiceSampleDist<T, uniform, dim>;

  explicit Choice(const OpSpec &spec) : BaseImpl(spec), p_dist_("p", spec) {}

  /**
   * @brief Returns the number of elements that will be sampled for given sample_idx within batch.
   *
   * If the input is scalar, it represents the number of elements, otherwise it's the outermost
   * dimension (we treat the input as flat list of elements for sampling).
   */
  int64_t GetSampleNumElements(const TensorList<CPUBackend> &input, int sample_idx) const {
    if (input.sample_dim() == 0) {
      TYPE_SWITCH(input.type(), type2id, T, (DALI_CHOICE_0D_TYPES),
      (
        return input.tensor<T>(sample_idx)[0];
      ),  // NOLINT
      ());
    }
    return input.tensor_shape_span(sample_idx)[0];
  }

  int GetElementDim(const TensorList<CPUBackend> &input) {
    if (input.sample_dim() == 0) {
      return 0;
    }
    return input.sample_dim() - 1;
  }

  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {
    const auto &input = ws.Input<CPUBackend>(0);
    input_list_shape_.resize(nsamples);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      int64_t num_elements = GetSampleNumElements(ws.Input<CPUBackend>(0), sample_idx);
      DALI_ENFORCE(num_elements > 0,
                   make_string("Expected positive number of elements for sampling, got: ",
                               num_elements, " for sample: ", sample_idx, "."));
      input_list_shape_.set_tensor_shape(sample_idx, {num_elements});
    }
    if (p_dist_.HasValue()) {
      p_dist_.Acquire(spec, ws, nsamples, input_list_shape_);
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        double sum = 0.0;
        for (int i = 0; i < p_dist_[sample_idx].num_elements(); i++) {
          DALI_ENFORCE(p_dist_[sample_idx].data[i] >= 0.0 && p_dist_[sample_idx].data[i] <= 1.0,
                       make_string("Probabilities must be in range [0, 1], but got: ",
                                   p_dist_[sample_idx].data[i], " for sample: ", sample_idx,
                                   " at index ", i, "."));
          sum += p_dist_[sample_idx].data[i];
        }
        DALI_ENFORCE(std::fabs(sum - 1.0) < 1e-4,
                     make_string("Sum of probabilities must be 1.0, but got ", sum,
                                 " for sample: ", sample_idx, "."));
      }
    }
  }

  DALIDataType DefaultDataType(const OpSpec &spec, const Workspace &ws) const {
    if (ws.Input<CPUBackend>(0).sample_dim() == 0) {
      return DALI_INT32;
    } else {
      return ws.Input<CPUBackend>(0).type();
    }
  }

  /**
   * @brief We only allow sampling of the input elements or generating integral types if the
   * input is scalar.
   */
  void ValidateDataType(const OpSpec &spec, const Workspace &ws) const {
    if (ws.Input<CPUBackend>(0).sample_dim() != 0) {
      DALI_ENFORCE(ws.Input<CPUBackend>(0).type() == dtype_,
                   make_string("For output sampled from list of input samples "
                               "(when the input is not a scalar), the requested output type must "
                               "match the type of the input, expected: ",
                               ws.Input<CPUBackend>(0).type(), ", got: ", dtype_, "."));
    }
  }

  /**
   * @brief Concatenate the requested shape with the shape of sampled elements.
   */
  TensorListShape<> PostprocessShape(const OpSpec &spec, const Workspace &ws) {
    const auto &input = ws.Input<CPUBackend>(0);
    int element_dim = GetElementDim(input);
    TensorListShape<> shape(shape_.num_samples(), shape_.sample_dim() + element_dim);
    for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
      auto result = shape_cat(shape_[sample_idx], input.tensor_shape(sample_idx).last(element_dim));
      shape.set_tensor_shape(sample_idx, result);
    }
    return shape;
  }


  template <typename T>
  bool SetupDists(Impl<T, false, false> *dists_data, const Workspace &ws, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] =
          Impl<T, false, false>{p_dist_[s].data, p_dist_[s].data + p_dist_[s].num_elements()};
    }
    return true;
  }

  template <typename T>
  bool SetupDists(Impl<T, true, 0> *dists_data, const Workspace &ws, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = Impl<T, true, false>{input_list_shape_[s][0]};
    }
    return true;
  }


  template <typename T>
  bool SetupDists(Impl<T, false, true> *dists_data, const Workspace &ws, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] = Impl<T, false, true>{ws.Input<CPUBackend>(0).tensor<T>(s), p_dist_[s].data,
                                           p_dist_[s].data + p_dist_[s].num_elements()};
    }
    return true;
  }

  template <typename T>
  bool SetupDists(Impl<T, true, true> *dists_data, const Workspace &ws, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      dists_data[s] =
          Impl<T, true, true>{ws.Input<CPUBackend>(0).tensor<T>(s), input_list_shape_[s][0]};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    if (input.sample_dim() == 0) {
      TYPE_SWITCH(dtype_, type2id, T, (DALI_CHOICE_0D_TYPES), (
        if (p_dist_.HasValue()) {
          BaseImpl::template RunImplTyped<T, Impl<T, false, false>>(ws);
        } else {
          BaseImpl::template RunImplTyped<T, Impl<T, true, false>>(ws);
        }
      ), (  // NOLINT
        DALI_FAIL("Data type ", dtype_, " is not supported for 0D inputs. "
                  "Supported types are: ", ListTypeNames<DALI_CHOICE_0D_TYPES>(), ".");
      ));  // NOLINT
    } else if (input.sample_dim() == 1) {
      TYPE_SWITCH(dtype_, type2id, T, (DALI_CHOICE_1D_TYPES), (
        if (p_dist_.HasValue()) {
          BaseImpl::template RunImplTyped<T, Impl<T, false, true>>(ws);
        } else {
          BaseImpl::template RunImplTyped<T, Impl<T, true, true>>(ws);
        }
      ), (  // NOLINT
        DALI_FAIL("Data type ", dtype_, " is not supported for 1D inputs. "
                  "Supported types are: ", ListTypeNames<DALI_CHOICE_1D_TYPES>(), ".");
      ));  // NOLINT
    } else {
      DALI_FAIL("The operator only supports sampling of 0D elements, got: ", input.sample_dim(),
                "D input.");
    }
    if (!input.GetLayout().empty()) {
      if (input.sample_dim() == output.sample_dim()) {
        output.SetLayout(input.GetLayout());
      } else if (input.sample_dim() > 0 && output.sample_dim() == input.sample_dim() - 1) {
        output.SetLayout(input.GetLayout().sub(1));
      }
    } else {
      output.SetLayout("");
    }
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::backend_data_;
  using BaseImpl::dtype_;
  using BaseImpl::rng_;
  using BaseImpl::shape_;


  ArgValue<float, 1> p_dist_;
  // The shape of input as interpreted as the flat list of elements to be sampled.
  TensorListShape<1> input_list_shape_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_CHOICE_H_
