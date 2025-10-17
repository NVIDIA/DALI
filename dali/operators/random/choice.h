// Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
      double, DALIDataType, DALIImageType, DALIInterpType

namespace dali {

/**
 * @brief Variant for 1D input - we sample index and take the corresponding value from input pointer
 * @tparam indirect - whether the distribution result is used as the index into the `elements`
 * array (true) or returned directly (false).
 */
template <typename T, bool uniform, bool indirect = true>
struct ChoiceSampleDist {
  using DistType =
      std::conditional_t<uniform, std::uniform_int_distribution<>, std::discrete_distribution<>>;

  DALI_HOST_DEV explicit ChoiceSampleDist() {}

  template <bool uniform_ = uniform, typename = std::enable_if_t<!uniform_>>
  DALI_HOST_DEV ChoiceSampleDist(const T *elements, const float *p_first, const float *p_last)
      : elements_(elements), dist_{p_first, p_last} {
    static_assert(!uniform_, "This is non-uniform variant");
  }

  template <bool uniform_ = uniform, typename = std::enable_if_t<uniform_>>
  DALI_HOST_DEV ChoiceSampleDist(const T *elements, int64_t element_count)
      : elements_(elements), dist_(0, element_count - 1) {
    static_assert(uniform_, "This is uniform variant");
  }

  template <typename Generator>
  DALI_HOST_DEV T Generate(Generator &st) {
    auto choice_idx = dist_(st);
    return elements_[choice_idx];
  }

  const T *elements_ = nullptr;
  DistType dist_ = {};
};

/**
 * @brief Variant that uses the result of distribution directly, to generate values in the
 * [0, element_count) range for 0D case.
 */
template <typename T, bool uniform>
struct ChoiceSampleDist<T, uniform, false> {
  using DistType =
      std::conditional_t<uniform, std::uniform_int_distribution<T>, std::discrete_distribution<T>>;

  DALI_HOST_DEV explicit ChoiceSampleDist() {}

  template <bool uniform_ = uniform, typename = std::enable_if_t<!uniform_>>
  DALI_HOST_DEV ChoiceSampleDist(const float *p_first, const float *p_last)
      : dist_{p_first, p_last} {
    static_assert(!uniform_, "This is non-uniform variant");
  }


  template <bool uniform_ = uniform, typename = std::enable_if_t<uniform_>>
  DALI_HOST_DEV ChoiceSampleDist(int64_t element_count) : dist_(0, element_count - 1) {
    static_assert(uniform_, "This is uniform variant");
  }

  template <typename Generator>
  DALI_HOST_DEV T Generate(Generator &st) {
    return dist_(st);
  }

  DistType dist_ = {};
};

template <typename Backend>
class Choice : public rng::RNGBase<Backend, Choice<Backend>, false> {
 public:
  using BaseImpl = rng::RNGBase<Backend, Choice<Backend>, false>;

  template <typename T, bool uniform, int dim>
  using Impl = ChoiceSampleDist<T, uniform, dim>;

  explicit Choice(const OpSpec &spec) : BaseImpl(spec), p_dist_("p", spec) {}


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

  /**
   * @brief Set the same output type as input type.
   * Choice doesn't support customizing the output type.
   */
  DALIDataType DefaultDataType(const OpSpec &spec, const Workspace &ws) const {
    const auto &input = ws.Input<CPUBackend>(0);
    if (input.sample_dim() == 0) {
      DALI_ENFORCE(input.type() != DALI_BOOL && !IsFloatingPoint(input.type()),
                   make_string("Data type ", input.type(),
                               " is not supported for 0D inputs. Supported types are: ",
                               ListTypeNames<DALI_CHOICE_0D_TYPES>(), "."));
    }
    return ws.Input<CPUBackend>(0).type();
  }

  /**
   * @brief Concatenate the requested output shape with the shape of the sampled element.
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
      ElementCopy(ws);
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
  /**
   * @brief Returns the number of elements that will be sampled for given sample_idx within batch.
   *
   * If the input is scalar, it represents the number of elements, otherwise it's the outermost
   * dimension (we treat the input as flat list of elements for sampling).
   */
  int64_t GetSampleNumElements(const TensorList<CPUBackend> &input, int sample_idx) const {
    if (input.sample_dim() == 0) {
      TYPE_SWITCH(dtype_, type2id, T, (DALI_CHOICE_0D_TYPES),
      (
        return input.tensor<T>(sample_idx)[0];
      ),  // NOLINT
      (
        DALI_FAIL("Data type ", dtype_, " is not supported for 0D inputs. "
                  "Supported types are: ", ListTypeNames<DALI_CHOICE_0D_TYPES>(), ".");
      ));  // NOLINT
    }
    return input.tensor_shape_span(sample_idx)[0];
  }

  /**
   * @brief Get the dimension of the element to be sampled.
   */
  int GetElementDim(const TensorList<CPUBackend> &input) {
    if (input.sample_dim() == 0) {
      return 0;
    }
    return input.sample_dim() - 1;
  }

  /**
   * @brief Implement sampling from input of dimensionality > 1.
   *
   * The sampling is done by configuring indirect distribution to generate indices in
   * [0, num_elements), where `num_elements` is the outermost dimension of the input.
   * Next the selected samples are memcopied to the output.
   */
  void ElementCopy(Workspace &ws) {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    int num_samples = input.num_samples();
    auto &tp = ws.GetThreadPool();
    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
      int element_dim = GetElementDim(input);
      int64_t element_size = volume(input.tensor_shape(sample_idx).last(element_dim)) *
                             TypeTable::GetTypeInfo(input.type()).size();
      int64_t num_input_elements = input_list_shape_[sample_idx][0];
      int64_t num_output_elements =
          volume(output.tensor_shape(sample_idx).first(output.sample_dim() - element_dim));
      uint8_t *output_data = static_cast<uint8_t *>(output.raw_mutable_tensor(sample_idx));
      const uint8_t *input_data = static_cast<const uint8_t *>(input.raw_tensor(sample_idx));

      tp.AddWork(
          [=, this](int thread_id) {
            auto &rng = rng_[sample_idx];
            if (p_dist_.HasValue()) {
              auto dist = ChoiceSampleDist<int64_t, false, false>(
                  p_dist_[sample_idx].data,
                  p_dist_[sample_idx].data + p_dist_[sample_idx].num_elements());
              for (int64_t i = 0; i < num_output_elements; ++i) {
                auto source_idx = dist.Generate(rng);
                memcpy(output_data + i * element_size, input_data + source_idx * element_size,
                       element_size);
              }
            } else {
              auto dist = ChoiceSampleDist<int64_t, true, false>(num_input_elements);
              for (int64_t i = 0; i < num_output_elements; ++i) {
                auto source_idx = dist.Generate(rng);
                memcpy(output_data + i * element_size, input_data + source_idx * element_size,
                       element_size);
              }
            }
          },
          volume(output.tensor_shape(sample_idx)));
    }
    tp.RunAll();
  }


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
