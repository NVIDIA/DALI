// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_TEST_TENSOR_TEST_UTILS_H_
#define DALI_TEST_TENSOR_TEST_UTILS_H_

#include <gtest/gtest.h>
#include <functional>
#include <cmath>
#include <utility>
#include <random>
#include <string>
#include "dali/core/tensor_view.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/backend_tags.h"
#include "dali/core/util.h"
#include "dali/core/convert.h"

namespace dali {

namespace detail {
static constexpr int DefaultMaxErrors = 100;


}  // namespace detail

// COMPARISON

template <typename Output, typename Reference>
bool IsEqWithConvert(Output out, Reference ref, int ulp = 4) {
  for (int i = 0; i <= ulp; i++) {
    if (dali::ConvertSat<Output>(ref) == out)
      return true;
    ref = std::nextafter(ref, static_cast<Reference>(out));
  }
  return false;
}


template <int dim1, int dim2>
void CheckEqual(const TensorListShape<dim1> &s1, const TensorListShape<dim2> &s2, int &max_errors) {
  ASSERT_EQ(s1.sample_dim(), s2.sample_dim()) << "Tensor lists must have equal sample dimension";
  ASSERT_EQ(s1.num_samples(), s2.num_samples()) << "Tensor lists must have same number of samples";
  int n = s1.num_samples();
  int errors = 0;
  for (int i = 0; i < n; i++) {
    if (s1.tensor_shape_span(i) != s2.tensor_shape_span(i)) {
      if (++errors < max_errors) {
        EXPECT_EQ(s1.tensor_shape_span(i), s2.tensor_shape_span(i))
                      << "Samples #" << i
                      << " have different shapes " << s1[i] << " vs " << s2[i];
      }
    }
  }

  if (errors) {
    if (errors > max_errors) {
      max_errors = 0;
      FAIL() << " (" << (errors - max_errors) << " more)";
    } else {
      max_errors -= errors;
      FAIL();
    }
  }
}


template <int dim1, int dim2>
void CheckEqual(const TensorListShape<dim1> &s1, const TensorListShape<dim2> &s2) {
  int max_errors = detail::DefaultMaxErrors;
  CheckEqual(s1, s2, max_errors);
}


template <typename T>
T printable(T t) { return t; }


inline int printable(char c) { return c; }


inline int printable(signed char c) { return c; }


inline int printable(unsigned char c) { return c; }


template <typename StorageBackend, typename T1, typename T2,
        int dim1, int dim2, typename ElementsOkFunc>
void Check(const TensorView<StorageBackend, T1, dim1> &tv1,
           const TensorView<StorageBackend, T2, dim2> &tv2, ElementsOkFunc eq, int &max_errors) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorView backend");
  ASSERT_EQ(tv1.shape, tv2.shape)
                        << "Tensors have different shapes "
                        << tv1.shape << " vs " << tv2.shape << "\n";

  ptrdiff_t n = tv1.num_elements();
  int dim = tv1.dim();
  TensorShape<DynamicDimensions> pos;
  pos.resize(dim);

  int errors = 0;
  for (ptrdiff_t i = 0; i < n; i++) {
    if (!eq(tv1.data[i], tv2.data[i])) {
      if (errors++ < max_errors) {
        EXPECT_PRED2(eq, tv1.data[i], tv2.data[i]) << "Failed at index " << i << ", pos = " << pos;
      }
    }

    for (int j = dim - 1; j >= 0; j--) {
      if (++pos[j] < tv1.shape[j])
        break;
      pos[j] = 0;
    }
  }

  if (errors) {
    if (errors > max_errors) {
      max_errors = 0;
      FAIL() << " (" << (errors - max_errors) << " more)";
    } else {
      max_errors -= errors;
      FAIL();
    }
  }
}


struct Equal {
  template <typename T1, typename T2>
  bool operator()(const T1 &a, const T2 &b) const {
    return a == b;
  }
};


struct EqualRelative {
  EqualRelative() = default;

  explicit EqualRelative(double eps) : eps(eps) {}

  template<typename T1, typename T2>
  bool operator()(const T1 &a, const T2 &b) {
    return std::abs(a - b) <= eps * std::max<double>(std::abs(a), std::abs(b));
  }

  double eps = 1e-5;
};


struct EqualEps {
  EqualEps() = default;

  explicit EqualEps(double eps) : eps(eps) {}

  template <typename T1, typename T2>
  bool operator()(const T1 &a, const T2 &b) const {
    auto abs_diff = b >= a ? b - a : a - b;
    return abs_diff <= eps;
  }

  double eps = 1e-6;
};


/**
 * @brief A predicate that checks that the arguments are within relative or absolute error range.
 *
 * This predicate checks if the two values (a, b) are nearby. The condition is satisified when
 * the values differ by no more than fixed epsilon **OR** are within an error range relative
 * to their values. This allows to use the fixed epsilon for small numbers and relative range
 * for large numbers - e.g.
 * ```
 * eps = 1e-5
 * rel = 1e-4
 *
 * (0.00001, 0.000011)  <- condition satisfied, absolute error == 0.000001 less than 1e-5
 * (10000, 10001)       <- condition satisified, relative error == 1e-5 less than 1e-4
 * (0.00100, 0.00102)   <- condition not satisified - absolute error 2e-5, relative error 0.0196
 * ```
 */
struct EqualEpsRel {
  EqualEpsRel() = default;

  explicit EqualEpsRel(double eps, double rel) : eps(eps), rel(rel) {}

  template <typename T1, typename T2>
  bool operator()(const T1 &a, const T2 &b) const {
    auto dif = std::abs(b - a);
    return dif <= eps || dif <= rel * std::max<double>(std::abs(a), std::abs(b));
  }

  double eps = 1e-6, rel = 1e-4;
};


/**
 * @brief Functor for comparing (potentially rounded) value to a floating point reference
 * Performs ULP comparision.
 * Verifies using Saturation Cast.
 *
 * @remark Be aware, that when using this functor in `Check()`, order of arguments matters!
 */
struct EqualUlp {
  EqualUlp() = default;

  explicit EqualUlp(int ulp) : ulp_(ulp) {}

  template <typename Output, typename Reference>
  bool operator()(Output out, Reference ref) {
    return IsEqWithConvert(out, ref, ulp_);
  }

  int ulp_ = 4;
};


template <typename StorageBackend, typename T1, typename T2,
        int dim1, int dim2, typename ElementsOkFunc = Equal>
void Check(const TensorView<StorageBackend, T1, dim1> &tv1,
           const TensorView<StorageBackend, T2, dim2> &tv2, ElementsOkFunc eq = {}) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorView backend");
  int max_errors = detail::DefaultMaxErrors;
  Check(tv1, tv2, std::move(eq), max_errors);
}


template <typename StorageBackend, typename T1, typename T2,
        int dim1, int dim2, typename ElementsOkFunc = Equal>
void Check(const TensorListView<StorageBackend, T1, dim1> &tv1,
           const TensorListView<StorageBackend, T2, dim2> &tv2, ElementsOkFunc eq = {}) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorListView backend");
  int max_errors = detail::DefaultMaxErrors;
  CheckEqual(tv1.shape, tv2.shape, max_errors);
  int n = tv1.num_samples();
  for (int i = 0; i < n; i++) {
    Check(tv1[i], tv2[i], eq, max_errors);
  }
}


template <typename Backend1, typename Backend2, typename T1, typename T2, int dim1, int dim2>
void Check(const TensorView<Backend1, T1, dim1> &t1, const TensorView<Backend2, T2, dim2> &t2) {
  static_assert(is_cpu_accessible<Backend1>::value,
                "This function is applicable only for CPU accessible backends");
  static_assert(is_cpu_accessible<Backend2>::value,
                "This function is applicable only for CPU accessible backends");
  auto t1cpu = make_tensor_cpu(t1.data, t1.shape);
  auto t2cpu = make_tensor_cpu(t2.data, t2.shape);
  Check(t1cpu, t2cpu);
}


// FILLING

template <typename Collection, typename Generator>
if_iterable<Collection, void> Fill(Collection &&collection, Generator &&generator) {
  for (auto &x : collection)
    x = generator();
}


template <typename StorageBackend, typename DataType, int ndim, typename Generator>
void Fill(const TensorView<StorageBackend, DataType, ndim> &tv, Generator &&generator) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorView backend");
  Fill(make_span(tv.data, tv.num_elements()), std::forward<Generator>(generator));
}


template <typename StorageBackend, typename DataType, int ndim, typename Generator>
void Fill(const TensorListView<StorageBackend, DataType, ndim> &tlv, Generator &&generator) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorListView backend");
  for (int i = 0; i < tlv.num_samples(); i++) {
    Fill(tlv[i], std::forward<Generator>(generator));
  }
}


template <typename T>
std::enable_if_t<std::is_floating_point<T>::value,
        std::uniform_real_distribution<T>>
uniform_distribution(T lo, T hi) {
  return std::uniform_real_distribution<T>(lo, hi);
}


template <typename T>
std::enable_if_t<std::is_integral<T>::value,
        std::uniform_int_distribution<T>>
uniform_distribution(T lo, T hi) {
  return std::uniform_int_distribution<T>(lo, hi);
}


template <typename Collection, typename RandomGenerator>
if_iterable<Collection, void>
UniformRandomFill(Collection &&c, RandomGenerator &rng, element_t<Collection> lo,
                  element_t<Collection> hi) {
  auto dist = uniform_distribution(lo, hi);
  auto generator = [&]() { return dist(rng); };
  Fill(std::forward<Collection>(c), generator);
}


template <typename StorageBackend, typename DataType, int ndim, typename RandomGenerator>
void
UniformRandomFill(const TensorView<StorageBackend, DataType, ndim> &tv, RandomGenerator &generator,
                  same_as_t<DataType> lo, same_as_t<DataType> hi) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorView backend");
  UniformRandomFill(make_span(tv.data, tv.num_elements()), generator, lo, hi);
}


template <typename StorageBackend, typename DataType, int ndim, typename RandomGenerator>
void UniformRandomFill(const TensorListView<StorageBackend, DataType, ndim> &tlv,
                       RandomGenerator &generator, same_as_t<DataType> lo, same_as_t<DataType> hi) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorListView backend");
  for (int i = 0; i < tlv.num_samples(); i++)
    UniformRandomFill(tlv[i], generator, lo, hi);
}


template <typename C>
if_iterable<C, void> ConstantFill(C &&c, const element_t<C> &value = {}) {
  for (auto &x : c)
    x = value;
}


template <typename StorageBackend, typename DataType, int dim>
void
ConstantFill(const TensorView<StorageBackend, DataType, dim> &tv, same_as_t<DataType> value = {}) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorView backend");
  ConstantFill(make_span(tv.data, tv.num_elements()), value);
}


template <typename StorageBackend, typename DataType, int dim>
void ConstantFill(const TensorListView<StorageBackend, DataType, dim> &tlv,
                  same_as_t<DataType> value = {}) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorListView backend");
  for (int i = 0; i < tlv.num_samples(); i++)
    ConstantFill(tlv[i], value);
}

template <typename C>
if_iterable<C, void> SequentialFill(C &&c, const element_t<C> &start_value = {}) {
  auto value = start_value;
  for (auto &x : c)
    x = value++;
}


template <typename StorageBackend, typename DataType, int dim>
void
SequentialFill(const TensorView<StorageBackend, DataType, dim> &tv,
               same_as_t<DataType> start_value = {}) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorView backend");
  SequentialFill(make_span(tv.data, tv.num_elements()), start_value);
}


template <typename StorageBackend, typename DataType, int dim>
void SequentialFill(const TensorListView<StorageBackend, DataType, dim> &tlv,
                  same_as_t<DataType> start_value = {}) {
  static_assert(is_cpu_accessible<StorageBackend>::value,
                "Function available only for CPU-accessible TensorListView backend");
  for (int i = 0; i < tlv.num_samples(); i++)
    SequentialFill(tlv[i], start_value);
}


template <typename TensorListView>
std::string BatchToStr(const TensorListView &batch, const std::string &sample_prefix = "Sample ") {
  std::stringstream ss;
  for (int i = 0; i < batch.num_samples(); i++) {
    ss << sample_prefix << i << ":";
    for (auto &x : make_span(batch[i].data, batch[i].num_elements()))
      ss << " " << x;
  }
  return ss.str();
}

}  // namespace dali

#endif  // DALI_TEST_TENSOR_TEST_UTILS_H_
