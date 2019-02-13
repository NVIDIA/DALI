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

#ifndef DALI_KERNELS_TEST_TENSOR_TEST_UTILS_H_
#define DALI_KERNELS_TEST_TENSOR_TEST_UTILS_H_

#include <gtest/gtest.h>
#include <functional>
#include <cmath>
#include <utility>
#include <random>
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/backend_tags.h"
#include "dali/kernels/util.h"

namespace dali {
namespace kernels {

namespace detail {
  static constexpr int DefaultMaxErrors = 100;
}  // namespace detail

// COMPARISON

template <int dim1, int dim2>
void CheckEqual(
    const TensorListShape<dim1> &s1,
    const TensorListShape<dim2> &s2,
    int &max_errors) {
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

template <typename T1, typename T2, int dim1, int dim2, typename ElementsOkFunc>
void Check(
    const TensorView<StorageCPU, T1, dim1> &tv1,
    const TensorView<StorageCPU, T2, dim2> &tv2,
    ElementsOkFunc eq,
    int &max_errors) {
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
        EXPECT_TRUE(eq(tv1.data[i], tv2.data[i]))
          << "Failed at offset " << i << ", pos = " << pos
          << " tv1[" << i << "] = " << printable(tv1.data[i])
          << " tv2[" << i << "] = " << printable(tv2.data[i]);
      }
    }

    for (int j = dim-1; j >= 0; j--) {
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

struct EqualEps {
  EqualEps() = default;
  explicit EqualEps(double eps) : eps(eps) {}

  template <typename T1, typename T2>
  bool operator()(const T1 &a, const T2 &b) const {
    return std::abs(b-a) <= eps;
  }
  double eps = 1e-6;
};

template <typename T1, typename T2, int dim1, int dim2, typename ElementsOkFunc = Equal>
void Check(
    const TensorView<StorageCPU, T1, dim1> &tv1,
    const TensorView<StorageCPU, T2, dim2> &tv2,
    ElementsOkFunc eq = {}) {
  int max_errors = detail::DefaultMaxErrors;
  Check(tv1, tv2, std::move(eq), max_errors);
}


template <typename T1, typename T2, int dim1, int dim2, typename ElementsOkFunc = Equal>
void Check(
    const TensorListView<StorageCPU, T1, dim1> &tv1,
    const TensorListView<StorageCPU, T2, dim2> &tv2,
    ElementsOkFunc eq = {}) {
  int max_errors = detail::DefaultMaxErrors;
  CheckEqual(tv1.shape, tv2.shape, max_errors);
  int n = tv1.num_samples();
  for (int i = 0; i < n; i++) {
    Check(tv1[i], tv2[i], eq, max_errors);
  }
}


// FILLING

template <typename Collection, typename Generator>
if_iterable<Collection, void> Fill(Collection &&collection, Generator &generator) {
  for (auto &x : collection)
    x = generator();
}

template <typename DataType, int ndim, typename Generator>
void Fill(const TensorView<StorageCPU, DataType, ndim> &tv, Generator &generator) {
  Fill(make_span(tv.data, tv.num_elements()), generator);
}

template <typename DataType, int ndim, typename Generator>
void Fill(const TensorListView<StorageCPU, DataType, ndim> &tlv, Generator &generator) {
  Fill(make_span(tlv.data, tlv.num_elements()), generator);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value,
                        std::uniform_real_distribution<T>>::type
uniform_distribution(T lo, T hi) {
  return std::uniform_real_distribution<T>(lo, hi);
}
template <typename T>
typename std::enable_if<std::is_integral<T>::value,
                        std::uniform_int_distribution<T>>::type
uniform_distribution(T lo, T hi) {
  return std::uniform_int_distribution<T>(lo, hi);
}


template <typename Collection, typename RandomGenerator>
if_iterable<Collection, void> UniformRandomFill(
    Collection &&c,
    RandomGenerator &rng, element_t<Collection> lo, element_t<Collection> hi) {
  auto dist = uniform_distribution(lo, hi);
  auto generator = [&]() { return dist(rng); };
  Fill(std::forward<Collection>(c), generator);
}

template <typename DataType, int ndim, typename RandomGenerator>
void UniformRandomFill(
    const TensorListView<StorageCPU, DataType, ndim> &tlv,
    RandomGenerator &generator,
    same_as_t<DataType> lo, same_as_t<DataType> hi) {
  UniformRandomFill(make_span(tlv.data, tlv.num_elements()), generator, lo, hi);
}

template <typename DataType, int ndim, typename RandomGenerator>
void UniformRandomFill(
    const TensorView<StorageCPU, DataType, ndim> &tv,
    RandomGenerator &generator,
    same_as_t<DataType> lo, same_as_t<DataType> hi) {
  UniformRandomFill(make_span(tv.data, tv.num_elements()), generator, lo, hi);
}

template <typename C>
if_iterable<C, void> ConstantFill(C &&c, const element_t<C> &value = {}) {
  for (auto &x : c)
    x = value;
}

template <typename DataType, int dim>
void ConstantFill(
    const TensorListView<StorageCPU, DataType, dim> &tlv,
    same_as_t<DataType> value = {}) {
  ConstantFill(make_span(tlv.data, tlv.num_elements()), value);
}

template <typename DataType, int dim>
void ConstantFill(
    const TensorView<StorageCPU, DataType, dim> &tv,
    same_as_t<DataType> value = {}) {
  ConstantFill(make_span(tv.data, tv.num_elements()), value);
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TEST_TENSOR_TEST_UTILS_H_
