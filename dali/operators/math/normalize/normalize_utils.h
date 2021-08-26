// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_UTILS_H_
#define DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_UTILS_H_

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#include <algorithm>
#include "dali/core/tensor_view.h"
#include "dali/core/math_util.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {
namespace normalize {

// Pairwise, elementwise sum of arrays
template <typename T>
void SumArrays(T *const *arrays, int num_arrays, int64_t array_size) {
  if (num_arrays > 1) {
    int mid = num_arrays >> 1;
    if (mid > 1)
      SumArrays(arrays, mid, array_size);

    if (num_arrays - mid > 1)
      SumArrays(arrays + mid, num_arrays - mid, array_size);

    T *a0 = arrays[0];
    const T *a1 = arrays[mid];
    for (int64_t i = 0; i < array_size; i++) {
      a0[i] += a1[i];
    }
  }
}

/**
 * @brief Calculates the elementwise sum of all samples in the TensorList and stores it
 *        in the first sample.
 *
 * The function sums the samples pairwise to minimize error. Intermediate results are stored in the
 * supplied tensor list - only the tlv[0] contains well-defined data after a call to this function.
 *
 * @remarks The operation is destructive and the contents of other samples should not be relied
 *          upon after a call to this function.
 */
template <typename T>
void SumSamples(const TensorListView<StorageCPU, T> &tlv) {
  int n = tlv.num_samples();
  if (!n)
    return;  // empty list - nothing to do

  assert(is_uniform(tlv.shape));
  auto v = volume(tlv.shape[0]);
  SumArrays(tlv.data.data(), n, v);
}

inline int64_t ReducedVolume(const TensorListShape<> &shape, span<const int> axes) {
  int64_t v = 0;
  for (int i = 0; i < shape.num_samples(); i++) {
    auto sample_shape = shape.tensor_shape_span(i);
    int64_t sample_v = 1;
    for (auto a : axes)
      sample_v *= sample_shape[a];
    v += sample_v;
  }
  return v;
}


template <typename T>
void UniformFill(TensorList<CPUBackend> &tl, const T &value) {
  const auto &shape = tl.shape();
  for (int i = 0; i < shape.num_samples(); i++) {
    T *data = tl.mutable_tensor<T>(i);
    std::fill(data, data + shape.tensor_size(i), value);
  }
}

/**
 * @brief Calculates inverse of the standard deviation, adding epsilon to varianace
 *        and scaling the result
 *
 * The output elements are calculated as:
 * ```
 * inv = scale / sqrt(stddev * stddev + epsilon)
 * ```
 *
 * @param inv     output
 * @param stddev  the standard deviation
 * @param epsilon a small positive value added to the variance
 * @param scale   a multiplier, applied to the final result
 */
static void CalcInvStdDev(const TensorListView<StorageCPU, float> &inv,
                          const TensorListView<StorageCPU, const float> &stddev,
                          float epsilon,
                          float scale) {
  assert(inv.shape == stddev.shape);
  for (int i = 0; i < inv.shape.num_samples(); i++) {
    int64_t v = volume(inv.shape.tensor_shape_span(i));
    if (epsilon) {
      for (int64_t j = 0; j < v; j++) {
        inv.data[i][j] = scale * rsqrt(stddev.data[i][j] * stddev.data[i][j] + epsilon);
      }
    } else {
      for (int64_t j = 0; j < v; j++) {
        inv.data[i][j] = stddev.data[i][j] ? scale / stddev.data[i][j] : 0;
      }
    }
  }
}

/**
 * @brief Calculates `mul/sqrt(data[i] * rdiv + eps)` for nonzero argument of sqrt and 0 otherwise
 *
 * @param rdiv reciprocal of the divisor
 * @param mul scaling factor
 * @param eps epsilon added to data[i] * rdiv to avoid reciprocals of small numbers
 *
 * @remarks The scaling is split into these two values for precision.
 */
static void ScaleRSqrtKeepZero(float *data, int64_t n, float eps, float rdiv, float mul) {
  int64_t i = 0;

#ifdef __SSE__
  // Vectorized version of the loop below

  // We calculate the following:
  // mul * rsqrt(data[i] * rdiv + eps)
  //
  // rsqrt needs an extra step of Newton-Raphson refinement:
  // y = approx_rsqrt(x)  // rough approximation
  // precise = y * (3 - x*y*y) * 0.5
  //
  // The multiplication by half is fused with mul, so the vectorized multipliplier is halved

  __m128 rdivx4 = _mm_set1_ps(rdiv);
  __m128 mulx4 = _mm_set1_ps(mul * 0.5f);  // halve here - save a multiplication in the inner loop
  __m128 three = _mm_set1_ps(3.0f);
  if (eps) {  // epsilon is present - no need for masking, but we need to add it
    __m128 epsx4 = _mm_set1_ps(eps);
    for (; i + 4 <= n; i += 4) {
      __m128 x = _mm_loadu_ps(&data[i]);
      x = _mm_mul_ps(x, rdivx4);
      x = _mm_add_ps(x, epsx4);
      __m128 y = _mm_rsqrt_ps(x);
      __m128 y2 = _mm_mul_ps(y, y);
      __m128 xy2 = _mm_mul_ps(x, y2);
      __m128 three_minus_xy2 = _mm_sub_ps(three, xy2);
      y = _mm_mul_ps(y, three_minus_xy2);
      y = _mm_mul_ps(y, mulx4);
      _mm_storeu_ps(&data[i], y);
    }
  } else {  // no epsilon - need to mask zeros
    __m128 zero = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
      __m128 x = _mm_loadu_ps(&data[i]);
      x = _mm_mul_ps(x, rdivx4);
      __m128 mask = _mm_cmpneq_ps(x, zero);
      __m128 y = _mm_rsqrt_ps(x);
      y = _mm_and_ps(y, mask);  // mask whatever garbage was there and force to zero
      __m128 y2 = _mm_mul_ps(y, y);
      __m128 xy2 = _mm_mul_ps(x, y2);
      __m128 three_minus_xy2 = _mm_sub_ps(three, xy2);
      y = _mm_mul_ps(y, three_minus_xy2);
      y = _mm_mul_ps(y, mulx4);
      _mm_storeu_ps(&data[i], y);
    }
  }
#endif

  if (eps) {
    for (; i < n; i++)
      data[i] = rsqrt(data[i] * rdiv + eps) * mul;
  } else {
    for (; i < n; i++) {
      float x = data[i] * rdiv;
      data[i] = x ? rsqrt(x) * mul : 0;
    }
  }
}

static void ScaleRSqrtKeepZero(const TensorView<StorageCPU, float> &inout,
                               float eps, float rdiv, float mul) {
  ScaleRSqrtKeepZero(inout.data, inout.num_elements(), eps, rdiv, mul);
}

static void SumSquare2InvStdDev(const TensorView<StorageCPU, float> &inout,
                                const TensorShape<> &data_shape,
                                int degrees_of_freedom,
                                double epsilon,
                                double scale) {
  if (inout.num_elements() == 0) {
    return;
  }
  assert(data_shape.num_elements() >= inout.num_elements());

  int64_t v = data_shape.num_elements() / inout.num_elements();
  float rdiv = 0;
  if (v > degrees_of_freedom) {
    rdiv = static_cast<float>(1.0 / (v - degrees_of_freedom));  // reciprocal in double precision
  } else {
    if (epsilon == 0) {
      rdiv = 1;
      scale = 0;
    }
  }
  ScaleRSqrtKeepZero(inout, static_cast<float>(epsilon), rdiv, scale);
}


}  // namespace normalize
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_UTILS_H_
