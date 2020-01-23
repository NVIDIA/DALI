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

int64_t ReducedVolume(const TensorListShape<> &shape, span<int> axes) {
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
  T *data = tl.mutable_data<T>();
  std::fill(data, data + tl.size(), value);
}

void CalcInvStdDev(const TensorListView<StorageCPU, float> &inv,
                   const TensorListView<StorageCPU, const float> &stddev,
                   float scale) {
  assert(inv.shape == stddev.shape);
  for (int i = 0; i < inv.shape.num_samples(); i++) {
    auto v = volume(inv.shape.tensor_shape_span(i));
    for (int j = 0; j < v; j++) {
      inv.data[i][j] = stddev.data[i][j] ? scale / stddev.data[i][j] : 0;
    }
  }
}

/**
 * @brief Calculates mul/sqrt(x * rdiv) for nonzero x and keeps 0 when x == 0
 *
 * @param rdiv reciprocal of the divisor
 * @param mul scaling factor
 *
 * @remarks The scaling is split into these two values for precision.
 */
static void ScaleRSqrtKeepZero(float *data, int64_t n, float rdiv, float mul) {
  int i = 0;

#ifdef __SSE__
  // vectorized version of the loop below
  __m128 rdivx4 = _mm_set1_ps(rdiv);
  __m128 mulx4 = _mm_set1_ps(mul);
  __m128 zero = _mm_setzero_ps();
  for (; i + 4 <= n; i += 4) {
    __m128 x = _mm_loadu_ps(&data[i]);
    x = _mm_mul_ps(x, rdivx4);
    __m128 mask = _mm_cmpneq_ps(x, zero);
    x = _mm_rsqrt_ps(x);
    x = _mm_and_ps(x, mask);
    x = _mm_mul_ps(x, mulx4);
    _mm_storeu_ps(&data[i], x);
  }
#endif

  for (; i < n; i++)
    if (data[i])
      data[i] = rsqrt(data[i] * rdiv) * mul;
}

static void ScaleRSqrtKeepZero(const TensorView<StorageCPU, float> &inout, float rdiv, float mul) {
  ScaleRSqrtKeepZero(inout.data, inout.num_elements(), rdiv, mul);
}

static void SumSquare2InvStdDev(const TensorView<StorageCPU, float> &inout,
                                const TensorShape<> &data_shape,
                                double scale) {
  int64_t v = data_shape.num_elements() / inout.num_elements();
  float rdiv = static_cast<float>(1.0 / v);  // reciprocal in double precision
  ScaleRSqrtKeepZero(inout, rdiv, scale);
}


}  // namespace normalize
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_NORMALIZE_NORMALIZE_UTILS_H_
