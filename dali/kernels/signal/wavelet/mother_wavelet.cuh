// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_
#define DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {

// wavelets are represented by functors
// they can store any necessary parameters
// they must overload () operator

template <typename T>
class HaarWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  HaarWavelet() = default;
  explicit HaarWavelet(const std::vector<T> &args);
  ~HaarWavelet() = default;

  __device__ T operator()(const T &t) const;
};

template <typename T>
class MeyerWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  MeyerWavelet() = default;
  explicit MeyerWavelet(const std::vector<T> &args);
  ~MeyerWavelet() = default;

  __device__ T operator()(const T &t) const;
};

template <typename T>
class MexicanHatWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  MexicanHatWavelet() = default;
  explicit MexicanHatWavelet(const std::vector<T> &args);
  ~MexicanHatWavelet() = default;

  __device__ T operator()(const T &t) const;

 private:
  T sigma;
};

template <typename T>
class MorletWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  MorletWavelet() = default;
  explicit MorletWavelet(const std::vector<T> &args);
  ~MorletWavelet() = default;

  __device__ T operator()(const T &t) const;

 private:
  T C;
};

template <typename T>
class ShannonWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  ShannonWavelet() = default;
  explicit ShannonWavelet(const std::vector<T> &args);
  ~ShannonWavelet() = default;

  __device__ T operator()(const T &t) const;

 private:
  T fb;
  T fc;
};

template <typename T>
class FbspWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  FbspWavelet() = default;
  explicit FbspWavelet(const std::vector<T> &args);
  ~FbspWavelet() = default;

  __device__ T operator()(const T &t) const;

 private:
  T m;
  T fb;
  T fc;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_
