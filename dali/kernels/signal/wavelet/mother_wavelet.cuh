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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

#include <vector>

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
  HaarWavelet(const std::vector<T> &args);
  ~HaarWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class DaubechiesWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  DaubechiesWavelet() = default;
  DaubechiesWavelet(const std::vector<T> &args);
  ~DaubechiesWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class SymletWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  SymletWavelet() = default;
  SymletWavelet(const std::vector<T> &args);
  ~SymletWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class CoifletWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  CoifletWavelet() = default;
  CoifletWavelet(const std::vector<T> &args);
  ~CoifletWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class MeyerWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  MeyerWavelet() = default;
  MeyerWavelet(const std::vector<T> &args);
  ~MeyerWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class GaussianWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  GaussianWavelet() = default;
  GaussianWavelet(const std::vector<T> &args);
  ~GaussianWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class MexicanHatWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  MexicanHatWavelet() = default;
  MexicanHatWavelet(const std::vector<T> &args);
  ~MexicanHatWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;

 private:
  T sigma;
};

template <typename T>
class MorletWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  MorletWavelet() = default;
  MorletWavelet(const std::vector<T> &args);
  ~MorletWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;

 private:
  T C;
};

template <typename T>
class ShannonWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  ShannonWavelet() = default;
  ShannonWavelet(const std::vector<T> &args);
  ~ShannonWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;
};

template <typename T>
class FbspWavelet {
  static_assert(std::is_floating_point<T>::value,
    "Data type should be floating point");
 public:
  FbspWavelet() = default;
  FbspWavelet(const std::vector<T> &args);
  ~FbspWavelet() = default;

  __device__ T operator()(const T &t, const T &a, const T &b) const;

 private:
  T m;
  T fb;
  T fc;
};

}  // namespace signal
}  // namespace kernel
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WAVELET_MOTHER_WAVELET_CUH_
