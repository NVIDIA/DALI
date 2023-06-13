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

#include <cmath>
#include <vector>
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"
#include "dali/core/math_util.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
HaarWavelet<T>::HaarWavelet(const std::vector<T> &args) {
  if (args.size() != 0) {
    throw new std::invalid_argument("HaarWavelet doesn't accept any arguments.");
  }
}

template <typename T>
__device__ T HaarWavelet<T>::operator()(const T &t) const {
  if (0.0 <= t && t < 0.5) {
    return 1.0;
  }
  if (0.5 <= t && t < 1.0) {
    return -1.0;
  }
  return 0.0;
}

template class HaarWavelet<float>;
template class HaarWavelet<double>;

template <typename T>
GaussianWavelet<T>::GaussianWavelet(const std::vector<T> &args) {
  if (args.size() != 1) {
    throw new std::invalid_argument("GaussianWavelet accepts exactly one argument - n.");
  }
  if (args[0] < 1.0 || args[0] > 8.0) {
    throw new std::invalid_argument(
      "GaussianWavelet's argument n should be integer from range [1,8].");
  }
  this->n = args[0];
}

template <typename T>
__device__ T GaussianWavelet<T>::operator()(const T &t) const {
  T expTerm = std::exp(-std::pow(t, 2.0));
  T sqrtTerm = 1.2533141373155001;  // std::sqrt(M_PI/2.0)
  switch (static_cast<int>(n)) {
  case 1:
    return -2.0*t*expTerm/std::sqrt(sqrtTerm);
  case 2:
    return (-4.0*std::pow(t, 2.0)+2.0)*expTerm/std::sqrt(3.0*sqrtTerm);
  case 3:
    return (8.0*std::pow(t, 3.0)-12.0*t)*expTerm/std::sqrt(15.0*sqrtTerm);
  case 4:
    return (-48.0*std::pow(t, 2.0)+16.0*std::pow(t, 4.0)+12.0)*expTerm/std::sqrt(105.0*sqrtTerm);
  case 5:
    return (-32.0*std::pow(t, 5.0)+160.0*std::pow(t, 3.0)-120.0*t)*
              expTerm/std::sqrt(945.0*sqrtTerm);
  case 6:
    return (-64.0*std::pow(t, 6.0)+480.0*std::pow(t, 4.0)-720.0*std::pow(t, 2.0)+120.0)*
              expTerm/std::sqrt(10395.0*sqrtTerm);
  case 7:
    return (128.0*std::pow(t, 7.0)-1344.0*std::pow(t, 5.0)+3360.0*std::pow(t, 3.0)-1680.0*t)*
              expTerm/std::sqrt(135135.0*sqrtTerm);
  case 8:
    return (256.0*std::pow(t, 8.0)-3584.0*std::pow(t, 6.0)+13440.0*std::pow(t, 4.0)-13440.0*
              std::pow(t, 2.0)+1680.0)*expTerm/std::sqrt(2027025.0*sqrtTerm);
  }
}

template class GaussianWavelet<float>;
template class GaussianWavelet<double>;

template <typename T>
MexicanHatWavelet<T>::MexicanHatWavelet(const std::vector<T> &args) {
  if (args.size() != 1) {
    throw new std::invalid_argument("MexicanHatWavelet accepts exactly one argument - sigma.");
  }
  this->sigma = args[0];
}

template <typename T>
__device__ T MexicanHatWavelet<T>::operator()(const T &t) const {
  return 2.0/(std::sqrt(3.0*sigma)*std::pow(M_PI, 0.25))*(1.0-std::pow(t/sigma, 2.0))*
            std::exp(-std::pow(t, 2.0)/(2.0*std::pow(sigma, 2.0)));
}

template class MexicanHatWavelet<float>;
template class MexicanHatWavelet<double>;

template <typename T>
MorletWavelet<T>::MorletWavelet(const std::vector<T> &args) {
  if (args.size() != 1) {
    throw new std::invalid_argument("MorletWavelet accepts exactly 1 argument - C.");
  }
  this->C = args[0];
}

template <typename T>
__device__ T MorletWavelet<T>::operator()(const T &t) const {
  return C * std::exp(-std::pow(t, 2.0) / 2.0) * std::cos(5.0 * t);
}

template class MorletWavelet<float>;
template class MorletWavelet<double>;

template <typename T>
ShannonWavelet<T>::ShannonWavelet(const std::vector<T> &args) {
  if (args.size() != 2) {
    throw new std::invalid_argument(
      "ShannonWavelet accepts exactly 2 arguments -> fb, fc in that order.");
  }
  this->fb = args[0];
  this->fc = args[1];
}

template <typename T>
__device__ T ShannonWavelet<T>::operator()(const T &t) const {
  auto res = std::cos((T)(2.0*M_PI)*fc*t)*std::sqrt(fb);
  return t == 0.0 ? res : res*std::sin(t*fb*(T)(M_PI))/(t*fb*(T)(M_PI));
}

template class ShannonWavelet<float>;
template class ShannonWavelet<double>;

template <typename T>
FbspWavelet<T>::FbspWavelet(const std::vector<T> &args) {
  if (args.size() != 3) {
    throw new std::invalid_argument(
      "FbspWavelet accepts exactly 3 arguments -> m, fb, fc in that order.");
  }
  this->m = args[0];
  this->fb = args[1];
  this->fc = args[2];
}

template <typename T>
__device__ T FbspWavelet<T>::operator()(const T &t) const {
  auto res = std::cos((T)(2.0*M_PI)*fc*t)*std::sqrt(fb);
  return t == 0.0 ? res : res*std::pow(std::sin((T)(M_PI)*t*fb/m)/((T)(M_PI)*t*fb/m), m);
}

template class FbspWavelet<float>;
template class FbspWavelet<double>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
