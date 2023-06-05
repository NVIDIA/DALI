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
__device__ T HaarWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  T x = std::pow(2.0, a) * t - b;
  if (0.0 <= x && x < 0.5) {
    return std::pow(2.0, a / 2.0);
  }
  if (0.5 <= x && x < 1.0) {
    return -std::pow(2.0, a / 2.0);
  }
  return 0.0;
}

template class HaarWavelet<float>;
template class HaarWavelet<double>;

template <typename T>
DaubechiesWavelet<T>::DaubechiesWavelet(const std::vector<T> &args) {

}

template <typename T>
__device__ T DaubechiesWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  return 0.0;
}

template class DaubechiesWavelet<float>;
template class DaubechiesWavelet<double>;

template <typename T>
SymletWavelet<T>::SymletWavelet(const std::vector<T> &args) {

}

template <typename T>
__device__ T SymletWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  return 0.0;
}

template class SymletWavelet<float>;
template class SymletWavelet<double>;

template <typename T>
CoifletWavelet<T>::CoifletWavelet(const std::vector<T> &args) {

}

template <typename T>
__device__ T CoifletWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  return 0.0;
}

template class CoifletWavelet<float>;
template class CoifletWavelet<double>;

template <typename T>
MeyerWavelet<T>::MeyerWavelet(const std::vector<T> &args) {
  if (args.size() != 0) {
    throw new std::invalid_argument("MeyerWavelet doesn't accept any arguments.");
  }
}

template <typename T>
__device__ T MeyerWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  T x = std::pow(2.0, a) * t - b - 0.5;
  T psi1 = (4/(3*M_PI)*x*std::cos((2*M_PI)/3*x)-1/M_PI*std::sin((4*M_PI)/3*x))/(x-16/9*std::pow(x, 3.0));
  T psi2 = (8/(3*M_PI)*x*std::cos(8*M_PI/3*x)+1/M_PI*std::sin((4*M_PI)/3)*x)/(x-64/9*std::pow(x, 3.0));
  return std::pow(2.0, a / 2.0) * (psi1 + psi2);
}

template class MeyerWavelet<float>;
template class MeyerWavelet<double>;

template <typename T>
GaussianWavelet<T>::GaussianWavelet(const std::vector<T> &args) {

}

template <typename T>
__device__ T GaussianWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  return 0.0;
}

template class GaussianWavelet<float>;
template class GaussianWavelet<double>;

template <typename T>
MexicanHatWavelet<T>::MexicanHatWavelet(const std::vector<T> &args) {
  if (args.size() != 1) {
    throw new std::invalid_argument("MexicanHatWavelet accepts exactly one argument - sigma.");
  }
  this->sigma = *args.begin();
}

template <typename T>
__device__ T MexicanHatWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  T x = std::pow(2.0, a) * t - b;
  return std::pow(2.0, a / 2.0) * (2/(std::sqrt(3*sigma)*std::pow(M_PI, 0.25))*(1-std::pow(x/sigma, 2.0))*std::exp(-std::pow(x, 2.0)/(2*std::pow(sigma, 2.0))));
}

template class MexicanHatWavelet<float>;
template class MexicanHatWavelet<double>;

template <typename T>
MorletWavelet<T>::MorletWavelet(const std::vector<T> &args) {
  if (args.size() != 1) {
    throw new std::invalid_argument("MorletWavelet accepts exactly 1 argument - C.");
  }
  this->C = *args.begin();
}

template <typename T>
__device__ T MorletWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  T x = std::pow(2.0, a) * t - b;
  return std::pow(2.0, a / 2.0) * (C * std::exp(-std::pow(x, 2.0)) * std::cos(5 * x));
}

template class MorletWavelet<float>;
template class MorletWavelet<double>;

template <typename T>
ShannonWavelet<T>::ShannonWavelet(const std::vector<T> &args) {
  if (args.size() != 0) {
    throw new std::invalid_argument("ShannonWavelet doesn't accept any arguments.");
  }
}

template <typename T>
__device__ T ShannonWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  T x = std::pow(2.0, a) * t - b;
  return std::pow(2.0, a / 2.0) * (sinc(x - 0.5) - 2 * sinc(2 * x - 1));
}

template class ShannonWavelet<float>;
template class ShannonWavelet<double>;

template <typename T>
FbspWavelet<T>::FbspWavelet(const std::vector<T> &args) {
  if (args.size() != 0) {
    throw new std::invalid_argument("FbspWavelet accepts exactly 3 arguments -> m, fb, fc in that order.");
  }
  this->m = *args.begin();
  this->fb = *(args.begin()+1);
  this->fc = *(args.begin()+2);
}

template <typename T>
__device__ T FbspWavelet<T>::operator()(const T &t, const T &a, const T &b) const {
  T x = std::pow(2.0, a) * t - b;
  return std::pow(2.0, a / 2.0) * (std::sqrt(fb)*std::pow(sinc(x/std::pow(fb, m)), m)*std::exp(2*M_PI*fc*x));
}

template class FbspWavelet<float>;
template class FbspWavelet<double>;

}  // namespace signal
}  // namespace kernel
}  // namespace dali
