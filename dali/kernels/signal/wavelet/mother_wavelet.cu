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

#include <vector>
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
__device__
T HaarWavelet(T t, T a, T b) {
  T x = std::pow(2.0, a) - b;
  if (0.0 <= x && x < 0.5) {
    return std::pow(2.0, a / 2.0);
  }
  if (0.5 <= x && x < 1.0) {
    return -std::pow(2.0, a / 2.0);
  }
  return 0.0;
}

template <typename T>
__device__
T DaubechiesWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T SymletWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T CoifletWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T BiorthogonalWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T MeyerWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T GaussianWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T MexicanHatWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T MorletWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T ComplexGaussianWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T ShannonWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T FbspWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
__device__
T ComplexMorletWavelet(T t, T a, T b) {
  return 0.0;
}

template <typename T>
MotherWavelet<T>::MotherWavelet(const WaveletName& name) {
  switch(name) {
    case WaveletName::HAAR:
    waveletFunc = &HaarWavelet;
    break;

    case WaveletName::DB:
    waveletFunc = &DaubechiesWavelet;
    break;

    case WaveletName::SYM:
    waveletFunc = &SymletWavelet;
    break;

    case WaveletName::COIF:
    waveletFunc = &CoifletWavelet;
    break;

    case WaveletName::BIOR:
    waveletFunc = &BiorthogonalWavelet;
    break;

    case WaveletName::MEY:
    waveletFunc = &MeyerWavelet;
    break;

    case WaveletName::GAUS:
    waveletFunc = &GaussianWavelet;
    break;

    case WaveletName::MEXH:
    waveletFunc = &MexicanHatWavelet;
    break;

    case WaveletName::MORL:
    waveletFunc = &MorletWavelet;
    break;

    case WaveletName::CGAU:
    waveletFunc = &ComplexGaussianWavelet;
    break;

    case WaveletName::SHAN:
    waveletFunc = &ShannonWavelet;
    break;

    case WaveletName::FBSP:
    waveletFunc = &FbspWavelet;
    break;

    case WaveletName::CMOR:
    waveletFunc = &ComplexMorletWavelet;
    break;
    
    default:
    throw new std::invalid_argument("Unknown wavelet name.");
  }
}

}  // namespace signal
}  // namespace kernel
}  // namespace dali
