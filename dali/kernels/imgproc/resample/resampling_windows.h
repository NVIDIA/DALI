// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_WINDOWS_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_WINDOWS_H_

#include <cuda_runtime.h>
#include <functional>
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

inline __host__ __device__ float GaussianWindow(float x, float sigma = 0.5f) {
  x /= sigma;
  return expf(-0.5f * x * x);
}

inline __host__ __device__ float ExpMinusX2(float x) {
  return expf(-x * x);
}

inline __host__ __device__ float TriangularWindow(float x) {
  float t = 1 - fabsf(x);
  return t < 0 ? 0 : t;
}

inline __host__ __device__ float RectangularWindow(float x) {
  return -0.5f <= x && x < 0.5f ? 1 : 0;
}

inline __host__ __device__ float sinc(float x) {
  return x ? sinf(x * M_PI) / (x * M_PI) : 1;
}

inline __host__ __device__ float LanczosWindow(float x, float a) {
  if (fabsf(x) >= a)
    return 0.0f;
  return sinc(x)*sinc(x / a);
}

inline __host__ __device__ float Lanczos3Window(float x) {
  return LanczosWindow(x, 3);
}

inline __host__ __device__ float CubicWindow(float x) {
  x = fabsf(x);
  if (x >= 2)
    return 0;

  float x2 = x*x;
  float x3 = x2*x;
  if (x > 1)
    return -0.5f*x3 + 2.5f*x2 - 4.0f*x + 2.0f;
  else
    return 1.5f*x3 - 2.5f*x2 + 1.0f;
}

struct FilterWindow {
  float size, anchor, scale;
  std::function<float(float)> func;
  float operator()(float x) const {
    return func((x - anchor) * scale);
  }

  int support() const {
    return ceilf(size);
  }
};

inline FilterWindow GaussianFilter(float radius, float sigma = 0) {
  float scale = sigma ? M_SQRT1_2 / sigma : 2 / radius;
  radius = floorf(radius + 0.4f) + 0.5f;
  return { 2*radius, radius, scale, ExpMinusX2 };
}

inline FilterWindow TriangularFilter(float radius) {
  if (radius < 1)
    return { 1, 2, 1, TriangularWindow };
  return { 2*radius, radius, 1/radius, TriangularWindow };
}

inline  FilterWindow CubicFilter() {
  return { 4, 2, 1, CubicWindow };
}

inline  FilterWindow Lanczos3Filter() {
  return { 6, 3, 1, Lanczos3Window };
}

inline  FilterWindow LanczosFilter(float a) {
  return { 2*a, a, 1, [a](float x) { return LanczosWindow(x, a); } };
}

inline  FilterWindow LinearFilter() {
  return { 2, 1, 1, TriangularWindow };
}

inline  FilterWindow NNFilter() {
  return { 1, 0.5f, 1, RectangularWindow };
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_WINDOWS_H_
