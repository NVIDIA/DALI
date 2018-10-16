// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_HALF_H_
#define DALI_UTIL_HALF_H_


// Macros for writing RunImpl templated code for derived operators
template <typename Out>
class CastOut {
 public:
  inline Out operator()(float arg)        { return static_cast<Out>(arg); }
};

#ifndef __CUDA_ARCH__
#ifdef __F16C__
  #include <x86intrin.h>

template <>
class CastOut<uint16_t> {
 public:
  inline uint16_t operator()(float arg)   { return _cvtss_sh(arg, 0); }
};

  typedef uint16_t _float16;
#else
  #include "dali/util/half.hpp"

  typedef half_float::half _float16;
#endif
#else
  typedef __half _float16;
#endif  //  __CUDA_ARCH__

#define RUN_IMPL_CPU(ws, idx)    RUN_IMPL(ws, idx, _float16)
#define RUN_IMPL_GPU(ws, idx)    RUN_IMPL(ws, idx, float16)

#define RUN_IMPLEMENT(ws, idx, out, cast)       \
     DataDependentSetup(ws, idx);               \
     if (output_type_ == DALI_FLOAT16)          \
        RunHelper<out, cast>(ws, idx);          \
     else                                       \
        DALI_IMAGE_TYPE_SWITCH_NO_FLOAT16(      \
            output_type_, imgType,              \
            RunHelper<imgType, CastOut<imgType>>(ws, idx))

#define RUN_IMPL(ws, idx, out) RUN_IMPLEMENT(ws, idx, out, CastOut<out>)

#endif  // DALI_UTIL_HALF_H_
