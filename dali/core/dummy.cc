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

#include <cuda_runtime_api.h>
#include "dali/core/api_helper.h"

namespace dali {

#if (CUDART_VERSION >= 10200 && CUDART_VERSION < 11100)
// add this alignment to work around a patchelf bug/feature which
// changes TLS alignment and break DALI interoperability with CUDA RT
alignas(0x1000) thread_local volatile bool __dali_core_force_tls_align;

DLL_PUBLIC void __dali_core_force_tls_align_fun(void) {
  __dali_core_force_tls_align = 0;
}
#else
DLL_PUBLIC void __dali_core_test_force_tls_align_fun(void) {}
#endif

}  // namespace dali
