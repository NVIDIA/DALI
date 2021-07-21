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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "dali/test/dali_cuda_finalize_test.h"

#if (CUDART_VERSION >= 10200 && CUDART_VERSION < 11100)
// add this alignment to work around a patchelf bug/feature which
// changes TLS alignment and break DALI interoperability with CUDA RT
alignas(0x1000) thread_local volatile bool __dali_kernel_test_force_tls_align;

void __dali_kernel_test_force_tls_align_fun(void) {
  __dali_kernel_test_force_tls_align = 0;
}
#else
void __dali_kernel_test_force_tls_align_fun(void) {}
#endif

int main(int argc, char **argv) {
  __dali_kernel_test_force_tls_align_fun();
  ::testing::InitGoogleTest(&argc, argv);

  // Gets hold of the event listener list.
  ::testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
  // Adds a listener to the end.  googletest takes the ownership.
  listeners.Append(new dali::CudaFinalizeEventListener);

  return RUN_ALL_TESTS();
}
