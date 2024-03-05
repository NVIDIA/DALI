// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dali/core/mm/detail/stream_id_hint.h"
#include "dali/core/cuda_error.h"

using cuStreamGetId_t = CUresult(CUstream, unsigned long long *);  // NOLINT(runtime/int)

namespace {

inline int getTID() {
  return syscall(SYS_gettid);
}

constexpr uint64_t MakeLegacyStreamId(int dev, int tid) {
  return (uint64_t)dev << 32 | tid;
}

CUresult cuStreamGetIdFallback(CUstream stream, unsigned long long *id) {  // NOLINT(runtime/int)
  // If the stream handle is a pseudohandle, use some special treatment....
  if (stream == 0 || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD) {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess)
      return CUDA_ERROR_INVALID_CONTEXT;
    // If we use a per-thread stream, get TID; otherwise use -1 as a pseudo-tid
    *id = MakeLegacyStreamId(dev, stream == CU_STREAM_PER_THREAD ? getTID() : -1);
    return CUDA_SUCCESS;
  } else {
    // Otherwise just use the handle - it's not perfactly safe, but should do.
    *id = (uint64_t)stream;
    return CUDA_SUCCESS;
  }
}

cuStreamGetId_t *getRealStreamIdFunc() {
  static cuStreamGetId_t *fn = []() {
    void *sym = nullptr;
    // If it fails, we'll just return nullptr.
#if CUDA_VERSION >= 12000
    (void)cuGetProcAddress("cuStreamGetId", &sym, 12000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
#else
    (void)cuGetProcAddress("cuStreamGetId", &sym, 12000, CU_GET_PROC_ADDRESS_DEFAULT);
#endif
    return reinterpret_cast<cuStreamGetId_t *>(sym);
  }();
  return fn;
}

inline bool hasPreciseHint() {
  static bool ret = getRealStreamIdFunc() != nullptr;
  return ret;
}

CUresult cuStreamGetIdBootstrap(CUstream stream, unsigned long long *id);  // NOLINT(runtime/int)

cuStreamGetId_t *_cuStreamGetId = cuStreamGetIdBootstrap;

CUresult cuStreamGetIdBootstrap(CUstream stream, unsigned long long *id) {  // NOLINT(runtime/int)
  cuStreamGetId_t *realFunc = getRealStreamIdFunc();
  if (realFunc)
    _cuStreamGetId = realFunc;
  else
    _cuStreamGetId = cuStreamGetIdFallback;

  return _cuStreamGetId(stream, id);
}

}  // namespace

namespace dali {

DLL_PUBLIC bool stream_id_hint::is_unambiguous() {
  return hasPreciseHint();
}

DLL_PUBLIC uint64_t stream_id_hint::from_handle(CUstream stream) {
  static auto initResult = cuInit(0);
  (void)initResult;
  unsigned long long id;  // NOLINT(runtime/int)
  CUDA_CALL(_cuStreamGetId(stream, &id));
  return id;
}

}  // namespace dali
