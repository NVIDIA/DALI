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

#include "dali/util/npp.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"

namespace dali {

int NPPInterpForDALIInterp(DALIInterpType type, NppiInterpolationMode *npp_type) {
  switch (type) {
  case DALI_INTERP_NN:
    *npp_type =  NPPI_INTER_NN;
    break;
  case DALI_INTERP_LINEAR:
    *npp_type =  NPPI_INTER_LINEAR;
    break;
  case DALI_INTERP_CUBIC:
    *npp_type =  NPPI_INTER_CUBIC;
    break;
  default:
    return DALIError;
  }
  return DALISuccess;
}

namespace {

NppStreamContext GetNppContextImpl() {
  NppStreamContext ctx;
  ctx.hStream = 0;
  CUDA_CALL(cudaGetDevice(&ctx.nCudaDeviceId));
  int driver_ver, runtime_ver;
  cudaDriverGetVersion(&driver_ver);
  cudaRuntimeGetVersion(&runtime_ver);

  CUDA_CALL(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                                   cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId));

  CUDA_CALL(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                                   cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId));

  CUDA_CALL(cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags));
  cudaDeviceProp dev_prop;
  CUDA_CALL(cudaGetDeviceProperties(&dev_prop, ctx.nCudaDeviceId));

  ctx.nMultiProcessorCount = dev_prop.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = dev_prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = dev_prop.sharedMemPerBlock;
  return ctx;
}

}  // namespace

NppStreamContext GetNppContext(cudaStream_t stream) {
  static NppStreamContext ctx = GetNppContextImpl();
  auto ret_ctx = ctx;
  ret_ctx.hStream = stream;
  return ret_ctx;
}

}  // namespace dali
