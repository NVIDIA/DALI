/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
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

#include "dali/pipeline/operators/transpose/cutt/CudaMemcpy.h"

#include "dali/pipeline/operators/transpose/cutt/CudaUtils.h"
#include "dali/util/cuda_utils.h"
#include "dali/util/dynlink_cuda.h"

const int numthread = 64;

// -----------------------------------------------------------------------------------
//
// Copy using scalar loads and stores
//
template <typename T>
__global__ void scalarCopyKernel(const int n, const T* data_in, T* data_out) {

  for (int i = threadIdx.x + blockIdx.x*blockDim.x;i < n;i += blockDim.x*gridDim.x) {
    data_out[i] = data_in[i];
  }

}
template <typename T>
void scalarCopy(const int n, const T* data_in, T* data_out, cudaStream_t stream) {

  int numblock = (n - 1)/numthread + 1;

  scalarCopyKernel<T> <<< numblock, numthread, 0, stream >>>
  (n, data_in, data_out);

  CUDA_CALL(cudaGetLastError());
}
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
//
// Copy using vectorized loads and stores
//
template <typename T>
__global__ void vectorCopyKernel(const int n, T* data_in, T* data_out) {

  // Maximum vector load is 128 bits = 16 bytes
  const int vectorLength = 16/sizeof(T);

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  // Vector elements
  for (int i = idx;i < n/vectorLength;i += blockDim.x*gridDim.x) {
    reinterpret_cast<int4*>(data_out)[i] = reinterpret_cast<int4*>(data_in)[i];
  }

  // Remaining elements
  for (int i = idx + (n/vectorLength)*vectorLength;i < n;i += blockDim.x*gridDim.x + threadIdx.x) {
    data_out[i] = data_in[i];
  }

}

template <typename T>
void vectorCopy(const int n, T* data_in, T* data_out, cudaStream_t stream) {

  const int vectorLength = 16/sizeof(T);

  int numblock = (n/vectorLength - 1)/numthread + 1;
  // numblock = min(65535, numblock);
  int shmemsize = 0;

  vectorCopyKernel<T> <<< numblock, numthread, shmemsize, stream >>>
  (n, data_in, data_out);

  CUDA_CALL(cudaGetLastError());
}
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
//
// Copy using vectorized loads and stores
//
template <int numElem>
__global__ void memcpyFloatKernel(const int n, float4 *data_in, float4* data_out) {
  int index = threadIdx.x + numElem*blockIdx.x*blockDim.x;
  float4 a[numElem];
#pragma unroll
  for (int i=0;i < numElem;i++) {
    if (index + i*blockDim.x < n) a[i] = data_in[index + i*blockDim.x];
  }
#pragma unroll
  for (int i=0;i < numElem;i++) {
    if (index + i*blockDim.x < n) data_out[index + i*blockDim.x] = a[i];
  }
}

template <int numElem>
__global__ void memcpyFloatLoopKernel(const int n, float4 *data_in, float4* data_out) {
  for (int index=threadIdx.x + blockIdx.x*numElem*blockDim.x;index < n;index += numElem*gridDim.x*blockDim.x)
  {
    float4 a[numElem];
#pragma unroll
    for (int i=0;i < numElem;i++) {
      if (index + i*blockDim.x < n) a[i] = data_in[index + i*blockDim.x];
    }
#pragma unroll
    for (int i=0;i < numElem;i++) {
      if (index + i*blockDim.x < n) data_out[index + i*blockDim.x] = a[i];
    }
  }
}

#define NUM_ELEM 2
void memcpyFloat(const int n, float* data_in, float* data_out, cudaStream_t stream) {

  int numblock = (n/(4*NUM_ELEM) - 1)/numthread + 1;
  int shmemsize = 0;
  memcpyFloatKernel<NUM_ELEM> <<< numblock, numthread, shmemsize, stream >>>
  (n/4, (float4 *)data_in, (float4 *)data_out);

  // int numblock = 64;
  // int shmemsize = 0;
  // memcpyFloatLoopKernel<NUM_ELEM> <<< numblock, numthread, shmemsize, stream >>>
  // (n/4, (float4 *)data_in, (float4 *)data_out);

  CUDA_CALL(cudaGetLastError());
}
// -----------------------------------------------------------------------------------

// Explicit instances
template void scalarCopy<int>(const int n, const int* data_in, int* data_out, cudaStream_t stream);
template void scalarCopy<long long int>(const int n, const long long int* data_in, long long int* data_out, cudaStream_t stream);
template void vectorCopy<int>(const int n, int* data_in, int* data_out, cudaStream_t stream);
template void vectorCopy<long long int>(const int n, long long int* data_in, long long int* data_out, cudaStream_t stream);
void memcpyFloat(const int n, float* data_in, float* data_out, cudaStream_t stream);
