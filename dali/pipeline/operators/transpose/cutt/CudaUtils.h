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

#ifndef DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUDAUTILS_H
#define DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUDAUTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

#include "dali/util/cuda_utils.h"
#include "dali/util/dynlink_cuda.h"

// Need by warp-level primitives
#define FULL_MASK 0xffffffff

void set_device_array_async_T(void *data, int value, const size_t ndata, cudaStream_t stream, const size_t sizeofT);
void set_device_array_T(void *data, int value, const size_t ndata, const size_t sizeofT);

template <class T>
void set_device_array(T *data, int value, const size_t ndata, cudaStream_t stream=0) {
  set_device_array_async_T(data, value, ndata, stream, sizeof(T));
}

template <class T>
void set_device_array_sync(T *data, int value, const size_t ndata) {
  set_device_array_T(data, value, ndata, sizeof(T));
}

void allocate_device_T(void **pp, const size_t len, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_device(T **pp, const size_t len) {
  allocate_device_T((void **)pp, len, sizeof(T));
}

void deallocate_device_T(void **pp);
//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
template <class T>
void deallocate_device(T **pp) {
  deallocate_device_T((void **)pp);
}
//----------------------------------------------------------------------------------------

void copy_HtoD_async_T(const void *h_array, void *d_array, size_t array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_HtoD_T(const void *h_array, void *d_array, size_t array_len,
     const size_t sizeofT);
void copy_DtoH_async_T(const void *d_array, void *h_array, const size_t array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_DtoH_T(const void *d_array, void *h_array, const size_t array_len, const size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
template <class T>
void copy_HtoD(const T *h_array, T *d_array, size_t array_len, cudaStream_t stream=0) {
  copy_HtoD_async_T(h_array, d_array, array_len, stream, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device using synchronous calls
//
template <class T>
void copy_HtoD_sync(const T *h_array, T *d_array, size_t array_len) {
  copy_HtoD_T(h_array, d_array, array_len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
template <class T>
void copy_DtoH(const T *d_array, T *h_array, const size_t array_len, cudaStream_t stream=0) {
  copy_DtoH_async_T(d_array, h_array, array_len, stream, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host using synchronous calls
//
template <class T>
void copy_DtoH_sync(const T *d_array, T *h_array, const size_t array_len) {
  copy_DtoH_T(d_array, h_array, array_len, sizeof(T));
}

#ifdef ENABLE_NVTOOLS
void gpuRangeStart(const char *range_name);
void gpuRangeStop();
#endif

#endif // DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUDAUTILS_H