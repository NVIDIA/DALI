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

#include "dali/pipeline/operators/transpose/cutt/cuttkernel.h"

#include <iostream>

#include "dali/util/dynlink_cuda.h"
#include "dali/util/cuda_utils.h"

#include "dali/pipeline/operators/transpose/cutt/CudaUtils.h"
#include "dali/pipeline/operators/transpose/cutt/LRUCache.h"

#define RESTRICT __restrict__

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMk-1)/TILEDIM+1), 1, plan.volMbar);
//
template <typename T>
__global__ void transposeTiled(
  const int numMm, const int volMbar, const int sizeMbar,
  const int2 tiledVol, const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* RESTRICT glMbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory
  __shared__ T shTile[TILEDIM][TILEDIM+1];

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int bx = (blockIdx.x % numMm)*TILEDIM;
  const int by = (blockIdx.x / numMm)*TILEDIM;

  const int xin = bx + threadIdx.x;
  const int yin = by + threadIdx.y;

  const int xout = bx + threadIdx.y;
  const int yout = by + threadIdx.x;

  const unsigned int maskIny = __ballot_sync(FULL_MASK, (yin + warpLane < tiledVol.y))*(xin < tiledVol.x);
  const unsigned int maskOutx = __ballot_sync(FULL_MASK, (xout + warpLane < tiledVol.x))*(yout < tiledVol.y);

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += gridDim.z)
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMajorIn += __shfl_xor_sync(FULL_MASK, posMajorIn, i);
      posMajorOut += __shfl_xor_sync(FULL_MASK, posMajorOut, i);
    }
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Read from global memory
    __syncthreads();

    // Read data into shared memory tile
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // int pos = posIn + j*cuDimMk;
      // if (xin < readVol.x && yin + j < readVol.y) {
      if ((maskIny & (1 << j)) != 0) {
        shTile[threadIdx.y + j][threadIdx.x] = dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write to global memory
    __syncthreads();

#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // int pos = posOut + j*cuDimMm;
      // if (xout + j < readVol.x && yout < readVol.y) {
      if ((maskOutx & (1 << j)) != 0 ) {
        dataOut[posOut] = shTile[threadIdx.x][threadIdx.y + j];
      }
      posOut += posOutAdd;
    }

  }
  
}

//
// Packed transpose. Thread block loads plan.volMmk number of elements
//
template <typename T, int numRegStorage>
__global__ void transposePacked(
  const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* RESTRICT gl_Mmk,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const TensorConv* RESTRICT gl_Msh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory. volMmk elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx.x & (warpSize - 1);

  TensorConvInOut Mmk;
  Mmk.c_in = 1;
  Mmk.d_in = 1;
  Mmk.c_out = 1;
  Mmk.d_out = 1;
  if (warpLane < sizeMmk) {
    Mmk = gl_Mmk[warpLane];
  }
  TensorConv Msh;
  Msh.c = 1;
  Msh.d = 1;
  if (warpLane < sizeMmk) {
    Msh = gl_Msh[warpLane];
  }

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
  int posSh[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      posMmkIn[j]  += ((posMmk / __shfl_sync(FULL_MASK, Mmk.c_in,i)) % __shfl_sync(FULL_MASK, Mmk.d_in,i))*__shfl_sync(FULL_MASK, Mmk.ct_in,i);
      posMmkOut[j] += ((posMmk / __shfl_sync(FULL_MASK, Mmk.c_out,i)) % __shfl_sync(FULL_MASK, Mmk.d_out,i))*__shfl_sync(FULL_MASK, Mmk.ct_out,i);
      posSh[j]     += ((posMmk / __shfl_sync(FULL_MASK, Msh.c,i)) % __shfl_sync(FULL_MASK, Msh.d,i))*__shfl_sync(FULL_MASK, Msh.ct,i);
    }
  }

  // 6 registers
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  for (int posMbar=blockIdx.x;posMbar < volMbar;posMbar += gridDim.x)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += __shfl_xor_sync(FULL_MASK, posMbarOut, i);
    }

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += __shfl_xor_sync(FULL_MASK, posMbarIn, i);
    }

    __syncthreads();

    // Read from global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posIn];
    }

    __syncthreads();

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posOut = posMbarOut + posMmkOut[j];
      if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j]];
    }


  }
  
}

//
// Packed method with a split rank
//
// dim nthread(((volMmkWithSplit - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize, 1, 1)
// dim nblock(ts.numSplit, min(256, max(1, ts.volMbar)), 1)
//
template <typename T, int numRegStorage>
__global__ void transposePackedSplit(
  const int splitDim, const int volMmkUnsplit, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const int cMmSplit, const int cMkSplit,
  const TensorConvInOut* RESTRICT glMmk,
  const TensorConvInOut* RESTRICT glMbar,
  const TensorConv* RESTRICT glMsh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory. max(volSplit)*volMmkUnsplit T elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx.x & (warpSize - 1);

  // const int plusone = (blockIdx.x < (splitDim % gridDim.x));
  const int p0 = blockIdx.x*splitDim/gridDim.x;
  const int volSplit = (blockIdx.x + 1)*splitDim/gridDim.x - p0;
  const int plusone = volSplit - splitDim/gridDim.x;

  TensorConvInOut Mmk;
  Mmk.c_in = 1;
  Mmk.d_in = 1;
  Mmk.c_out = 1;
  Mmk.d_out = 1;
  if (warpLane < sizeMmk) {
    Mmk = glMmk[warpLane + plusone*sizeMmk];
  }
  TensorConv Msh;
  Msh.c = 1;
  Msh.d = 1;
  if (warpLane < sizeMmk) {
    Msh = glMsh[warpLane + plusone*sizeMmk];
  }

  // gridDim.x = number of splits
  // blockIdx.x = {0 ... gridDim.x - 1} is the split-index
  // Volume of this split
  // const int volSplit = (splitDim/gridDim.x) + plusone;
  // Start position in this split
  // const int p0 = (splitDim/gridDim.x)*blockIdx.x + min(blockIdx.x, (splitDim % gridDim.x));
  const int posMmkIn0  = p0*cMmSplit;
  const int posMmkOut0 = p0*cMkSplit;
  // Volume of split Mmk
  const int volMmkSplit = volSplit*volMmkUnsplit;

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
  int posSh[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j]  = posMmkIn0;
    posMmkOut[j] = posMmkOut0;
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int t = threadIdx.x + j*blockDim.x;
      posMmkIn[j]  += ((t/__shfl_sync(FULL_MASK, Mmk.c_in,i)) % __shfl_sync(FULL_MASK, Mmk.d_in,i))*__shfl_sync(FULL_MASK, Mmk.ct_in,i);
      posMmkOut[j] += ((t/__shfl_sync(FULL_MASK, Mmk.c_out,i)) % __shfl_sync(FULL_MASK, Mmk.d_out,i))*__shfl_sync(FULL_MASK, Mmk.ct_out,i);
      posSh[j]     += ((t/__shfl_sync(FULL_MASK, Msh.c,i)) % __shfl_sync(FULL_MASK, Msh.d,i))*__shfl_sync(FULL_MASK, Msh.ct,i);
    }
  }

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int posMbar0 = blockIdx.y*volMbar/gridDim.y;
  const int posMbar1 = (blockIdx.y + 1)*volMbar/gridDim.y;
  for (int posMbar=posMbar0;posMbar < posMbar1;posMbar++)
  // for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar+=gridDim.y)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += __shfl_xor_sync(FULL_MASK, posMbarOut, i);
    }

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += __shfl_xor_sync(FULL_MASK, posMbarIn, i);
    }

    // Read from global memory
    __syncthreads();

#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmkSplit) shBuffer[posMmk] = dataIn[posIn];
    }

    // Write to global memory
    __syncthreads();

#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posOut = posMbarOut + posMmkOut[j];
      if (posMmk < volMmkSplit) dataOut[posOut] = shBuffer[posSh[j]];
    }

  }

}

//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMkBar-1)/TILEDIM+1), 1, plan.volMbar);
//
template <typename T>
__global__ void transposeTiledCopy(
  const int numMm, const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const int2 tiledVol,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  const int bx = (blockIdx.x % numMm)*TILEDIM;
  const int by = (blockIdx.x / numMm)*TILEDIM;

  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;

  const unsigned int mask = __ballot_sync(FULL_MASK, (y + warpLane < tiledVol.y))*(x < tiledVol.x);

  const int posMinorIn = x + y*cuDimMk;
  const int posMinorOut = x + y*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += gridDim.z)
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMajorIn += __shfl_xor_sync(FULL_MASK, posMajorIn, i);
      posMajorOut += __shfl_xor_sync(FULL_MASK, posMajorOut, i);
    }
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Variables where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
      if ((mask & (1 << j)) != 0) {
        val[j/TILEROWS] = dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write global memory
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
      if ((mask & (1 << j)) != 0) {
        dataOut[posOut] = val[j/TILEROWS];
      }
      posOut += posOutAdd;
    }

  }
  
}
//######################################################################################
//######################################################################################
//######################################################################################

//
// Sets shared memory bank configuration for all kernels. Needs to be called once per device.
//
void cuttKernelSetSharedMemConfig() {  
#define CALL(NREG) CUDA_CALL(cudaFuncSetSharedMemConfig(transposePacked<float, NREG>, cudaSharedMemBankSizeFourByte ))
#include "calls.h"
#undef CALL

#define CALL(NREG) CUDA_CALL(cudaFuncSetSharedMemConfig(transposePacked<double, NREG>, cudaSharedMemBankSizeEightByte ))
#include "calls.h"
#undef CALL

#define CALL(NREG) CUDA_CALL(cudaFuncSetSharedMemConfig(transposePackedSplit<float, NREG>, cudaSharedMemBankSizeFourByte ))
#include "calls.h"
#undef CALL

#define CALL(NREG) CUDA_CALL(cudaFuncSetSharedMemConfig(transposePackedSplit<double, NREG>, cudaSharedMemBankSizeEightByte ))
#include "calls.h"
#undef CALL

  CUDA_CALL(cudaFuncSetSharedMemConfig(transposeTiled<float>, cudaSharedMemBankSizeFourByte));
  CUDA_CALL(cudaFuncSetSharedMemConfig(transposeTiledCopy<float>, cudaSharedMemBankSizeFourByte));

  CUDA_CALL(cudaFuncSetSharedMemConfig(transposeTiled<double>, cudaSharedMemBankSizeEightByte));
  CUDA_CALL(cudaFuncSetSharedMemConfig(transposeTiledCopy<double>, cudaSharedMemBankSizeEightByte));

}

// Caches for PackedSplit kernels. One cache for all devices
// NOTE: Not thread safe
const int CACHE_SIZE = 100000;
const int MAX_NUMWARP = (1024/32);
const int MAX_NUMTYPE = 2;
static int numDevices = -1;
LRUCache<unsigned long long int, int> nabCache(CACHE_SIZE, -1);

//
// Returns the maximum number of active blocks per SM
//
int getNumActiveBlock(const int method, const int sizeofType, const LaunchConfig& lc,
  const int deviceID, const cudaDeviceProp& prop) {

  int numActiveBlock = -1; // default init to silent warnings
  int numthread = lc.numthread.x * lc.numthread.y * lc.numthread.z;
  switch(method) {
    case Trivial:
    {
      // This value does not matter, but should be > 0
      numActiveBlock = 1;
    }
    break;

    case Packed:
    {
#define CALL0(TYPE, NREG) \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, \
    transposePacked<TYPE, NREG>, numthread, lc.shmemsize)
      switch(lc.numRegStorage) {
#define CALL(ICASE) case ICASE: if (sizeofType == 1) CALL0(char, ICASE);   \
                                if (sizeofType == 2) CALL0(short, ICASE);  \
                                if (sizeofType == 4) CALL0(float,  ICASE); \
                                if (sizeofType == 8) CALL0(double, ICASE); \
                                break
#include "calls.h"
      }
#undef CALL
#undef CALL0
    }
    break;

    case PackedSplit:
    {
      // Allocate cache structure if needed
      if (numDevices == -1) {
        CUDA_CALL(cudaGetDeviceCount(&numDevices));
      }
      // Build unique key for cache
      int key_warp = (numthread/prop.warpSize - 1);
      if (key_warp >= MAX_NUMWARP) {
        printf("getNumActiveBlock maximum number of warps exceeded\n");
        exit(1);
      }
      int key_reg = (lc.numRegStorage - 1);
      int key_type = (sizeofType == 4);
      unsigned long long int key = 
      (unsigned long long int)(lc.shmemsize/sizeofType)*MAX_NUMWARP*MAX_REG_STORAGE*MAX_NUMTYPE*numDevices + 
      (unsigned long long int)deviceID*MAX_NUMWARP*MAX_REG_STORAGE*MAX_NUMTYPE +
      (unsigned long long int)key_type*MAX_NUMWARP*MAX_REG_STORAGE + 
      (unsigned long long int)key_reg*MAX_NUMWARP + 
      (unsigned long long int)key_warp;

      numActiveBlock = nabCache.get(key);
      if (numActiveBlock == -1) {
        // key not found in cache, determine value and add it to cache
#define CALL0(TYPE, NREG) \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, \
    transposePackedSplit<TYPE, NREG>, numthread, lc.shmemsize)
      switch(lc.numRegStorage) {
#define CALL(ICASE) case ICASE: if (sizeofType == 1) CALL0(char, ICASE);   \
                                if (sizeofType == 2) CALL0(short, ICASE);  \
                                if (sizeofType == 4) CALL0(float,  ICASE); \
                                if (sizeofType == 8) CALL0(double, ICASE); \
                                break
#include "calls.h"
      }
#undef CALL
#undef CALL0
        nabCache.set(key, numActiveBlock);
      }
    }
    break;

    case Tiled:
    {
      if (sizeofType == 1) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<char>, numthread, lc.shmemsize);
      } else if (sizeofType == 2) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<short>, numthread, lc.shmemsize);
      } else if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case TiledCopy:
    {
      if (sizeofType == 1) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<char>, numthread, lc.shmemsize);
      } else if (sizeofType == 2) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<short>, numthread, lc.shmemsize);
      } else if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<double>, numthread, lc.shmemsize);
      }
    }
    break;
  }

  return numActiveBlock;
}

//
// Sets up kernel launch configuration
//
// Returns the number of active blocks per SM that can be achieved on the Packed kernel
// NOTE: Returns 0 when kernel execution is not possible
//
// Sets:
// lc.numthread
// lc.numblock
// lc.shmemsize
// lc.numRegStorage  (for Packed method)
//
int cuttKernelLaunchConfiguration(const int sizeofType, const TensorSplit& ts,
  const int deviceID, const cudaDeviceProp& prop, LaunchConfig& lc) {

  // Return value of numActiveBlock
  int numActiveBlockReturn = -1;

  switch(ts.method) {
    case Trivial:
    {
      // These values don't matter
      lc.numthread.x = 1;
      lc.numthread.y = 1;
      lc.numthread.z = 1;
      lc.numblock.x = 1;
      lc.numblock.y = 1;
      lc.numblock.z = 1;
      lc.numblock.z = 1;
      lc.numblock.z = 1;
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case Packed:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType); //ts.volMmk*sizeofType;

      // Check that we're not using too much shared memory per block
      if (lc.shmemsize > prop.sharedMemPerBlock) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      // Min and max number of threads we can use
      int minNumthread = ((ts.volMmk - 1)/(prop.warpSize*MAX_REG_STORAGE) + 1)*prop.warpSize;
      int maxNumthread = ((ts.volMmk - 1)/(prop.warpSize) + 1)*prop.warpSize;      
      if (minNumthread > prop.maxThreadsPerBlock) return 0;
      maxNumthread = min(prop.maxThreadsPerBlock, maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (ts.volMmk - 1)/maxNumthread + 1;
      int maxNumRegStorage = (ts.volMmk - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;
      int bestNumActiveBlock = 0;

      lc.numthread.y = 1;
      lc.numthread.z = 1;
      lc.numblock.x = max(1, ts.volMbar);
      lc.numblock.x = min(prop.multiProcessorCount*18, lc.numblock.x);
      lc.numblock.y = 1;
      lc.numblock.z = 1;

      for (lc.numRegStorage=minNumRegStorage;lc.numRegStorage <= maxNumRegStorage;lc.numRegStorage++) {
        lc.numthread.x = ((ts.volMmk - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize;

        int numActiveBlock = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
        // int val = numActiveBlock*lc.numthread.x;
        int val = ts.volMmkUsed()*numActiveBlock;
        if (val > bestVal) {
          bestVal = val;
          bestNumRegStorage = lc.numRegStorage;
          bestNumActiveBlock = numActiveBlock;
        }
      }

      if (bestNumRegStorage == 0) return 0;

      lc.numRegStorage = bestNumRegStorage;
      lc.numthread.x = ((ts.volMmk - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize;
      numActiveBlockReturn = bestNumActiveBlock;
    }
    break;

    case PackedSplit:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType);

      // Check that we're not using too much shared memory per block
      if (lc.shmemsize > prop.sharedMemPerBlock) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      int volMmkWithSplit = (ts.splitDim/ts.numSplit + ((ts.splitDim % ts.numSplit) > 0))*ts.volMmkUnsplit;

      // Min and max number of threads we can use
      int minNumthread = ((volMmkWithSplit - 1)/(prop.warpSize*MAX_REG_STORAGE) + 1)*prop.warpSize;
      int maxNumthread = ((volMmkWithSplit - 1)/(prop.warpSize) + 1)*prop.warpSize;      
      if (minNumthread > prop.maxThreadsPerBlock) return 0;
      maxNumthread = min(prop.maxThreadsPerBlock, maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (volMmkWithSplit - 1)/maxNumthread + 1;
      int maxNumRegStorage = (volMmkWithSplit - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;
      int bestNumActiveBlock = 0;

      lc.numthread.y = 1;
      lc.numthread.z = 1;
      lc.numblock.x = ts.numSplit;
      lc.numblock.y = max(1, min((prop.multiProcessorCount*18)/lc.numblock.x, ts.volMbar));
      lc.numblock.z = 1;

      for (lc.numRegStorage=minNumRegStorage;lc.numRegStorage <= maxNumRegStorage;lc.numRegStorage++) {
        lc.numthread.x = ((volMmkWithSplit - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize;

        int numActiveBlock = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
        // int val = numActiveBlock*lc.numthread.x*lc.numRegStorage;
        int val = ts.volMmkUsed()*numActiveBlock;
        if (val > bestVal) {
          bestVal = val;
          bestNumRegStorage = lc.numRegStorage;
          bestNumActiveBlock = numActiveBlock;
        }
      }

      if (bestNumRegStorage == 0) return 0;

      lc.numRegStorage = bestNumRegStorage;
      lc.numthread.x = ((volMmkWithSplit - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize;
      numActiveBlockReturn = bestNumActiveBlock;
    }
    break;

    case Tiled:
    {
      lc.numthread.x = TILEDIM;
      lc.numthread.y = TILEROWS;
      lc.numthread.z = 1;
      lc.numblock.x = ((ts.volMm - 1)/TILEDIM + 1)*((ts.volMk - 1)/TILEDIM + 1);
      lc.numblock.y = 1;
      lc.numblock.z = max(1, min((prop.multiProcessorCount*8)/(lc.numblock.x*lc.numblock.y), ts.volMbar));
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case TiledCopy:
    {
      lc.numthread.x = TILEDIM;
      lc.numthread.y = TILEROWS;
      lc.numthread.z = 1;
      lc.numblock.x = ((ts.volMm - 1)/TILEDIM + 1)*((ts.volMkBar - 1)/TILEDIM + 1);
      lc.numblock.y = 1;
      lc.numblock.z = ts.volMbar;
      lc.numblock.z = min((prop.multiProcessorCount*8)/(lc.numblock.x*lc.numblock.y), lc.numblock.z);
      lc.numblock.z = max(1, lc.numblock.z);
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;
  }

  if (lc.numblock.x > static_cast<unsigned int>(prop.maxGridSize[0]) ||
    lc.numblock.y > static_cast<unsigned int>(prop.maxGridSize[1]) ||
    lc.numblock.z > static_cast<unsigned int>(prop.maxGridSize[2])) return 0;

  // Return the number of active blocks with these settings
  if (numActiveBlockReturn == -1) {
    // Not set, get it
    numActiveBlockReturn = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
  }
  return numActiveBlockReturn;
}

bool cuttKernel(cuttPlan_t& plan, const void* dataIn, void* dataOut) {

  LaunchConfig& lc = plan.launchConfig;
  TensorSplit& ts = plan.tensorSplit;

  switch(ts.method) {
    case Trivial:
    {
      CUDA_CALL(cudaMemcpyAsync(dataOut, dataIn, ts.volMmk*ts.volMbar*plan.sizeofType,
        cudaMemcpyDeviceToDevice, plan.stream));
    }
    break;

    case Packed:
    {
      switch(lc.numRegStorage) {
#define CALL0(TYPE, NREG) \
    transposePacked<TYPE, NREG> <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.volMmk, ts.volMbar, ts.sizeMmk, ts.sizeMbar, \
      plan.Mmk, plan.Mbar, plan.Msh, (const TYPE *)dataIn, (TYPE *)dataOut)
#define CALL(ICASE) case ICASE: if (plan.sizeofType == 1) CALL0(char, ICASE);   \
                                if (plan.sizeofType == 2) CALL0(short, ICASE);  \
                                if (plan.sizeofType == 4) CALL0(float,  ICASE); \
                                if (plan.sizeofType == 8) CALL0(double, ICASE); \
                                break
#include "calls.h"
        default:
        printf("cuttKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case PackedSplit:
    {
      switch(lc.numRegStorage) {
#define CALL0(TYPE, NREG) \
    transposePackedSplit<TYPE, NREG> <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.splitDim, ts.volMmkUnsplit, ts. volMbar, ts.sizeMmk, ts.sizeMbar, \
        plan.cuDimMm, plan.cuDimMk, plan.Mmk, plan.Mbar, plan.Msh, (const TYPE *)dataIn, (TYPE *)dataOut)
#define CALL(ICASE) case ICASE: if (plan.sizeofType == 1) CALL0(char, ICASE);   \
                                if (plan.sizeofType == 2) CALL0(short, ICASE);  \
                                if (plan.sizeofType == 4) CALL0(float,  ICASE); \
                                if (plan.sizeofType == 8) CALL0(double, ICASE); \
                                break
#include "calls.h"
        default:
        printf("cuttKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case Tiled:
    {
#define CALL(TYPE) \
      transposeTiled<TYPE> <<< lc.numblock, lc.numthread, 0, plan.stream >>> \
      (((ts.volMm - 1)/TILEDIM + 1), ts.volMbar, ts.sizeMbar, plan.tiledVol, plan.cuDimMk, plan.cuDimMm, \
        plan.Mbar, (const TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 1) CALL(char);
      if (plan.sizeofType == 2) CALL(short);
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

    case TiledCopy:
    {
#define CALL(TYPE) \
      transposeTiledCopy<TYPE> <<< lc.numblock, lc.numthread, 0, plan.stream >>> \
      (((ts.volMm - 1)/TILEDIM + 1), ts.volMbar, ts.sizeMbar, plan.cuDimMk, plan.cuDimMm, plan.tiledVol, \
        plan.Mbar, (const TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 1) CALL(char);
      if (plan.sizeofType == 2) CALL(short);
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

  }

  CUDA_CALL(cudaGetLastError());
  return true;
}
