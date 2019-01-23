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

#include "dali/pipeline/operators/transpose/cutt/cuttGpuModelKernel.h"

#include "dali/util/dynlink_cuda.h"
#include "dali/util/cuda_utils.h"
#include "dali/pipeline/operators/transpose/cutt/CudaUtils.h"

#define RESTRICT //__restrict__

//
// Global memory access statistics
//
struct MemStat {
  int gld_tran;
  int gst_tran;
  int gld_req;
  int gst_req;
  int cl_full_l2;
  int cl_part_l2;
  int cl_full_l1;
  int cl_part_l1;
  // int l1_tran;
  __device__ __forceinline__ void clear() {
    gld_tran = 0;
    gst_tran = 0;
    gld_req = 0;
    gst_req = 0;
    cl_full_l2 = 0;
    cl_part_l2 = 0;
    cl_full_l1 = 0;
    cl_part_l1 = 0;
    // l1_tran = 0;
  }
};

//
// Returns scalar tensor position. Each lane has the same p
// NOTE: c and d on inactive warps must be 1 !!
//
__device__ __forceinline__
int tensorPos(
  const int p, const int rank, const int c, const int d, const int ct,
  const int numLane=warpSize
  ) {

  int r = ((p/c) % d)*ct;
#pragma unroll
  for (int i=numLane/2;i >= 1;i/=2) {
    r += __shfl_xor_sync(FULL_MASK, r, i);
  }
  return r;

}

//
// Counts number of global memory transactions for a warp that accesses
// memory at pos using warp lanes 0, ..., n - 1
//
__device__ __forceinline__
int countGlTransactions(const int pos, const int n, const int accWidth, const int warpLane) {
  int seg0 = pos/accWidth;
  int srcLane = (warpLane == 0 || warpLane >= n) ? (warpLane) : (warpLane - 1);
  int seg1 = __shfl_sync(FULL_MASK, seg0, srcLane);
  int count = __popc(__ballot_sync(FULL_MASK, seg0 != seg1)) + 1;
  count = (n == 0) ? 0 : count;
  return count;
}

//
// Counts number of global memory transactions for a warp that accesses
// memory at pos using warp lanes 0, ..., n - 1
//
__device__ __forceinline__
int countGlTransactions(const int* segbuf, const int n) {
  int count = 0;
  for (int i = threadIdx.x;i < n;i += blockDim.x) {
    int seg      = segbuf[i];
    int seg_prev = (i - 1 >= 0) ? segbuf[i - 1] : -1;
    count += (seg != seg_prev);
  }
  return count;
}

//
// Counts number of full and partial cache lines for a warp that accesses per warp
// memory at pos using warp lanes 0, ..., n - 1
//
__device__ __forceinline__
void countCacheLines(const int pos, const int n, const int cacheWidth, const int warpLane,
  int& cl_full, int& cl_part) {

  int seg = pos/cacheWidth;
  // Lane is at the beginning of a full cache line, if seg0 matches seg0 cacheWidth - 1 away
  int readLane = warpLane + (cacheWidth - 1);
  int val = (seg == __shfl_sync(FULL_MASK, seg, readLane));
  val = (readLane < n) ? val : 0;
  cl_full += val;

  unsigned int valbit = (((val << cacheWidth) - 1)*val) << warpLane;
  // Perform warpSize-way bitwise or
#pragma unroll
  for (int i=warpSize/2;i >= 1;i/=2) {
    valbit |= __shfl_xor_sync(FULL_MASK, valbit, i);
  }
  // Now: lanes with valbit set are part of a full cache line,
  //      lanes with valbit unset are part of a partial cache line
  int full = (valbit >> warpLane) & 1;

  seg = (warpLane < n) ? seg : -1;
  int segP1 = __shfl_down_sync(FULL_MASK, seg, 1);
  segP1 = (warpLane + 1 < warpSize) ? segP1 : -1;
  int val2 = ((!full) && seg != segP1);
  cl_part += val2;
}

//
// Counts number of full and partial cache lines for a warp that accesses
// memory at cachelines segbuf[0] ... segbuf[n - 1]
//
__device__ __forceinline__
void countCacheLines(int* segbuf, const int n, const int cacheWidth,
  int& cl_full, int& cl_part) {

  const int topbit = (1 << 31);
  const int lowbits = ~(1 << 31);

  for (int i = threadIdx.x;i < n;i += blockDim.x) {
    // seg[i] is at the beginning of a full cache line, if seg[i] matches seg[i + cacheWidth - 1]
    int i1 = i + (cacheWidth - 1);
    int val = 0;
    if (i1 < n) val = ((segbuf[i] & lowbits) == (segbuf[i1] & lowbits));
    cl_full += val;
    // Mark full cache lines with top bit set to 1
    if (val) {
      for (int j=0;j < cacheWidth;j++) {
        if (i + j < n) segbuf[i + j] |= topbit;
      }
    }
  }
  __syncthreads();

  for (int i = threadIdx.x;i < n;i += blockDim.x) {
    int seg = segbuf[i];
    int segP1 = (i + 1 < n) ? segbuf[i + 1] : -1;
    int part = ((seg & topbit) == 0);
    int val2 = (part && seg != segP1);
    cl_part += val2;
  }

  // Clear top bits
  __syncthreads();
  for (int i = threadIdx.x;i < n;i += blockDim.x) {
    segbuf[i] &= lowbits;
  }

}

//
// Runs countGlTransactions and countCacheLines counters for testing
// Unused values in posData[] are marked with "-1"
//
__global__ void runCountersKernel(const int* posData, const int numPosData,
  const int accWidth, const int cacheWidth, int* tranData, int* cl_fullData, int* cl_partData) {

  const int warpLane = threadIdx.x & (warpSize - 1);

  for (int i=threadIdx.x + blockIdx.x*blockDim.x;i < numPosData;i+=blockDim.x*gridDim.x) {
    int pos = posData[i];
    int flag = (pos == -1);
    int ffsval = __ffs(__ballot_sync(FULL_MASK, flag)) - 1;
    int n = (__any_sync(FULL_MASK, flag)) ? ffsval : warpSize;
    int tran = countGlTransactions(pos, n, accWidth, warpLane);
    int cl_full = 0;
    int cl_part = 0;
    countCacheLines(pos, n, cacheWidth, warpLane, cl_full, cl_part);
#pragma unroll
    for (int k=warpSize/2;k >= 1;k/=2) {
      cl_full += __shfl_xor_sync(FULL_MASK, cl_full, k);
      cl_part += __shfl_xor_sync(FULL_MASK, cl_part, k);
    }
    int j = i / warpSize;
    tranData[j] = tran;
    cl_fullData[j] = cl_full;
    cl_partData[j] = cl_part;
  }

}

//
// Reduce memStat within warp and write result to global memory
// NOTE: Not super-efficient since every warp does atomicAdd().
//
__device__ __forceinline__
void writeMemStat(const int warpLane, MemStat memStat, MemStat* RESTRICT glMemStat) {
  for (int i=16;i >= 1;i/=2) {
    // memStat.gld_tran += __shfl_xor_sync(FULL_MASK, memStat.gld_tran, i);
    // memStat.gst_tran += __shfl_xor_sync(FULL_MASK, memStat.gst_tran, i);
    // memStat.gld_req  += __shfl_xor_sync(FULL_MASK, memStat.gld_req, i);
    // memStat.gst_req  += __shfl_xor_sync(FULL_MASK, memStat.gst_req, i);
    memStat.cl_full_l2  += __shfl_xor_sync(FULL_MASK, memStat.cl_full_l2, i);
    memStat.cl_part_l2  += __shfl_xor_sync(FULL_MASK, memStat.cl_part_l2, i);
    memStat.cl_full_l1  += __shfl_xor_sync(FULL_MASK, memStat.cl_full_l1, i);
    memStat.cl_part_l1  += __shfl_xor_sync(FULL_MASK, memStat.cl_part_l1, i);
    // memStat.l1_tran     += __shfl_xor_sync(FULL_MASK, memStat.l1_tran, i);
  }
  if (warpLane == 0) {
    atomicAdd(&(glMemStat->gld_tran), memStat.gld_tran);
    atomicAdd(&(glMemStat->gst_tran), memStat.gst_tran);
    atomicAdd(&(glMemStat->gld_req), memStat.gld_req);
    atomicAdd(&(glMemStat->gst_req), memStat.gst_req);
    atomicAdd(&(glMemStat->cl_full_l2), memStat.cl_full_l2);
    atomicAdd(&(glMemStat->cl_part_l2), memStat.cl_part_l2);
    atomicAdd(&(glMemStat->cl_full_l1), memStat.cl_full_l1);
    atomicAdd(&(glMemStat->cl_part_l1), memStat.cl_part_l1);
    // atomicAdd(&(glMemStat->l1_tran), memStat.l1_tran);
  }
}

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMk-1)/TILEDIM+1), 1, plan.volMbar);
//
__global__ void
__launch_bounds__(TILEDIM*TILEROWS, 1)
countTiled(
  const int numMm, const int volMbar, const int sizeMbar,
  const int2 tiledVol, const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* RESTRICT glMbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat) {

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

  MemStat memStat;
  memStat.clear();

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

    // Read data into shared memory tile
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int n = __popc(__ballot_sync(FULL_MASK, maskIny & (1 << j)));
      memStat.gld_tran += countGlTransactions(posIn, n, accWidth, warpLane);
      memStat.gld_req += __any_sync(FULL_MASK, n > 0);
      posIn += posInAdd;
    }

#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int n = __popc(__ballot_sync(FULL_MASK, maskOutx & (1 << j)));
      memStat.gst_tran += countGlTransactions(posOut, n, accWidth, warpLane);
      memStat.gst_req += __any_sync(FULL_MASK, n > 0);
      countCacheLines(posOut, n, cacheWidth, warpLane, memStat.cl_full_l2, memStat.cl_part_l2);
      posOut += posOutAdd;
    }

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat);

}

//
// Packed transpose. Thread block loads plan.volMmk number of elements
//
template <int numRegStorage>
__global__ void
__launch_bounds__(1024, 1)
countPacked(
  const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* RESTRICT gl_Mmk,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat) {

  extern __shared__ int shSegOut[];

  const int warpLane = threadIdx.x & (warpSize - 1);

  TensorConvInOut Mmk;
  Mmk.c_in = 1;
  Mmk.d_in = 1;
  Mmk.c_out = 1;
  Mmk.d_out = 1;
  if (warpLane < sizeMmk) {
    Mmk = gl_Mmk[warpLane];
  }

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      posMmkIn[j]  += ((posMmk / __shfl_sync(FULL_MASK, Mmk.c_in,i)) % __shfl_sync(FULL_MASK, Mmk.d_in,i))*__shfl_sync(FULL_MASK, Mmk.ct_in,i);
      posMmkOut[j] += ((posMmk / __shfl_sync(FULL_MASK, Mmk.c_out,i)) % __shfl_sync(FULL_MASK, Mmk.d_out,i))*__shfl_sync(FULL_MASK, Mmk.ct_out,i);
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

  MemStat memStat;
  memStat.clear();

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

    // Read from global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posIn = posMbarIn + posMmkIn[j];
      int n = __popc(__ballot_sync(FULL_MASK, posMmk < volMmk));
      memStat.gld_tran += countGlTransactions(posIn, n, accWidth, warpLane);
      memStat.gld_req += __any_sync(FULL_MASK, n > 0);
    }

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posOut = posMbarOut + posMmkOut[j];
      int n = __popc(__ballot_sync(FULL_MASK, posMmk < volMmk));
      memStat.gst_tran += countGlTransactions(posOut, n, accWidth, warpLane);
      memStat.gst_req += __any_sync(FULL_MASK, n > 0);
      if (posMmk < volMmk) shSegOut[posMmk] = posOut/cacheWidth;
    }

    __syncthreads();
    countCacheLines(shSegOut, volMmk, cacheWidth, memStat.cl_full_l2, memStat.cl_part_l2);
    // Go from L2 segments to L1 segments
    __syncthreads();
    const int L2toL1 = accWidth/cacheWidth;
    for (int i=threadIdx.x;i < volMmk;i+=blockDim.x) {
      shSegOut[i] /= L2toL1;
    }
    __syncthreads();
    countCacheLines(shSegOut, volMmk, accWidth, memStat.cl_full_l1, memStat.cl_part_l1);

    // __syncthreads();
    // memStat.l1_tran += countGlTransactions(shSegOut, volMmk);

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat);
  
}

//
// Packed method with a split rank
//
// dim nthread(((volMmkWithSplit - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize, 1, 1)
// dim nblock(ts.numSplit, min(256, max(1, ts.volMbar)), 1)
//
template <int numRegStorage>
__global__ void
__launch_bounds__(1024, 1)
countPackedSplit(
  const int splitDim, const int volMmkUnsplit, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const int cMmSplit, const int cMkSplit,
  const TensorConvInOut* RESTRICT glMmk,
  const TensorConvInOut* RESTRICT glMbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat) {

  extern __shared__ int shSegOut[];

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
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j]  = posMmkIn0;
    posMmkOut[j] = posMmkOut0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int t = threadIdx.x + j*blockDim.x;
      posMmkIn[j]  += ((t/__shfl_sync(FULL_MASK, Mmk.c_in,i)) % __shfl_sync(FULL_MASK, Mmk.d_in,i))*__shfl_sync(FULL_MASK, Mmk.ct_in,i);
      posMmkOut[j] += ((t/__shfl_sync(FULL_MASK, Mmk.c_out,i)) % __shfl_sync(FULL_MASK, Mmk.d_out,i))*__shfl_sync(FULL_MASK, Mmk.ct_out,i);
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

  MemStat memStat;
  memStat.clear();

  for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar+=gridDim.y)
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
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posIn = posMbarIn + posMmkIn[j];
      int n = __popc(__ballot_sync(FULL_MASK, posMmk < volMmkSplit));
      memStat.gld_tran += countGlTransactions(posIn, n, accWidth, warpLane);
      memStat.gld_req += __any_sync(FULL_MASK, n > 0);
    }

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posOut = posMbarOut + posMmkOut[j];
      int n = __popc(__ballot_sync(FULL_MASK, posMmk < volMmkSplit));
      memStat.gst_tran += countGlTransactions(posOut, n, accWidth, warpLane);
      memStat.gst_req += __any_sync(FULL_MASK, n > 0);
      if (posMmk < volMmkSplit) shSegOut[posMmk] = posOut / cacheWidth;
      // countCacheLines(posOut, n, cacheWidth, warpLane, memStat.cl_full, memStat.cl_part);
    }

    __syncthreads();
    countCacheLines(shSegOut, volMmkSplit, cacheWidth, memStat.cl_full_l2, memStat.cl_part_l2);
    // Go from L2 segments to L1 segments
    __syncthreads();
    const int L2toL1 = accWidth/cacheWidth;
    for (int i=threadIdx.x;i < volMmkSplit;i+=blockDim.x) {
      shSegOut[i] /= L2toL1;
    }
    __syncthreads();
    countCacheLines(shSegOut, volMmkSplit, accWidth, memStat.cl_full_l1, memStat.cl_part_l1);

    // __syncthreads();
    // memStat.l1_tran += countGlTransactions(shSegOut, volMmkSplit);

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat);

}

//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMkBar-1)/TILEDIM+1), 1, plan.volMbar);
//
__global__ void
__launch_bounds__(TILEDIM*TILEROWS, 1)
countTiledCopy(
  const int numMm, const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const int2 tiledVol,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat) {

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

  MemStat memStat;
  memStat.clear();

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += gridDim.z)
  {

    // Read global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      pos0 += x + y*cuDimMk;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos  = pos0  + j*cuDimMk;
        int n = __popc(__ballot_sync(FULL_MASK, (x < tiledVol.x) && (y + j < tiledVol.y)));
        memStat.gld_tran += countGlTransactions(pos, n, accWidth, warpLane);
        memStat.gld_req += __any_sync(FULL_MASK, n > 0);
      }
    }

    // Write global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      pos0 += x + y*cuDimMm;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        int n = __popc(__ballot_sync(FULL_MASK, (x < tiledVol.x) && (y + j < tiledVol.y)));
        memStat.gst_tran += countGlTransactions(pos, n, accWidth, warpLane);
        memStat.gst_req += __any_sync(FULL_MASK, n > 0);
        countCacheLines(pos, n, cacheWidth, warpLane, memStat.cl_full_l2, memStat.cl_part_l2);
      }
    }

  }
  
  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat);

}

//######################################################################################
//######################################################################################
//######################################################################################

void runCounters(const int warpSize, const int* hostPosData, const int numPosData,
  const int accWidth, const int cacheWidth, int* host_tran, int* host_cl_full, int* host_cl_part) {
  
  const int numWarp = numPosData/warpSize;

  int* devPosData;
  allocate_device<int>(&devPosData, numPosData);
  copy_HtoD<int>(hostPosData, devPosData, numPosData);

  int* dev_tran;
  int* dev_cl_full;
  int* dev_cl_part;
  allocate_device<int>(&dev_tran, numWarp);
  allocate_device<int>(&dev_cl_full, numWarp);
  allocate_device<int>(&dev_cl_part, numWarp);

  int nthread = 512;
  int nblock = (numPosData - 1)/nthread + 1;
  runCountersKernel<<< nblock, nthread >>>(devPosData, numPosData,
    accWidth, cacheWidth, dev_tran, dev_cl_full, dev_cl_part);
  CUDA_CALL(cudaGetLastError());

  copy_DtoH<int>(dev_tran,    host_tran,    numWarp);
  copy_DtoH<int>(dev_cl_full, host_cl_full, numWarp);
  copy_DtoH<int>(dev_cl_part, host_cl_part, numWarp);
  CUDA_CALL(cudaDeviceSynchronize());

  deallocate_device<int>(&dev_tran);
  deallocate_device<int>(&dev_cl_full);
  deallocate_device<int>(&dev_cl_part);

  deallocate_device<int>(&devPosData);
}

bool cuttGpuModelKernel(cuttPlan_t& plan, const int accWidth, const int cacheWidth,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1) {

  LaunchConfig& lc = plan.launchConfig;
  TensorSplit& ts = plan.tensorSplit;

  MemStat* devMemStat;
  allocate_device<MemStat>(&devMemStat, 1);
  set_device_array<MemStat>(devMemStat, 0, 1, plan.stream);

  switch(ts.method) {
    case Trivial:
    {
      return false;
    }

    case Packed:
    {
      switch(lc.numRegStorage) {
#define CALL0(NREG) \
    countPacked<NREG> <<< lc.numblock, lc.numthread, ts.volMmk*sizeof(int), plan.stream >>> \
      (ts.volMmk, ts.volMbar, ts.sizeMmk, ts.sizeMbar, \
      plan.Mmk, plan.Mbar, accWidth, cacheWidth, devMemStat)
#define CALL(ICASE) case ICASE: CALL0(ICASE); break
#include "calls.h"
        default:
        printf("cuttGpuModelKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case PackedSplit:
    {

      // Calculate max. volume of split Mmk
      const int volSplit = (ts.splitDim/ts.numSplit) + ((ts.splitDim % ts.numSplit) != 0);
      const int volMmkSplit = volSplit*ts.volMmkUnsplit;

      switch(lc.numRegStorage) {
#define CALL0(NREG) \
    countPackedSplit<NREG> <<< lc.numblock, lc.numthread, volMmkSplit*sizeof(int), plan.stream >>> \
      (ts.splitDim, ts.volMmkUnsplit, ts. volMbar, ts.sizeMmk, ts.sizeMbar, \
        plan.cuDimMm, plan.cuDimMk, plan.Mmk, plan.Mbar, accWidth, cacheWidth, devMemStat)
#define CALL(ICASE) case ICASE: CALL0(ICASE); break
#include "calls.h"
        default:
        printf("cuttGpuModelKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case Tiled:
    {
      countTiled <<< lc.numblock, lc.numthread, 0, plan.stream >>>
      (((ts.volMm - 1)/TILEDIM + 1), ts.volMbar, ts.sizeMbar, plan.tiledVol, plan.cuDimMk, plan.cuDimMm,
        plan.Mbar, accWidth, cacheWidth, devMemStat);
    }
    break;

    case TiledCopy:
    {
      countTiledCopy <<< lc.numblock, lc.numthread, 0, plan.stream >>>
      (((ts.volMm - 1)/TILEDIM + 1), ts.volMbar, ts.sizeMbar, plan.cuDimMk, plan.cuDimMm, plan.tiledVol,
        plan.Mbar, accWidth, cacheWidth, devMemStat);
    }
    break;

  }

  CUDA_CALL(cudaGetLastError());

  MemStat hostMemStat;
  copy_DtoH<MemStat>(devMemStat, &hostMemStat, 1, plan.stream);
  CUDA_CALL(cudaDeviceSynchronize());
  deallocate_device<MemStat>(&devMemStat);

  gld_tran   = hostMemStat.gld_tran;
  gst_tran   = hostMemStat.gst_tran;
  gld_req    = hostMemStat.gld_req;
  gst_req    = hostMemStat.gst_req;
  cl_full_l2 = hostMemStat.cl_full_l2;
  cl_part_l2 = hostMemStat.cl_part_l2;
  cl_full_l1 = hostMemStat.cl_full_l1;
  cl_part_l1 = hostMemStat.cl_part_l1;
  // l1_tran    = hostMemStat.l1_tran;

  return true;
}
