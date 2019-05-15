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


#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cstring>               // memcpy

#include "dali/pipeline/operators/transpose/cutt/cuttGpuModel.h"

#include "dali/pipeline/operators/transpose/cutt/cuttGpuModelKernel.h"
#ifdef ENABLE_NVTOOLS
#include "dali/pipeline/operators/transpose/cutt/CudaUtils.h"
#endif

// #define CALC_L1_CACHELINES

//
// Returns integer log2(a) rounded down
//
int ilog2(int a) {
  int k = 0;
  while (a >>= 1) k++;
  return k;
}

//
// Count number of global memory transactions per one request for potentially
// scattered accesses to elements listed in seg
// NOTE: Assumes seg is sorted
//
inline int glTransactions(const int* seg, const int n) {
  int count = (n > 0);
  for (int i=1;i < n;i++) {
    count += (seg[i-1] != seg[i]);
  }
  return count;
}

//
// Slower reference version of glTransactions
//
int glTransactionsRef(const int* pos, const int n, const int accWidth) {
  int count = 0;
  int iseg_prev = -1;
  for (int i=0;i < n;i++) {
    int iseg = pos[i]/accWidth;
    count += (iseg != iseg_prev);
    iseg_prev = iseg;
  }
  return count;
}

//
// Count number of global memory transactions per request for contigious memory access
// of n elements starting at pos
//
int glTransactions(const int pos, const int n, const int accWidth) {
  if (n == 0) return 0;
  // Segment at the first memory location
  int seg0 = pos/accWidth;
  // Segment at the last memory location
  int seg1 = (pos + n - 1)/accWidth;
  return (seg1 - seg0 + 1);
}

//
// Count number of full and partial cache-lines accessed for potentially
// scattered, but monotonically increasing accesses listed in pos
//
// cl_full = Number of full cache lines
// cl_part = Number of partial cache lines
//
void countCacheLines(const int* seg, const int n, const int cacheWidth, int& cl_full, int& cl_part) {

  cl_full = 0;
  cl_part = 0;

  int i = 0;
  while (i < n) {
    if (i + cacheWidth <= n && seg[i] == seg[i + cacheWidth - 1]) {
      cl_full++;
      i += cacheWidth;
    } else {
      // Count the first partial cache line and advance to next position
      cl_part++;
      i++;
      // Loop until the next full cache line
      while (i < n && seg[i] != ((i + cacheWidth <= n) ? seg[i + cacheWidth - 1] : -1)) {
        cl_part += (seg[i] != seg[i-1]);
        i++;
      }
    }
  }
}

void countCacheLines0(int_vector* segbuf, const int n, const int cacheWidth, int_vector& cl_full, int_vector& cl_part) {
  int_vector topbit(1 << 31);
  int_vector lowbits( ~(1 << 31) );

  cl_full = int_vector(0);
  cl_part = int_vector(0);

  for (int i=0;i < n;i++) {
    // seg[i] is at the beginning of a full cache line, if seg[i] matches seg[i + cacheWidth - 1]
    int i1 = i + (cacheWidth - 1);
    int_vector val(0);
    if (i1 < n) val = ((segbuf[i] & lowbits) == (segbuf[i1] & lowbits));
    cl_full += val;
    // Mark full cache lines with top bit set to 1
    if (val) {
      int_vector topbit_mask = bool_to_mask(val) & topbit;
      int m = std::min(i + cacheWidth, n);
      for (int j=i;j < m;j++) {
        segbuf[j] |= topbit_mask;
      }
    }
  }

  for (int i=0;i < n;i++) {
    int_vector seg = segbuf[i];
    int_vector segP1 = (i + 1 < n) ? segbuf[i + 1] : int_vector(-1);
    int_vector part = ((seg & topbit) == int_vector(0));
    int_vector val2 = (part & (seg != segP1));
    cl_part += val2;
  }

}

//
// Slower reference version of countCacheLines
//
void countCacheLinesRef(const int* pos, const int n, const int cacheWidth, int& cl_full, int& cl_part) {

  cl_full = 0;
  cl_part = 0;
  
  if (n == 0) return;

  int i = 0;
  while (i < n) {
    int i0 = i;
    int m = std::min(i + cacheWidth, n);
    int seg0 = pos[i++]/cacheWidth;
    while (i < m) {
      int seg = pos[i]/cacheWidth;
      if (seg0 != seg) {
        break;
      }
      i++;
    }
    if (i == i0 + cacheWidth) {
      cl_full++;
    } else {
      cl_part++;
    }
  }
}

//
// Count number of full and partial cache-lines accessed for contigious memory access
// of n elements starting at pos
//
// cl_full = Number of full cache lines
// cl_part = Number of partial cache lines
//
void countCacheLines(const int pos, const int n, const int cacheWidth, int& cl_full, int& cl_part) {
  if (n == 0) {
    cl_full = 0;
    cl_part = 0;
    return;
  }
  if (n < cacheWidth) {
    cl_full = 0;
    cl_part = 1 + ((pos % cacheWidth) + n > cacheWidth);
  } else {
    int start_part = (pos % cacheWidth);
    int end_part = ((pos + n) % cacheWidth);
    //partial:   start full?          end full?
    cl_part = (start_part != 0) + (end_part != 0);
    //full:         number of start partials   number of end partials
    cl_full = (n - (cacheWidth - start_part)*(start_part != 0) - end_part)/cacheWidth;
  }
}

//
// Compute memory element positions
// Non-vectorized version
//
void computePos(const int vol0, const int vol1,
  const TensorConvInOut* conv, const int numConv,
  int* posIn, int* posOut) {

  int nvol = vol1 - vol0;
  for (int i=0;i <= nvol;i++) {
    int posInVal = 0;
    int posOutVal = 0;
    int j = i + vol0;
    for (int k=0;k < numConv;k++) {
      posInVal  += ((j / conv[k].c_in) % conv[k].d_in) * conv[k].ct_in;
      posOutVal += ((j / conv[k].c_out) % conv[k].d_out) * conv[k].ct_out;
    }
    posIn[i] = posInVal;
    posOut[i] = posOutVal;
  }
}

//
// Compute memory element positions
// Starts from zero
//
void computePos0(const int vol,
  const int* __restrict__ dIn, const int* __restrict__ cIn, 
  const int* __restrict__ dOut, const int* __restrict__ cOut,
  int* __restrict__ posIn, int* __restrict__ posOut) {

  // Element position vector
  std::vector<int> pIn(32, 0);
  std::vector<int> pOut(32, 0);
  // Scalar element position
  int posInVal = 0;
  int posOutVal = 0;
  for (int i=0;i < vol;i++) {
    posIn[i] = posInVal;
    posOut[i] = posOutVal;
    // Advance position

    int iIn = 0;
    while (++pIn[iIn] == dIn[iIn]) {
      pIn[iIn] = 0;
      iIn++;
    }
    posInVal += cIn[iIn];
    //
    int iOut = 0;
    while (++pOut[iOut] == dOut[iOut]) {
      pOut[iOut] = 0;
      iOut++;
    }
    posOutVal += cOut[iOut];
  }
}

void computePos0(const int vol,
  const TensorConvInOut* conv, const int numConv,
  int* posIn, int* posOut) {

  int dIn[32];
  int cIn[32];
  int dOut[32];
  int cOut[32];
  //
  int c_in_prev = conv[0].ct_in;
  int cIn_prev = conv[0].ct_in;
  int d_in_prev = 1;
  //
  int c_out_prev = conv[0].ct_out;
  int cOut_prev = conv[0].ct_out;
  int d_out_prev = 1;
  for (int i=0;i < numConv;i++) {
    dIn[i] = conv[i].d_in;
    cIn[i] = cIn_prev + conv[i].ct_in - d_in_prev*c_in_prev;
    cIn_prev = cIn[i];
    c_in_prev = conv[i].ct_in;
    d_in_prev = conv[i].d_in;
    //
    dOut[i] = conv[i].d_out;
    cOut[i] = cOut_prev + conv[i].ct_out - d_out_prev*c_out_prev;
    cOut_prev = cOut[i];
    c_out_prev = conv[i].ct_out;
    d_out_prev = conv[i].d_out;
  }

  computePos0(vol, dIn, cIn, dOut, cOut, posIn, posOut);
}

//
// Compute memory element positions
// *** Slow reference version
//
void computePosRef(int vol0, int vol1,
  std::vector<TensorConvInOut>::iterator it0, std::vector<TensorConvInOut>::iterator it1,
  std::vector<int>& posIn, std::vector<int>& posOut) {
  int i=0;
  for (int j=vol0;j <= vol1;j++,i++) {
    int posInVal = 0;
    int posOutVal = 0;
    for (auto it=it0;it != it1;it++) {
      posInVal  += ((j / it->c_in) % it->d_in) * it->ct_in;
      posOutVal += ((j / it->c_out) % it->d_out) * it->ct_out;
    }
    posIn[i] = posInVal;
    posOut[i] = posOutVal;
  }
}

//
// Count number of global memory transactions for Packed -method
//
void countPackedGlTransactions(const int warpSize, const int accWidth, const int cacheWidth,
  const int numthread, const int posMbarIn, const int posMbarOut, const int volMmk, 
  std::vector<int>& posMmkIn, std::vector<int>& posMmkOut,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1) {

  std::vector<int> readSeg(warpSize);
  std::vector<int> writeSeg(warpSize);
  std::vector<int> writeSegVolMmk(volMmk);

  const int accWidthShift = ilog2(accWidth);
  const int cacheWidthShift = ilog2(cacheWidth);

  int m = 0;
  for (int j00=0;j00 < volMmk;j00+=numthread) {
    int n0 = std::min(volMmk, j00 + numthread);
    for (int j0=j00;j0 < n0;j0+=warpSize) {
      int n = std::min(warpSize, volMmk - j0);

      for (int j1=0;j1 < n;j1++) {
        int j = j0 + j1;
        int posIn  = posMbarIn + posMmkIn[j];
        int posOut = posMbarOut + posMmkOut[j];
        readSeg[j1] = posIn >> accWidthShift;
        writeSeg[j1] = posOut >> accWidthShift;
        writeSegVolMmk[m] = posOut >> cacheWidthShift;
        m++;
      }

      // Global memory transactions
      gld_tran += glTransactions(readSeg.data(), n);
      gst_tran += glTransactions(writeSeg.data(), n);
      gld_req++;
      gst_req++;
    }
  }

  // Global write non-full cache-lines
  int cl_full_tmp, cl_part_tmp;
  countCacheLines(writeSegVolMmk.data(), volMmk, cacheWidth, cl_full_tmp, cl_part_tmp);
  cl_full_l2 += cl_full_tmp;
  cl_part_l2 += cl_part_tmp;

#ifdef CALC_L1_CACHELINES
#error "CALC_L1_CACHELINES currently not functional"
  countCacheLines(writePosVolMmk.data(), volMmk, accWidth, cl_full_tmp, cl_part_tmp);
  cl_full_l1 += cl_full_tmp;
  cl_part_l1 += cl_part_tmp;
#endif

}

#ifdef NO_ALIGNED_ALLOC
//
// From: http://stackoverflow.com/questions/12504776/aligned-malloc-in-c
//
void *aligned_malloc(size_t required_bytes, size_t alignment) {
    void *p1;
    void **p2;
    int offset=alignment-1+sizeof(void*);
    p1 = malloc(required_bytes + offset);               // the line you are missing
    p2=(void**)(((size_t)(p1)+offset)&~(alignment-1));  //line 5
    p2[-1]=p1; //line 6
    return p2;
}

void aligned_free( void* p ) {
    void* p1 = ((void**)p)[-1];         // get the pointer to the buffer we allocated
    free( p1 );
}
#endif

//
// Count number of global memory transactions for Packed -method
//
void countPackedGlTransactions0(const int warpSize, const int accWidth, const int cacheWidth,
  const int numthread, 
  const int numPos, const int posMbarIn[INT_VECTOR_LEN], const int posMbarOut[INT_VECTOR_LEN],
  const int volMmk,  const int* __restrict__ posMmkIn, const int* __restrict__ posMmkOut,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1) {

#ifdef NO_ALIGNED_ALLOC
  int_vector* writeSegVolMmk = (int_vector *)aligned_malloc(volMmk*sizeof(int_vector), sizeof(int_vector));
#else
  #if !defined(__AARCH64_QNX__)
  int_vector* writeSegVolMmk = (int_vector *)aligned_alloc(sizeof(int_vector), volMmk*sizeof(int_vector));
  #else
  int_vector* writeSegVolMmk = (int_vector *)memalign(sizeof(int_vector), volMmk*sizeof(int_vector));
  #endif
#endif

  const int accWidthShift = ilog2(accWidth);
  const int cacheWidthShift = ilog2(cacheWidth);

  int_vector posMbarInVec(posMbarIn);
  int_vector posMbarOutVec(posMbarOut);
  int_vector readSeg_prev(-1);
  int_vector writeSeg_prev(-1);
  int_vector gld_tran_tmp(0);
  int_vector gst_tran_tmp(0);
  for (int j=0;j < volMmk;) {
    int_vector posMmkInVec(posMmkIn[j]);
    int_vector posMmkOutVec(posMmkOut[j]);

    int_vector posIn  = posMbarInVec + posMmkInVec;
    int_vector posOut = posMbarOutVec + posMmkOutVec;
    int_vector readSeg = posIn >> accWidthShift;
    int_vector writeSeg = posOut >> accWidthShift;

    gld_tran_tmp += (readSeg != readSeg_prev);
    gst_tran_tmp += (writeSeg != writeSeg_prev);

    writeSegVolMmk[j] = (posOut >> cacheWidthShift);

    j++;
    readSeg_prev  = (j & 31) ? readSeg : int_vector(-1);
    writeSeg_prev = (j & 31) ? writeSeg : int_vector(-1);
  }

  // Global memory transactions
  int gld_tran_array[INT_VECTOR_LEN];
  int gst_tran_array[INT_VECTOR_LEN];
  gld_tran_tmp.copy(gld_tran_array);
  gst_tran_tmp.copy(gst_tran_array);
  for (int i=0;i < numPos;i++) {
    gld_tran += gld_tran_array[i];
    gst_tran += gst_tran_array[i];
  }
  gld_req += ((volMmk + warpSize - 1)/warpSize)*numPos;
  gst_req += ((volMmk + warpSize - 1)/warpSize)*numPos;

  // Global write non-full cache-lines
  int_vector cl_full_tmp, cl_part_tmp;
  countCacheLines0(writeSegVolMmk, volMmk, cacheWidth, cl_full_tmp, cl_part_tmp);
  int cl_full_array[INT_VECTOR_LEN];
  int cl_part_array[INT_VECTOR_LEN];
  cl_full_tmp.copy(cl_full_array);
  cl_part_tmp.copy(cl_part_array);
  for (int i=0;i < numPos;i++) {
    cl_full_l2 += cl_full_array[i];
    cl_part_l2 += cl_part_array[i];
  }

#ifdef CALC_L1_CACHELINES
#error "CALC_L1_CACHELINES currently not functional"
  countCacheLines(writePosVolMmk.data(), volMmk, accWidth, cl_full_tmp, cl_part_tmp);
  cl_full_l1 += cl_full_tmp;
  cl_part_l1 += cl_part_tmp;
#endif

#ifdef NO_ALIGNED_ALLOC
  aligned_free(writeSegVolMmk);
#else
  free(writeSegVolMmk);
#endif
}

//
// Count numnber of shared memory transactions for Packed -method
//
void countPackedShTransactions0(const int warpSize, const int bankWidth, const int numthread,
  const int volMmk, const TensorConv* msh, const int numMsh,
  int& sld_tran, int& sst_tran, int& sld_req, int& sst_req) {

  int p[32];
  int d[32];
  int add[32];
  for (int i=0;i < 32;i++) p[i] = 0;
  //
  int c_prev = msh[0].ct;
  int add_prev = msh[0].ct;
  int d_prev = 1;
  for (int i=0;i < numMsh;i++) {
    d[i] = msh[i].d;
    add[i] = add_prev + msh[i].ct - d_prev*c_prev;
    add_prev = add[i];
    c_prev = msh[i].ct;
    d_prev = msh[i].d;
  }

  const int bankWidthMask = bankWidth - 1;

  int pos = 0;

  for (int j00=0;j00 < volMmk;j00+=numthread) {
    int n0 = std::min(volMmk, j00 + numthread);
    for (int j0=j00;j0 < n0;j0+=warpSize) {
      // Number of accesses for each bank
      std::vector<int> numAccess(warpSize, 0);
      int maxNumAccess = 0;
      int n = std::min(warpSize, volMmk - j0);
      for (int j1=0;j1 < n;j1++) {
        int bank = pos & bankWidthMask;
        maxNumAccess = std::max(maxNumAccess, ++numAccess[bank]);
        // Advance position
        int ii = 0;
        while (++p[ii] == d[ii]) {
          p[ii] = 0;
          ii++;
        }
        pos += add[ii];
      }
      sld_tran += maxNumAccess;
      sst_tran++;
      sld_req++;
      sst_req++;
    }
  }
}

//
// Count numnber of shared memory transactions for Packed -method
// *** Slow reference version
//
void countPackedShTransactionsRef(const int warpSize, const int bankWidth, const int numthread,
  const int volMmk, const TensorConv* msh, const int numMsh,
  int& sld_tran, int& sst_tran, int& sld_req, int& sst_req) {

  const int bankWidthMask = bankWidth - 1;

  for (int j00=0;j00 < volMmk;j00+=numthread) {
    int n0 = std::min(volMmk, j00 + numthread);
    for (int j0=j00;j0 < n0;j0+=warpSize) {
      // Number of accesses for each bank
      std::vector<int> numAccess(warpSize, 0);
      int maxNumAccess = 0;
      int n = std::min(warpSize, volMmk - j0);
      for (int j1=0;j1 < n;j1++) {
        int j = j0 + j1;
        int pos = 0;
        for (int k=0;k < numMsh;k++) {
          pos += ((j / msh[k].c) % msh[k].d) * msh[k].ct;
        }
        int bank = pos & bankWidthMask;
        maxNumAccess = std::max(maxNumAccess, ++numAccess[bank]);
      }
      sld_tran += maxNumAccess;
      sst_tran++;
      sld_req++;
      sst_req++;
    }
  }
}

//
// Count number of global memory transactions for Tiled method
//
void countTiledGlTransactions(const bool isCopy,
  const int numPosMbarSample, const int volMm, const int volMk, const int volMbar,
  const int cIn, const int cOut, const int accWidth, const int cacheWidth,
  std::vector<TensorConvInOut>& hostMbar, const int sizeMbar,
  int& num_iter, float& mlp, int& gld_tran, int& gst_tran, int& gld_req, int& gst_req, int& cl_full, int& cl_part) {

  int ntile = ((volMm - 1)/TILEDIM + 1)*((volMk - 1)/TILEDIM + 1);
  num_iter = volMbar*ntile;

  gld_tran = 0;
  gst_tran = 0;
  gld_req = 0;
  gst_req = 0;
  cl_full = 0;
  cl_part = 0;

  // Random number generator
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, volMbar - 1);

  // Number of elements inside the horizontally clipped tiles
  int h = volMm % TILEDIM;
  // Number of elements inside the vertically clipped tiles
  int v = volMk % TILEDIM;

  // Number of full tiles
  int ntile_full = (volMm/TILEDIM)*(volMk/TILEDIM);
  // Number of tiles that are clipped in horizontal direction
  int ntile_horz = (h > 0)*(volMk/TILEDIM);
  // Number of tiles that are clipped in vertical direction
  int ntile_vert = (v > 0)*(volMm/TILEDIM);
  // Number of corner tiles (0 or 1)
  int ntile_corn = (h > 0)*(v > 0);

  if (isCopy) {
    // Total number of memory level parallelism
    int mlp_tot = (TILEDIM/TILEROWS)*(ntile_full + ntile_horz) + ((v - 1)/TILEROWS + 1)*(ntile_vert + ntile_corn);
    // Average memory level parallelism per tile
    mlp = (float)mlp_tot/(float)ntile;
  } else {
    // Total number of memory level parallelism
    int mlp_tot = (TILEDIM/TILEROWS)*(2*ntile_full + ntile_horz + ntile_vert) + 
    ((v - 1)/TILEROWS + 1)*(ntile_vert + ntile_corn) + ((h - 1)/TILEROWS + 1)*(ntile_horz + ntile_corn);
    // Average memory level parallelism per tile
    mlp = (float)mlp_tot/(float)(2*ntile);
  }

  int num_iposMbar = (numPosMbarSample == 0) ? volMbar : numPosMbarSample;

  for (int iposMbar=0;iposMbar < num_iposMbar;iposMbar++) {
    int posMbar = (numPosMbarSample == 0) ? iposMbar : distribution(generator);

    int posMbarIn;
    int posMbarOut;
    computePos(posMbar, posMbar, hostMbar.data(), sizeMbar, &posMbarIn, &posMbarOut);
    // computePos(posMbar, posMbar, hostMbar.begin(), hostMbar.begin() + sizeMbar, posMbarInV, posMbarOutV);

    // Reads happen at {posMbarIn, posMbarIn + cuDimMk, posMbarIn + 2*cuDimMk, ..., posMbarIn + (TILEDIM - 1)*cuDimMk}
    // Each tile has same number of transactions

    if (ntile_full > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      for (int i=0;i < TILEDIM;i++) {
        int posIn  = posMbarIn + i*cIn;
        int posOut = posMbarOut + i*cOut;
        gld_tran_tmp += glTransactions(posIn, TILEDIM, accWidth);
        gst_tran_tmp += glTransactions(posOut, TILEDIM, accWidth);
        int cl_full_tmp2, cl_part_tmp2;
        countCacheLines(posOut, TILEDIM, cacheWidth, cl_full_tmp2, cl_part_tmp2);
        cl_full_tmp += cl_full_tmp2;
        cl_part_tmp += cl_part_tmp2;
      }
      gld_tran += gld_tran_tmp*ntile_full;
      gst_tran += gst_tran_tmp*ntile_full;
      cl_full += cl_full_tmp*ntile_full;
      cl_part += cl_part_tmp*ntile_full;
    }

    if (ntile_horz > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      if (isCopy) {
        for (int i=0;i < TILEDIM;i++) {
          int posIn  = posMbarIn + i*cIn;
          int posOut = posMbarOut + i*cOut;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
          gst_tran_tmp += glTransactions(posOut, h, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, h, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      } else {
        for (int i=0;i < TILEDIM;i++) {
          int posIn  = posMbarIn + i*cIn;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
        }
        for (int i=0;i < h;i++) {
          int posOut = posMbarOut + i*cOut;
          gst_tran_tmp += glTransactions(posOut, TILEDIM, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, TILEDIM, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      }
      gld_tran += gld_tran_tmp*ntile_horz;
      gst_tran += gst_tran_tmp*ntile_horz;
      cl_full += cl_full_tmp*ntile_horz;
      cl_part += cl_part_tmp*ntile_horz;
    }

    if (ntile_vert > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      if (isCopy) {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          int posOut = posMbarOut + i*cOut;
          gld_tran_tmp += glTransactions(posIn, TILEDIM, accWidth);
          gst_tran_tmp += glTransactions(posOut, TILEDIM, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, TILEDIM, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      } else {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          gld_tran_tmp += glTransactions(posIn, TILEDIM, accWidth);
        }
        for (int i=0;i < TILEDIM;i++) {
          int posOut = posMbarOut + i*cOut;
          gst_tran_tmp += glTransactions(posOut, v, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, v, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      }
      gld_tran += gld_tran_tmp*ntile_vert;
      gst_tran += gst_tran_tmp*ntile_vert;
      cl_full += cl_full_tmp*ntile_vert;
      cl_part += cl_part_tmp*ntile_vert;
    }

    if (ntile_corn > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      if (isCopy) {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          int posOut = posMbarOut + i*cOut;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
          gst_tran_tmp += glTransactions(posOut, h, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, h, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      } else {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
        }
        for (int i=0;i < h;i++) {
          int posOut = posMbarOut + i*cOut;
          gst_tran_tmp += glTransactions(posOut, v, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, v, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      }
      gld_tran += gld_tran_tmp*ntile_corn;
      gst_tran += gst_tran_tmp*ntile_corn;
      cl_full += cl_full_tmp*ntile_corn;
      cl_part += cl_part_tmp*ntile_corn;
    }

  }
  // Requests
  if (isCopy) {
    gld_req = num_iposMbar*( TILEDIM*ntile_full + TILEDIM*ntile_horz + v*ntile_vert + v*ntile_corn );
    gst_req = gld_req;
  } else {
    gld_req = num_iposMbar*( TILEDIM*ntile_full + TILEDIM*ntile_horz + v*ntile_vert + v*ntile_corn );
    gst_req = num_iposMbar*( TILEDIM*ntile_full + TILEDIM*ntile_vert + h*ntile_horz + h*ntile_corn );
  }
}

struct GpuModelProp {
  double base_dep_delay;
  double base_mem_latency;
  double sh_mem_latency;
  double iter_cycles;
  double fac;

  GpuModelProp(int major) {
    if (major <= 3) {
      // Kepler
      base_dep_delay = 14.0;
      base_mem_latency = 358.0;
      sh_mem_latency = 11.0;
      iter_cycles = 50.0;
      fac = 2.0;
    } else if (major <= 5) {
      // Maxwell
      base_dep_delay = 2.5;
      base_mem_latency = 385.0;
      sh_mem_latency = 1.0;
      iter_cycles = 220.0;
      fac = 2.0;
    } else {
      // Pascal and above
      base_dep_delay = 2.8;
      base_mem_latency = 485.0;
      sh_mem_latency = 1.0;
      iter_cycles = 260.0;
      fac = 2.0;
    } 
  }
};

void prepmodel5(cudaDeviceProp& prop, GpuModelProp& gpuModelProp,
  int nthread, int numActiveBlock, float mlp,
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran,
  int cl_full, int cl_part,
  double& delta_ll, double& mem_cycles, double& sh_mem_cycles, double& MWP) {

  double active_SM = prop.multiProcessorCount;
  // Memory bandwidth in GB/s
  double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  if (prop.ECCEnabled) mem_BW *= (1.0 - 0.125);
  // GPU clock in GHz
  double freq = (double)prop.clockRate/1.0e6;
  int warpSize = prop.warpSize;

  int active_warps_per_SM = nthread*numActiveBlock/warpSize;

  // avg. number of memory transactions per memory request
  // double num_trans_per_request = ((double)gld_tran + (double)gst_tran*(1.0 + part_cl)) / (double)(gld_req + gst_req);
  // double num_trans_per_request = ((double)gld_tran + (double)gst_tran + (double)cl_part) / (double)(gld_req + gst_req);
  double cl = (double)cl_part/(double)(cl_full + cl_part);
  double num_trans_per_request = ((double)gld_tran + ((double)gst_tran)*(1.0 + cl)) / (double)(gld_req + gst_req);
  double shnum_trans_per_request = (double)(sld_tran + sst_tran) / (double)(sld_req + sst_req);

  double mem_l = gpuModelProp.base_mem_latency + (num_trans_per_request - 1.0) * gpuModelProp.base_dep_delay;

  const double hitrate = 0.2;

  // Avg. number of memory cycles per warp per iteration
  mem_cycles = gpuModelProp.fac * mem_l * mlp;
  sh_mem_cycles = 2.0 * shnum_trans_per_request * gpuModelProp.sh_mem_latency * mlp;

  // The final value of departure delay
  double dep_delay = num_trans_per_request * gpuModelProp.base_dep_delay;

  // double bytes_per_request = num_trans_per_request*128;
  double bytes_per_request = (num_trans_per_request*(1.0 - hitrate) + hitrate)*128.0;

  delta_ll = gpuModelProp.base_dep_delay;
  double BW_per_warp = freq*bytes_per_request/mem_l;
  double MWP_peak_BW = mem_BW/(BW_per_warp*active_SM);
  MWP = mem_l / dep_delay;
  MWP = std::min(MWP*mlp, std::min(MWP_peak_BW, (double)active_warps_per_SM));
}

double cyclesPacked(const bool isSplit, const size_t sizeofType, cudaDeviceProp& prop,
  int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, int cl_full, int cl_part) {

  int warps_per_block = nthread/32;

  GpuModelProp gpuModelProp(prop.major);

  double delta_ll, mem_cycles, sh_mem_cycles, MWP;
  prepmodel5(prop, gpuModelProp, nthread, numActiveBlock, mlp,
    gld_req, gst_req, gld_tran, gst_tran,
    sld_req, sst_req, sld_tran, sst_tran, cl_full, cl_part,
    delta_ll, mem_cycles, sh_mem_cycles, MWP);
  double ldst_cycles = mem_cycles*warps_per_block/MWP;
  double sync_cycles = 0.0;//2.0*delta_ll*(warps_per_block - 1.0);
  double cycles = (ldst_cycles + sh_mem_cycles + sync_cycles + gpuModelProp.iter_cycles)*num_iter;

  return cycles;
}

double cyclesTiled(const bool isCopy, const size_t sizeofType, cudaDeviceProp& prop,
  int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, int cl_full, int cl_part) {

  int warps_per_block = nthread/32;

  GpuModelProp gpuModelProp(prop.major);

  double delta_ll, mem_cycles, sh_mem_cycles, MWP;
  prepmodel5(prop, gpuModelProp, nthread, numActiveBlock, mlp,
    gld_req, gst_req, gld_tran, gst_tran,
    sld_req, sst_req, sld_tran, sst_tran, cl_full, cl_part,
    delta_ll, mem_cycles, sh_mem_cycles, MWP);
  double ldst_cycles = mem_cycles*warps_per_block/MWP;
  double sync_cycles = 0.0;//2.0*delta_ll*(warps_per_block - 1.0);
  if (isCopy) {
    sh_mem_cycles = 0.0;
    sync_cycles = 0.0;
  }
  double cycles = (ldst_cycles + sh_mem_cycles + sync_cycles + gpuModelProp.iter_cycles)*num_iter;

  return cycles;
}

bool check_results(const int tran, const int cl_full, const int cl_part, const int* results) {
  if (tran != results[0] || cl_full != results[1] || cl_part != results[2] ) return false;
  return true;
}

void print_pos(const char* name, const int n, const int* pos) {
  printf("%s", name);
  for (int i=0;i < n;i++) {
    printf(" %d", pos[i]);
  }
  printf("\n");
}

bool testCounters(const int warpSize, const int accWidth, const int cacheWidth) {

  if (warpSize != 32) return false;

  const int numArray = 10;

  const int posData[numArray][32] =
{{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
{0,1,2,4,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{43,44,45,46,47,48,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,-1,-1,-1,-1},
{0,3,6,9,12,15,18,21,24,27,30,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,-1,-1,-1,-1,-1,-1},
{0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124},
{0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248},
{0,1,2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,102,104,106,108,110,112,114,116,118,120,-1,-1,-1,-1,-1},
{5,6,7,8,9,91,92,93,94,95,96,97,98,99,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,-1,-1,-1}};

  const int arrayResultsDouble[numArray][3] =
{{0, 0, 0},
{2, 8, 0},
{1, 0, 2},
{4, 6, 4},
{2, 0, 8},
{5, 0, 19},
{8, 0, 32},
{16, 0, 32},
{5, 0, 18},
{4, 5, 4}};

  const int arrayResultsFloat[numArray][3] = 
{{0, 0, 0},
{1, 4, 0},
{1, 0, 1},
{2, 2, 4},
{1, 0, 4},
{3, 0, 10},
{4, 0, 16},
{8, 0, 32},
{3, 0, 10},
{4, 1, 5}};

  const int contResultsDouble[16*33][3] =
{{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{1,3,1},{1,3,1},{1,3,1},{1,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{2,5,1},{2,5,1},{2,5,1},
{2,6,0},{2,6,1},{2,6,1},{2,6,1},{2,7,0},{2,7,1},{2,7,1},{2,7,1},{2,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},
{2,6,1},{2,6,2},{2,6,2},{2,6,2},{2,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{2,6,2},{2,6,2},{2,6,2},
{2,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},
{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{2,6,2},{2,6,2},{2,6,2},{2,7,1},{3,7,2},{3,7,2},{3,7,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{2,5,1},{2,5,1},{2,5,1},
{2,6,0},{2,6,1},{2,6,1},{2,6,1},{2,7,0},{3,7,1},{3,7,1},{3,7,1},{3,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},
{2,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{3,6,2},{3,6,2},{3,6,2},
{3,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},
{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{3,7,2},{3,7,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{2,5,1},{2,5,1},{2,5,1},
{2,6,0},{3,6,1},{3,6,1},{3,6,1},{3,7,0},{3,7,1},{3,7,1},{3,7,1},{3,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{3,5,2},{3,5,2},{3,5,2},
{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},
{3,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},
{2,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{3,7,2},{3,7,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{2,1,1},{2,1,1},{2,1,1},{2,2,0},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{3,5,1},{3,5,1},{3,5,1},
{3,6,0},{3,6,1},{3,6,1},{3,6,1},{3,7,0},{3,7,1},{3,7,1},{3,7,1},{3,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{3,4,2},{3,4,2},{3,4,2},{3,5,1},{3,5,2},{3,5,2},{3,5,2},
{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{3,4,2},{3,4,2},{3,4,2},{3,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},
{3,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{3,4,2},{3,4,2},{3,4,2},
{3,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{3,7,2},{3,7,2}};

  const int contResultsFloat[32*33][3] =
{{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},
{1,1,1},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},
{1,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},
{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},
{1,1,1},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},
{1,1,1},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{2,1,1},{2,1,1},{2,1,1},
{2,1,1},{2,1,1},{2,1,1},{2,1,1},{2,2,0},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},
{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},
{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2}
};

const int shTestData[138][3] = 
{{32, 6, 1},{1, 6, 1},{96, 180, 2},{1, 6, 1},{6, 30, 6},{960, 4680, 3},{1, 6, 1},{6, 30, 6},
{180, 26, 180},{96, 84, 2},{1, 6, 1},{6, 14, 6},{640, 2520, 3},{1, 6, 1},{6, 30, 84},
{180, 14, 6},{160, 756, 3},{1, 6, 1},{6, 9, 84},{54, 14, 6},{960, 4212, 4},{1, 6, 1},
{6, 3, 54},{18, 26, 162},{468, 9, 6},{960, 5616, 4},{1, 6, 1},{6, 4, 54},{24, 26, 216},
{624, 9, 6},{576, 2808, 4},{1, 6, 1},{6, 2, 54},{12, 26, 108},{312, 9, 6},{864, 2808, 4},
{1, 6, 1},{6, 2, 54},{12, 26, 108},{312, 9, 6},{864, 4212, 4},{1, 6, 1},{6, 3, 54},
{18, 26, 162},{468, 9, 6},{896, 4368, 4},{1, 6, 1},{6, 2, 84},{12, 26, 168},{312, 14, 6},
{1024, 5292, 4},{1, 6, 1},{6, 7, 756},{42, 9, 84},{378, 14, 6},{1024, 6048, 4},{1, 6, 1},
{6, 8, 756},{48, 9, 84},{432, 14, 6},{256, 1512, 4},{1, 6, 1},{6, 2, 756},{12, 9, 84},
{108, 14, 6},{768, 1512, 4},{1, 6, 1},{6, 2, 756},{12, 9, 84},{108, 14, 6},{768, 2268, 4},
{1, 6, 1},{6, 3, 756},{18, 9, 84},{162, 14, 6},{1024, 5994, 3},{1, 6, 1},{6, 111, 54},
{666, 9, 6},{1024, 6048, 3},{1, 6, 1},{6, 112, 54},{672, 9, 6},{128, 594, 3},{1, 6, 1},
{6, 11, 54},{66, 9, 6},{128, 648, 3},{1, 6, 1},{6, 12, 54},{72, 9, 6},{992, 5940, 4},
{1, 6, 1},{6, 5, 54},{30, 9, 6},{270, 22, 270},{960, 3564, 4},{1, 6, 1},{6, 3, 54},
{18, 9, 6},{162, 22, 162},{960, 4752, 4},{1, 6, 1},{6, 4, 54},{24, 9, 6},{216, 22, 216},
{1024, 5880, 3},{1, 6, 1},{6, 70, 84},{420, 14, 6},{1024, 5964, 3},{1, 6, 1},{6, 71, 84},
{426, 14, 6},{192, 1008, 3},{1, 6, 1},{6, 12, 84},{72, 14, 6},{1024, 5292, 4},{1, 6, 1},
{6, 7, 756},{42, 9, 84},{378, 14, 6},{1024, 6048, 4},{1, 6, 1},{6, 8, 756},{48, 9, 84},
{432, 14, 6},{928, 3780, 4},{1, 6, 1},{6, 5, 756},{30, 9, 84},{270, 14, 6},{928, 4536, 4},
{1, 6, 1},{6, 6, 756},{36, 9, 84},{324, 14, 6}};


  const int accWidthShift = ilog2(accWidth);
  const int cacheWidthShift = ilog2(cacheWidth);

  //
  // Test array version
  //
  for (int i=0;i < numArray;i++) {
    int n = 0;
    while (n < warpSize && posData[i][n] != -1) n++;
    std::vector<int> segData(warpSize);
    for (int j=0;j < n;j++) segData[j] = posData[i][j] >> accWidthShift;
    int tran = glTransactions(segData.data(), n);
    // int tran = glTransactions(posData[i], n, accWidth);
    int cl_full;
    int cl_part;
    for (int j=0;j < n;j++) segData[j] = posData[i][j] >> cacheWidthShift;
    countCacheLines(segData.data(), n, cacheWidth, cl_full, cl_part);
    // countCacheLines(posData[i], n, cacheWidth, cl_full, cl_part);
    bool ok = true;
    const int *results = (accWidth == 16) ? arrayResultsDouble[i] : arrayResultsFloat[i];
    ok = check_results(tran, cl_full, cl_part, results);
    if (!ok) {
      for (int j=0;j < warpSize;j++) {
        int pos = posData[i][j];
        if (pos == -1) {
          printf("X ");
        } else {
          printf("%d ", pos);
        }
      }
      printf("\n");
      for (int j=0;j < n;j++) {
        printf("%d ", segData[j]);
      }
      printf("\n");
      printf("n %d | tran %d cl %d %d | tran %d cl %d %d\n", n, tran, cl_full, cl_part, results[0], results[1], results[2]);
      return false;
    }
  }

  int numCont = accWidth*(warpSize + 1);
  int* gpuPosData = new int[(numArray + numCont)*warpSize];

  //
  // Test contigious version
  //
  {
    int i = 0;
    for (int pos=0;pos < accWidth;pos++) {
      for (int n=0;n <= warpSize;n++,i++) {
        int tran = glTransactions(pos, n, accWidth);
        int cl_full;
        int cl_part;
        countCacheLines(pos, n, cacheWidth, cl_full, cl_part);

        std::vector<int> posvec(warpSize, -1);
        for (int i=0;i <n;i++) posvec[i] = pos + i;

        memcpy(&gpuPosData[(numArray + i)*warpSize], posvec.data(), warpSize*sizeof(int));

        std::vector<int> segvec(warpSize);
        for (int i=0;i < n;i++) segvec[i] = posvec[i] >> accWidthShift;

        int tran2 = glTransactions(segvec.data(), n);
        // int tran2 = glTransactions(posvec.data(), n, accWidth);
        int cl_full2;
        int cl_part2;
        for (int i=0;i < n;i++) segvec[i] = posvec[i] >> cacheWidthShift;
        countCacheLines(segvec.data(), n, cacheWidth, cl_full2, cl_part2);

        bool ok = true;
        if (accWidth == 16) {
          ok = check_results(tran, cl_full, cl_part, contResultsDouble[i]);
        } else {
          ok = check_results(tran, cl_full, cl_part, contResultsFloat[i]);
        }

        if (tran != tran2 || cl_full != cl_full2 || cl_part != cl_part2) ok = false;

        if (!ok) {
          printf("%d:%d\n", pos, pos + n - 1);
          printf("tran %d %d cl_full %d %d cl_part %d %d\n", tran, tran2, cl_full, cl_full2, cl_part, cl_part2);
          return false;        
        }

      }
    }
  }

  //
  // Test shared memory transaction counter.
  //
  {
    int i = 0;
    while (i < 138) {
      int testInd = i;
      int numthread = shTestData[i][0];
      int volMmk    = shTestData[i][1];
      int numMsh    = shTestData[i][2];
      i++;
      std::vector<TensorConv> msh(numMsh);
      for (int j=0;j < numMsh;j++) {
        msh[j].c  = shTestData[i][0];
        msh[j].d  = shTestData[i][1];
        msh[j].ct = shTestData[i][2];
        i++;
      }

      int sld_tran_ref = 0, sst_tran_ref = 0, sld_req_ref = 0, sst_req_ref = 0;
      countPackedShTransactionsRef(warpSize, warpSize, numthread, volMmk, msh.data(), numMsh,
        sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);

      int sld_tran = 0, sst_tran = 0, sld_req = 0, sst_req = 0;
      countPackedShTransactions0(warpSize, warpSize, numthread, volMmk, msh.data(), numMsh,
        sld_tran, sst_tran, sld_req, sst_req);

      if (sld_tran != sld_tran_ref || sst_tran != sst_tran_ref ||
        sld_req != sld_req_ref || sst_req != sst_req_ref) {
        printf("Error in countPackedShTransactions. Test %d\n", testInd);
        printf("Ref: %d %d %d %d\n", sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
        printf("Vec: %d %d %d %d\n", sld_tran, sst_tran, sld_req, sst_req);
        return false;
      }

    }
  }

  {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> bindist(0, 1);
    std::uniform_int_distribution<int> dimdist(2, 10);
    for (int rank=3;rank <= 6;rank++) {

      for (int nsample=0;nsample < 100;nsample++) {

        // Take random dimensions
        int dim[32];
        for (int i=0;i < rank;i++) {
          dim[i] = dimdist(generator);
        }    

        // Include random set of sub-ranks
        int subrank = 0;
        int rankIn[32];
        while (subrank == 0) {
          subrank = 0;
          for (int i=0;i < rank;i++) {
            int val = bindist(generator);
            rankIn[i] = val;
            subrank += val;
          }
        }

        // Build tensor conversion
        int vol;
        std::vector<TensorConvInOut> conv(32);
        {
          int j = 0;
          int c = 1;
          int ct = 1;
          for (int i=0;i < rank;i++) {
            if (rankIn[i]) {
              conv[j].c_in  = c;
              conv[j].d_in  = dim[i];
              conv[j].ct_in = ct;
              //
              conv[j].c_out  = conv[j].c_in;
              conv[j].d_out  = conv[j].d_in;
              conv[j].ct_out = conv[j].ct_in;
              //
              j++;
              c *= dim[i];
            }
            ct *= dim[i];
          }
          vol = c;
        }

        int dIn[32];
        int cIn[32];
        int c_prev = conv[0].ct_in;
        int cIn_prev = conv[0].ct_in;
        int d_prev = 1;
        for (int i=0;i < subrank;i++) {
          dIn[i] = conv[i].d_in;
          cIn[i] = cIn_prev + conv[i].ct_in - d_prev*c_prev;
          cIn_prev = cIn[i];
          c_prev = conv[i].ct_in;
          d_prev = conv[i].d_in;
        }

        std::vector<int> posIn(vol, -1);
        std::vector<int> posOut(vol, -1);
        std::vector<int> posInRef(vol, -2);
        std::vector<int> posOutRef(vol, -2);
#ifdef ENABLE_NVTOOLS
        gpuRangeStart("computePos0");
#endif
        computePos0(vol, conv.data(), subrank, posIn.data(), posOut.data());
        // computePos0(vol, dIn, cIn, dIn, cIn, posIn.data(), posOut.data());
#ifdef ENABLE_NVTOOLS
        gpuRangeStop();
        gpuRangeStart("computePosRef");
#endif
        computePosRef(0, vol - 1, conv.begin(), conv.begin() + subrank, posInRef, posOutRef);
#ifdef ENABLE_NVTOOLS
        gpuRangeStop();
#endif
        for (int i=0;i < vol;i++) {
          if (posIn[i] != posInRef[i] || posOut[i] != posOutRef[i]) {
            printf("computePos0 fails rank %d subrank %d\n", rank, subrank);
            for (int j=0;j < rank;j++) {
              if (rankIn[j]) {
                printf("%d* ", dim[j]);
              } else {
                printf("%d ", dim[j]);
              }
            }
            printf("\n");
            printf("d");
            for (int j=0;j < subrank;j++) {
              printf(" %d", dIn[j]);
            }
            printf(" c");
            for (int j=0;j < subrank;j++) {
              printf(" %d", cIn[j]);
            }
            printf(" ct");
            for (int j=0;j < subrank;j++) {
              printf(" %d", conv[j].ct_in);
            }
            printf("\n");
            print_pos("posIn   ", vol, posIn.data());
            print_pos("posInRef", vol, posInRef.data());
            return false;
          }
        }

        // print_pos("posIn", vol, posIn.data());
        // print_pos("posInRef", vol, posInRef.data());

      }

    }

  }

  //
  // Test GPU version
  //
  {
    for (int i=0;i < numArray;i++) {
      memcpy(&gpuPosData[i*warpSize], posData[i], warpSize*sizeof(int));
    }
    int* tran_data = new int[numArray + numCont];
    int* cl_full_data = new int[numArray + numCont];
    int* cl_part_data = new int[numArray + numCont];
    runCounters(warpSize, gpuPosData, (numArray + numCont)*warpSize, accWidth, cacheWidth, tran_data, cl_full_data, cl_part_data);

    for (int i=0;i < numArray;i++) {
      bool ok = true;
      const int *p = (accWidth == 16) ? arrayResultsDouble[i] : arrayResultsFloat[i];
      if (accWidth == 16) {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      } else {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      }
      if (!ok) {
        printf("Array %d\n", i);
        for (int j=0;j < warpSize;j++) {
          int pos = gpuPosData[i*warpSize + j];
          if (pos == -1) {
            printf("X ");
          } else {
            printf("%d ", pos);
          }
        }
        printf("\n");
        printf("tran %d cl %d %d\n", tran_data[i], cl_full_data[i], cl_part_data[i]);
        printf("tran %d cl %d %d\n", p[0], p[1], p[2]);
        return false;
      }
    }

    for (int i=numArray;i < numArray + numCont;i++) {
      bool ok = true;
      const int *p = (accWidth == 16) ? contResultsDouble[i - numArray] : contResultsFloat[i - numArray];
      if (accWidth == 16) {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      } else {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      }
      if (!ok) {
        printf("Cont %d\n", i - numArray);
        for (int j=0;j < warpSize;j++) {
          int pos = gpuPosData[i*warpSize + j];
          if (pos == -1) {
            printf("X ");
          } else {
            printf("%d ", pos);
          }
        }
        printf("\n");
        printf("tran %d cl %d %d\n", tran_data[i], cl_full_data[i], cl_part_data[i]);
        printf("tran %d cl %d %d\n", p[0], p[1], p[2]);
        return false;
      }
    }

    delete [] tran_data;
    delete [] cl_full_data;
    delete [] cl_part_data;
  }

  delete [] gpuPosData;

  return true;
}
