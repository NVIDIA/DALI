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

#ifndef DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUTTPLAN_H
#define DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUTTPLAN_H

#include <list>
#include <vector>
#include <cuda_runtime_api.h>

#include "dali/util/dynlink_cuda.h"
#include "dali/pipeline/operators/transpose/cutt/cuttTypes.h"

const int TILEDIM = 32;
const int TILEROWS = 8;

// Transposing methods
enum {Unknown, Trivial, Packed, PackedSplit,
  Tiled, TiledCopy,
  NumTransposeMethods};

// Tells how tensor is split into Mm and Mk and what method is used
// NOTE: sizeMm and sizeMk fully define the split
class TensorSplit {
public:
  // Transposing method
  int method;

  // Input volume
  int sizeMm;
  int volMm;

  // Output volume
  int sizeMk;
  int volMk;

  // {Input} U {Output}
  int sizeMmk;
  int volMmk;

  // {Input} CUT {Output} = Mk which is not in Mm
  int sizeMkBar;
  int volMkBar;

  // Remaining volume
  int sizeMbar;
  int volMbar;

  // For Packed and PackedSplit methods:
  // Amount of contigious volume
  int volMmkInCont;
  int volMmkOutCont;

  // For PackedSplit method:
  // Number of splits
  int numSplit;

  // Rank that is split
  int splitRank;
  int splitDim;

  // volMmk that is left unsplit
  int volMmkUnsplit;

  TensorSplit();

  void print();

  void update(const int sizeMm_in, const int sizeMk_in, const int rank,
    const int* dim, const int* permutation);

  // Number of elements in shared memory space
  size_t shmem() const;

  // Number of elements in Mmk that are used effectively
  size_t volMmkUsed() const;

  // Bytes the shared memory space that needs to be allocated
  // (can be larger than volShmem() due to padding)
  size_t shmemAlloc(int sizeofType) const;

};

class LaunchConfig {
public:
 // Kernel launch configuration
  dim3 numthread;
  dim3 numblock;
  size_t shmemsize;

  // For the Packed method, number of registers to use for storage
  int numRegStorage;

  void print();

 };

// Class that stores the plan data
class cuttPlan_t {
public:
  // Device for which this plan was made
  int deviceID;

  // CUDA stream associated with the plan
  cudaStream_t stream;

  // Kernel launch configuration
  LaunchConfig launchConfig;
  
  // Rank of the tensor
  int rank;

  // Size of the tensor elements in bytes
  size_t sizeofType;

  TensorSplit tensorSplit;

  // Number of active thread blocks
  int numActiveBlock;

  int cuDimMk;
  int cuDimMm;

  int2 tiledVol;

  // Number of iterations of the kernel
  int num_iter;
  // Average memory level parallelism = average unroll count
  float mlp;
  int gld_req, gst_req, gld_tran, gst_tran;
  int cl_full_l2, cl_part_l2;
  int cl_full_l1, cl_part_l1;
  int sld_req, sst_req, sld_tran, sst_tran;
  double cycles;

  //--------------
  // Host buffers
  //--------------
  std::vector<TensorConvInOut> hostMbar;
  std::vector<TensorConvInOut> hostMmk;
  std::vector<TensorConv> hostMsh;

  //----------------
  // Device buffers
  //----------------
  // sizeMbar
  TensorConvInOut* Mbar;

  // sizeMmk
  TensorConvInOut* Mmk;

  // sizeMmk
  TensorConv* Msh;

  // For TiledSingleInRank
  TensorConv* Mk;

  // For TiledSingleOutRank
  TensorConv* Mm;

  cuttPlan_t();
  ~cuttPlan_t();
  void print();
  void setStream(cudaStream_t stream_in);
  bool countCycles(cudaDeviceProp& prop, const int numPosMbarSample=0);
  void activate();
  void nullDevicePointers();

  static bool createPlans(const int rank, const int* dim, const int* permutation,
    const int redRank, const int* redDim, const int* redPermutation,
    const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans);

private:
  static bool createTrivialPlans(const int rank, const int* dim, const int* permutation,
    const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans);

  static bool createTiledPlans(const int rank, const int* dim, const int* permutation,
    const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans);

  static bool createTiledCopyPlans(const int rank, const int* dim, const int* permutation,
    const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans);

  static bool createPackedPlans(const int rank, const int* dim, const int* permutation,
    const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans);

  static bool createPackedSplitPlans(const int rank, const int* dim, const int* permutation,
    const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans);

  bool setup(const int rank_in, const int* dim, const int* permutation,
    const size_t sizeofType_in, const TensorSplit& tensorSplit_in,
    const LaunchConfig& launchConfig_in, const int numActiveBlock_in);

};

void printMatlab(cudaDeviceProp& prop, std::list<cuttPlan_t>& plans, std::vector<double>& times);

void reduceRanks(const int rank, const int* dim, const int* permutation,
  std::vector<int>& redDim, std::vector<int>& redPermutation);

std::list<cuttPlan_t>::iterator choosePlanHeuristic(std::list<cuttPlan_t>& plans);

#endif // DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUTTPLAN_H
