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
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <cmath>
#include <random>

#include "dali/pipeline/operators/transpose/cutt/cuttplan.h"

#include "dali/error_handling.h"
#include "dali/pipeline/operators/transpose/cutt/CudaUtils.h"
#include "dali/pipeline/operators/transpose/cutt/cuttkernel.h"
#include "dali/pipeline/operators/transpose/cutt/cuttGpuModel.h"
#include "dali/pipeline/operators/transpose/cutt/cuttGpuModelKernel.h"

void printMethod(int method) {
  switch(method) {
    case Trivial:
    printf("Trivial");
    break;
    case Packed:
    printf("Packed");
    break;
    case PackedSplit:
    printf("PackedSplit");
    break;
    case Tiled:
    printf("Tiled");
    break;
    case TiledCopy:
    printf("TiledCopy");
    break;
    case Unknown:
    printf("Unknown");
    return;
    break;
  };  
}

//
// Reduce ranks by combining groups of ranks are in consequtive order
//
void reduceRanks(const int rank, const int* dim, const int* permutation,
  std::vector<int>& redDim, std::vector<int>& redPermutation) {

  // Previous permutation value,
  // start with impossible value so that we always first do push_back(permutation[0])
  int prev = -2;
  for (int i=0;i < rank;i++) {
    int cur = permutation[i];
    if (cur == prev + 1)
    {
      // Skip over ranks that are in consequtive order and
      // combine dimensions
      redDim.back() *= dim[cur];
    } else {
      // Include ranks that start the consequtive sequence
      redPermutation.push_back(cur);
      // NOTE: redDim will be in permuted order, re-order after dust settles
      redDim.push_back(dim[cur]);
    }
    prev = cur;
  }

  // Re-number redPermutation
  std::vector<int> tmp(rank, -1);
  for (int i=0;i < static_cast<int>(redPermutation.size()); i++) {
    tmp[redPermutation[i]] = i;
  }
  int j = 0;
  for (int i=0; i < rank; i++) {
    if (tmp[i] != -1) {
      tmp[j++] = tmp[i];
    }
  }
  for (int i=0 ;i < static_cast<int>(redPermutation.size());i++) {
    redPermutation[tmp[i]] = i;
  }

  // Re-order redDim
  for (int i=0;i < static_cast<int>(redDim.size());i++) {
    tmp[redPermutation[i]] = redDim[i];
  }  
  for (int i=0;i < static_cast<int>(redDim.size());i++) {
    redDim[i] = tmp[i];
  }

  // for (int i=0;i < rank;i++) {
  //   printf("%d ", dim[i]);
  // }
  // printf("| ");
  // for (int i=0;i < rank;i++) {
  //   printf("%d ", permutation[i]);
  // }
  // printf("\n");

  // for (int i=0;i < redPermutation.size();i++) {
  //   printf("%d ", redDim[i]);
  // }
  // printf("| ");
  // for (int i=0;i < redPermutation.size();i++) {
  //   printf("%d ", redPermutation[i]);
  // }
  // printf("\n");

}

//
// Stores tensor c object
//
class TensorC {
private:
  const int rank;
  int* c;
  // map[i] tells where to find rank i in c[]
  int* map;
public:
  // rankInd[0 ... n - 1] = ranks that are included
  TensorC(const int rank, const int n, const int* rankInd, const int* dim) : rank(rank) {
    if (rank < 1 || n < 1 || n > rank) {
      printf("TensorC::TensorC, Invalid rank or n\n");
      exit(1);
    }
    map = new int[rank];
    for (int i=0;i < rank;i++) map[i] = -1;
    for (int i=0;i < n;i++) {
      map[rankInd[i]] = i;
    }
    c = new int[n];
    c[0] = 1;
    for (int i=1;i < n;i++) {
      c[i] = c[i-1]*dim[rankInd[i-1]];
    }
  }

  ~TensorC() {
    delete [] c;
    delete [] map;
  }

  int get(const int i) {
    int mapi;
    if (i < 0 || i >= rank || (mapi = map[i]) == -1) {
      printf("TensorC::get(), index out of range\n");
      exit(1);
    }
    return c[mapi];
  }

};

TensorSplit::TensorSplit() {
  method = Unknown;
  sizeMm = 0;
  volMm = 0;
  sizeMk = 0;
  volMk = 0;
  sizeMmk = 0;
  volMmk = 0;
  sizeMkBar = 0;
  volMkBar = 0;
  sizeMbar = 0;
  volMbar = 0;
  volMmkInCont = 0;
  volMmkOutCont = 0;
  numSplit = 1;
  splitRank = -1;
  splitDim = 0;
  volMmkUnsplit = 0;
}

void TensorSplit::print() {
  printf("sizeMm %d sizeMk %d sizeMmk %d sizeMbar %d sizeMkBar %d\n",
    sizeMm, sizeMk, sizeMmk, sizeMbar, sizeMkBar);
  printf("volMm %d volMk %d volMmk %d volMbar %d volMkBar %d\n",
    volMm, volMk, volMmk, volMbar, volMkBar);
  printf("volMmkInCont %d volMmkOutCont %d\n", volMmkInCont, volMmkOutCont);
  if (method == PackedSplit) printf("numSplit %d splitRank %d\n", numSplit, splitRank);
}

void TensorSplit::update(const int sizeMm_in, const int sizeMk_in, const int rank,
  const int* dim, const int* permutation) {

  sizeMm = sizeMm_in;
  sizeMk = sizeMk_in;

  // First sizeMm are in Mm
  volMm = 1;
  for (int i=0;i < sizeMm;i++) {
    volMm *= dim[i];
  }
  // First sizeMk in permuted order are in Mk
  volMk = 1;
  for (int i=0;i < sizeMk;i++) {
    volMk *= dim[permutation[i]];
  }

  int vol = 1;
  volMmk = 1;
  sizeMmk = 0;
  volMkBar = 1;
  sizeMkBar = 0;
  for (int i=0;i < rank;i++) {
    int pi = permutation[i];
    if (i < sizeMm) {
      volMmk *= dim[i];
      sizeMmk++;
    }
    if (i < sizeMk && pi >= sizeMm) {
      volMmk *= dim[pi];
      sizeMmk++;
      volMkBar *= dim[pi];
      sizeMkBar++;
    }
    vol *= dim[i];
  }

  sizeMbar = rank - sizeMmk;
  volMbar = vol/volMmk;

  if (splitRank >= 0) {
    splitDim = dim[splitRank];
    volMmkUnsplit = volMmk / splitDim;
  }

  std::vector<bool> isMmk(rank, false);
  for (int i=0;i < rank;i++) {
    if (i < sizeMm) {
      isMmk[i] = true;
    }
    if (i < sizeMk) {
      int pi = permutation[i];
      isMmk[pi] = true;
    }
  }

  volMmkInCont = 1;
  for (int i=0;i < rank;i++) {
    if (!isMmk[i]) break;
    if (i == splitRank) {
      volMmkInCont *= splitDim / numSplit + (splitDim % numSplit > 0);
      break;
    } else {
      volMmkInCont *= dim[i];
    }
  }

  volMmkOutCont = 1;
  for (int i=0;i < rank;i++) {
    int pi = permutation[i];
    if (!isMmk[pi]) break;
    if (pi == splitRank) {
      volMmkOutCont *= splitDim / numSplit + (splitDim % numSplit > 0);
      break;
    } else {
      volMmkOutCont *= dim[pi];
    }
  }

}

bool operator==(const TensorSplit& lhs, const TensorSplit& rhs) {
  if (lhs.method != rhs.method) return false;

  if (lhs.method == Trivial) return true;

  if (lhs.method == Tiled) {
    return
    (lhs.volMm == rhs.volMm) &&
    (lhs.volMk == rhs.volMk) &&
    (lhs.volMbar == rhs.volMbar);
  }

  if (lhs.method == TiledCopy) {
    return
    (lhs.volMm == rhs.volMm) &&
    (lhs.volMkBar == rhs.volMkBar) &&
    (lhs.volMbar == rhs.volMbar);
  }

  if (lhs.method == Packed || lhs.method == PackedSplit) {
    return
    (lhs.volMmkInCont == rhs.volMmkInCont) &&
    (lhs.volMmkOutCont == rhs.volMmkOutCont) &&
    (lhs.volMmk == rhs.volMmk) &&
    (lhs.volMbar == rhs.volMbar);
  }

  // if (lhs.method == Packed || lhs.method == PackedSplit) {
  //   return
  //   (lhs.sizeMmk == rhs.sizeMmk) &&
  //   (lhs.volMmk == rhs.volMmk) &&
  //   (lhs.sizeMbar == rhs.sizeMbar) &&
  //   (lhs.volMbar == rhs.volMbar) &&
  //   // (lhs.numActiveBlock == rhs.numActiveBlock) &&
  //   (lhs.numSplit == rhs.numSplit);
  // } else {
  //   return
  //   (lhs.sizeMm == rhs.sizeMm) &&
  //   (lhs.volMm == rhs.volMm) &&
  //   (lhs.sizeMk == rhs.sizeMk) &&
  //   (lhs.volMk == rhs.volMk) &&
  //   (lhs.sizeMmk == rhs.sizeMmk) &&
  //   (lhs.volMmk == rhs.volMmk) &&
  //   (lhs.sizeMkBar == rhs.sizeMkBar) &&
  //   (lhs.volMkBar == rhs.volMkBar) &&
  //   (lhs.sizeMbar == rhs.sizeMbar) &&
  //   (lhs.volMbar == rhs.volMbar) &&
  //   // (lhs.numActiveBlock == rhs.numActiveBlock) &&
  //   (lhs.numSplit == rhs.numSplit);    
  // }
  return false;
}

//
// Number of elements in shared memory space
//
size_t TensorSplit::shmem() const {

  size_t vol = 0;

  switch(method) {

    case Trivial:
    {
      vol = 0;
    }
    break;

    case Packed:
    {
      vol = volMmk;
    }
    break;

    case PackedSplit:
    {
      vol = (splitDim/numSplit + ((splitDim % numSplit) > 0))*volMmkUnsplit;
    }
    break;

    case Tiled:
    {
      vol = TILEDIM*TILEDIM;
    }
    break;

    case TiledCopy:
    {
      vol = 0;
    }
    break;

  }

  return vol;
}

//
// Number of elements in Mmk that are used effectively
//
size_t TensorSplit::volMmkUsed() const {
  size_t vol = 0;

  switch(method) {

    case Trivial:
    {
      vol = volMmk;
    }
    break;

    case Packed:
    {
      vol = volMmk;
    }
    break;

    case PackedSplit:
    {
      vol = (splitDim/numSplit)*volMmkUnsplit;
    }
    break;

    case Tiled:
    {
      vol = std::min(TILEDIM, volMm)*std::min(TILEDIM, volMk);
    }
    break;

    case TiledCopy:
    {
      vol = std::min(TILEDIM, volMm)*std::min(TILEDIM, volMk);
    }
    break;

  }

  return vol;
}

//
// Bytes the shared memory space that needs to be allocated
// (can be larger than shmem() due to padding)
//
size_t TensorSplit::shmemAlloc(int sizeofType) const {
  size_t vol = 0;

  switch(method) {

    case Trivial:
    {
      vol = 0;
    }
    break;

    case Packed:
    {
      vol = (size_t)volMmk*sizeofType;
    }
    break;

    case PackedSplit:
    {
      vol = (size_t)(splitDim/numSplit + ((splitDim % numSplit) > 0))*volMmkUnsplit*sizeofType;
    }
    break;

    case Tiled:
    {
      vol = (TILEDIM+1)*TILEDIM*sizeofType;
    }
    break;

    case TiledCopy:
    {
      vol = 0;
    }
    break;

  }

  return vol;
}

//
// Returns true if the plan with TensorSplit ts already exists
//
bool planExists(TensorSplit& ts, std::list<cuttPlan_t>& plans) {
  for (auto it=plans.begin();it != plans.end();it++) {
    if (it->tensorSplit == ts) return true;
  }
  return false;
}

bool cuttPlan_t::createTrivialPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  if (rank == 1) {
    TensorSplit ts;
    ts.method = Trivial;
    ts.update(1, 1, rank, dim, permutation);    
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, deviceID, prop, lc);
    if (numActiveBlock > 0 && !planExists(ts, plans)) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc, numActiveBlock)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool cuttPlan_t::createTiledPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  if (permutation[0] != 0 && rank > 1) {
    TensorSplit ts;
    ts.method = Tiled;
    ts.update(1, 1, rank, dim, permutation);    
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, deviceID, prop, lc);
    if (numActiveBlock > 0 && !planExists(ts, plans)) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc, numActiveBlock)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool cuttPlan_t::createTiledCopyPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  // Count number of Mm and Mk which are the same
  int numMmMkSame = 0;
  while (numMmMkSame < rank && permutation[numMmMkSame] == numMmMkSame) {
    numMmMkSame++;
  }
  if (numMmMkSame >= 1) {
    numMmMkSame = 1;
    TensorSplit ts;
    ts.method = TiledCopy;
    if (numMmMkSame < rank) {
      ts.update(numMmMkSame, numMmMkSame + 1, rank, dim, permutation);      
    } else {
      ts.update(numMmMkSame - 1, numMmMkSame, rank, dim, permutation);      
    }
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, deviceID, prop, lc);
    if (numActiveBlock > 0 && !planExists(ts, plans)) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc, numActiveBlock)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool cuttPlan_t::createPackedPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  LaunchConfig lc;
  for (int numMm=1;numMm < rank;numMm++) {
    for (int numMk=1;numMk < rank;numMk++) {
      TensorSplit ts;
      ts.method = Packed;
      ts.update(numMm, numMk, rank, dim, permutation);
      int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, deviceID, prop, lc);
      // Does not fit on the device, break out of inner loop
      if (numActiveBlock == 0) break;
      if (!planExists(ts, plans)) {
        cuttPlan_t plan;
        if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc, numActiveBlock)) return false;
        plans.push_back(plan);
      }
    }
  }

  return true;
}

bool cuttPlan_t::createPackedSplitPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  LaunchConfig lc;
  for (int numMm=1;numMm < rank;numMm++) {
    for (int numMk=1;numMk < rank;numMk++) {
      TensorSplit ts;
      ts.method = Packed;
      ts.update(numMm, numMk, rank, dim, permutation);
      // Amount of shared memory required
      size_t shmemsize = ts.shmemAlloc(sizeofType);
      if (shmemsize > prop.sharedMemPerBlock) {
        // Does not fit into shared memory, need to split
        ts.method = PackedSplit;
        // Minimum size of split dimension
        const int splitDimMin = 2;
        // Split the largest dimension
        int maxDim = 0;
        for (int i=0;i < ts.sizeMm;i++) {
          if (dim[i] > maxDim) {
            maxDim = dim[i];
            ts.splitRank = i;
          }
        }
        for (int i=0;i < ts.sizeMk;i++) {
          int pi = permutation[i];
          if (dim[pi] > maxDim) {
            maxDim = dim[pi];
            ts.splitRank = pi;
          }
        }
        //
        ts.update(numMm, numMk, rank, dim, permutation);
        int minNumSplit = (ts.splitDim*ts.volMmkUnsplit*sizeofType - 1)/prop.sharedMemPerBlock + 1;
        int maxNumSplit = std::max(minNumSplit, std::min(ts.splitDim/splitDimMin, minNumSplit + 60));

        // Sanity check: do not split too much
        if (minNumSplit > 10000) break;

        int bestNumSplit0 = 0;
        int bestVal1 = 0;
        int bestVal2 = 0;
        int bestNumSplit1 = 0;
        int bestNumSplit2 = 0;
        int numActiveBlock = 0;
        // Store number of active blocks and launch configs here so they
        // can be reused in plan.setup()
        int numActiveBlock0 = -1, numActiveBlock1 = -1, numActiveBlock2 = -1;
        LaunchConfig lc0, lc1, lc2;
        for (ts.numSplit=minNumSplit;ts.numSplit <= maxNumSplit;ts.numSplit++) {
          numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, deviceID, prop, lc);
          if (numActiveBlock != 0) {
            int volMmkUsed = ts.volMmkUsed();
            int val1 = volMmkUsed*numActiveBlock;
            int val2 = (lc.numthread.x*lc.numRegStorage*100)/volMmkUsed;
            if (bestVal1 < val1) {
              bestVal1 = val1;
              bestNumSplit1 = ts.numSplit;
              numActiveBlock1 = numActiveBlock;
              lc1 = lc;
            }
            if (bestVal2 < val2) {
              bestVal2 = val2;
              bestNumSplit2 = ts.numSplit;
              numActiveBlock2 = numActiveBlock;
              lc2 = lc;
            }
            if (bestNumSplit0 == 0) {
              bestNumSplit0 = ts.numSplit;
              numActiveBlock0 = numActiveBlock;
              lc0 = lc;
            }
          }
        }
        // Does not fit on the device, break out of inner loop
        if (numActiveBlock == 0) break;
        ts.numSplit = bestNumSplit0;
        ts.update(numMm, numMk, rank, dim, permutation);
        // Make sure splitDim*numSplit fits into an integer
        const unsigned long long int dim_cutoff = ((unsigned long long int)1 << 31);
        unsigned long long int dim0 = (unsigned long long int)ts.splitDim*(unsigned long long int)(ts.numSplit + 1);
        if (!planExists(ts, plans) && dim0 < dim_cutoff) {
          cuttPlan_t plan;
          if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc0, numActiveBlock0)) return false;
          plans.push_back(plan);
        }
        if (bestNumSplit1 != bestNumSplit0) {
          ts.numSplit = bestNumSplit1;
          ts.update(numMm, numMk, rank, dim, permutation);
          unsigned long long int dim1 = (unsigned long long int)ts.splitDim*(unsigned long long int)(ts.numSplit + 1);
          if (!planExists(ts, plans) && dim1 < dim_cutoff) {
            cuttPlan_t plan;
            if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc1, numActiveBlock1)) return false;
            plans.push_back(plan);
          }
        }
        if (bestNumSplit2 != bestNumSplit0 && bestNumSplit2 != bestNumSplit1) {
          ts.numSplit = bestNumSplit2;
          ts.update(numMm, numMk, rank, dim, permutation);
          unsigned long long int dim2 = (unsigned long long int)ts.splitDim*(unsigned long long int)(ts.numSplit + 1);
          if (!planExists(ts, plans) && dim2 < dim_cutoff) {
            cuttPlan_t plan;
            if (!plan.setup(rank, dim, permutation, sizeofType, ts, lc2, numActiveBlock2)) return false;
            plans.push_back(plan);
          }
        }
      }
    }
  }

  return true;
}

//
// Create all possible plans
//
bool cuttPlan_t::createPlans(const int rank, const int* dim, const int* permutation,
  const int rankRed, const int* dimRed, const int* permutationRed,
  const size_t sizeofType, const int deviceID, const cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  size_t size0 = plans.size();
  if (!createTrivialPlans(rankRed, dimRed, permutationRed, sizeofType, deviceID, prop, plans)) return false;
  // If Trivial plan was created, that's the only one we need
  if (size0 != plans.size()) return true;
  if (!createTiledCopyPlans(rankRed, dimRed, permutationRed, sizeofType, deviceID, prop, plans)) return false;
  if (!createTiledPlans(rankRed, dimRed, permutationRed, sizeofType, deviceID, prop, plans)) return false;
  if (!createPackedPlans(rank, dim, permutation, sizeofType, deviceID, prop, plans)) return false;
  if (!createPackedSplitPlans(rank, dim, permutation, sizeofType, deviceID, prop, plans)) return false;
  if (rank != rankRed) {
    if (!createPackedSplitPlans(rankRed, dimRed, permutationRed, sizeofType, deviceID, prop, plans)) return false;
  }
  return true;
}

bool operator>(const cuttPlan_t& lhs, const cuttPlan_t& rhs) {

  const TensorSplit& lts = lhs.tensorSplit;
  const TensorSplit& rts = rhs.tensorSplit;

  // Trivial method always wins
  if (lts.method == Trivial) return true;
  if (rts.method == Trivial) return false;

  // double dp = fabs(lhs.cycles - rhs.cycles)/std::min(lhs.cycles, rhs.cycles);
  // if (dp < 0.15 && (lts.method == Packed || lts.method == PackedSplit) &&
  //   (rts.method == Packed || rts.method == PackedSplit)) {

  //   if (lts.method == Packed && rts.method == Packed) {
  //     return (lhs.numActiveBlock > rhs.numActiveBlock);
  //   } else if (lts.method == Packed && rts.method == PackedSplit) {
  //     if (lhs.cl_part_l1 == 0 && rhs.cl_part_l1 == 0) {
  //       if (lts.volMmkOutCont == rts.volMmkOutCont) {
  //         return (lhs.cycles < rhs.cycles);
  //       } else {
  //         return (lts.volMmkOutCont > rts.volMmkOutCont);
  //       }
  //     } else {
  //       return (lhs.cl_part_l1 == 0);
  //     }
  //   } else if (lts.method == PackedSplit && rts.method == Packed) {
  //     if (lhs.cl_part_l1 == 0 && rhs.cl_part_l1 == 0) {
  //       if (lts.volMmkOutCont == rts.volMmkOutCont) {
  //         return (lhs.cycles < rhs.cycles);
  //       } else {
  //         return (lts.volMmkOutCont > rts.volMmkOutCont);
  //       }
  //     } else {
  //       return (rhs.cl_part_l1 != 0);
  //     }
  //   } else { 
  //     //else if (lts.method == PackedSplit && rts.method == PackedSplit) {
  //     if (lhs.cl_part_l1 == rhs.cl_part_l1) {
  //       return (lts.volMmkOutCont > rts.volMmkOutCont);        
  //     } else {
  //       return (lhs.cl_part_l1 < rhs.cl_part_l1);
  //     }
  //   }

  // } else {
    return (lhs.cycles < rhs.cycles);
  // }
}

bool operator<(const cuttPlan_t& lhs, const cuttPlan_t& rhs) {
  return !(lhs > rhs);
}

//
// Returns best plan according to heuristic criteria
// Returns plans.end() on invalid input or when nothing can be chosen
//
std::list<cuttPlan_t>::iterator choosePlanHeuristic(std::list<cuttPlan_t>& plans) {

  // Choose the "largest" plan
  auto bestIt = plans.end();
  for (auto it=plans.begin();it != plans.end();it++) {
    if (bestIt == plans.end() || *bestIt < *it) {
      bestIt = it;
    }
  }

  return bestIt;
}

void printMatlab(cudaDeviceProp& prop, std::list<cuttPlan_t>& plans, std::vector<double>& times) {
  static int count = 0;
  count++;
  int i = 0;
  // Conversion factor from wallclok time to total number of cycles = (GPU clock in Hz) x #SM
  double freq_SM = (double)(prop.clockRate*1000)*(double)prop.multiProcessorCount;
  for (auto it=plans.begin();it != plans.end();it++,i++) {
    TensorSplit& ts = it->tensorSplit;
    LaunchConfig& lc = it->launchConfig;
    if (ts.method == Packed || ts.method == PackedSplit ||
      ts.method == Tiled || ts.method == TiledCopy)
    {
      int numthread = lc.numthread.x*lc.numthread.y*lc.numthread.z;
      printf("MATLAB %d %d %d %d %1.3f %d %d %d %d %d %d %d %d %d %d %d %d %d %e %e\n", count, ts.method,
        it->num_iter, numthread, it->mlp, it->numActiveBlock,
        it->gld_req, it->gst_req, it->gld_tran, it->gst_tran, 
        it->sld_req, it->sst_req, it->sld_tran, it->sst_tran,
        it->cl_full_l2, it->cl_part_l2, it->cl_full_l1, it->cl_part_l1, times[i]*freq_SM, it->cycles);
    }
  }
}

void LaunchConfig::print() {
  printf("numthread %d %d %d numblock %d %d %d shmemsize %d numRegStorage %d\n",
    numthread.x, numthread.y, numthread.z,
    numblock.x, numblock.y, numblock.z,
    (int)shmemsize, numRegStorage);
}

//
// Output contents of the plan
//
void cuttPlan_t::print() {
  printf("method ");
  printMethod(tensorSplit.method);
  printf("\n");
  tensorSplit.print();
  launchConfig.print();
  printf("numActiveBlock %d cycles %e\n", numActiveBlock, cycles);
}


//
// Setup plan
// NOTE: Expects that cuttKernelLaunchConfiguration() has been called to setup
// launchConfig_in and numActiveBlock_in
//
bool cuttPlan_t::setup(const int rank_in, const int* dim, const int* permutation,
  const size_t sizeofType_in, const TensorSplit& tensorSplit_in,
  const LaunchConfig& launchConfig_in, const int numActiveBlock_in) {
  
  rank = rank_in;
  sizeofType = sizeofType_in;
  tensorSplit = tensorSplit_in;
  numActiveBlock = numActiveBlock_in;
  launchConfig = launchConfig_in;
  if (numActiveBlock == 0) return false;

  std::vector<bool> isMm(rank, false);
  std::vector<bool> isMk(rank, false);
  for (int i=0;i < tensorSplit.sizeMm;i++) {
    isMm[i] = true;
  }
  for (int i=0;i < tensorSplit.sizeMk;i++) {
    isMk[permutation[i]] = true;
  }

  // Setup launch configuration
  // numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, tensorSplit, prop, launchConfig);

  // Build cI
  int* I = new int[rank];
  for (int i=0;i < rank;i++) {
    I[i] = i;
  }
  TensorC cI(rank, rank, I, dim);
  delete [] I;

  // Build cO
  TensorC cO(rank, rank, permutation, dim);

  if (tensorSplit.method == Tiled) {
    cuDimMk = cI.get(permutation[0]);
    cuDimMm = cO.get(0);
    tiledVol.x = dim[0];
    tiledVol.y = dim[permutation[0]];
  } else if (tensorSplit.method == TiledCopy) {
    int rankMk = permutation[tensorSplit.sizeMk - 1];
    cuDimMk = cI.get(rankMk);
    cuDimMm = cO.get(rankMk);
    tiledVol.x = tensorSplit.volMm;
    tiledVol.y = dim[rankMk];
  }

  // Build MmI
  std::vector<int> MmI(tensorSplit.sizeMm);
  {
    int iMm = 0;
    int iMk = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i]) {
        MmI[iMm++] = i;
      }
    }
  }

  if (tensorSplit.sizeMbar > 0) {
    // Build MbarI = {s_1, ...., s_h}, indices in input order
    int* MbarI = new int[tensorSplit.sizeMbar];
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (!(isMm[i] || isMk[i])) {
        MbarI[j] = i;
        j++;
      }
    }
    TensorC cMbarI(rank, tensorSplit.sizeMbar, MbarI, dim);

    // Build MbarO = {s_l1, ...., s_lh}, indices in output (permuted) order
    int* MbarO = new int[tensorSplit.sizeMbar];
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (!(isMm[pi] || isMk[pi])) {
        MbarO[j] = pi;
        j++;
      }
    }

    hostMbar.resize(tensorSplit.sizeMbar);
    for (int i=0;i < tensorSplit.sizeMbar;i++) {
      int si = MbarI[i];
      hostMbar[i].c_in  = cMbarI.get(si);
      hostMbar[i].d_in  = dim[si];
      hostMbar[i].ct_in = cI.get(si);
      int sli = MbarO[i];
      hostMbar[i].c_out  = cMbarI.get(sli);
      hostMbar[i].d_out  = dim[sli];
      hostMbar[i].ct_out = cO.get(sli);
    }

    delete [] MbarI;
    delete [] MbarO;
  }

  gld_req = 1;
  gst_req = 1;
  gld_tran = 1;
  gst_tran = 1;
  cl_full_l2 = 0;
  cl_part_l2 = 0;
  cl_full_l1 = 0;
  cl_part_l1 = 0;
  sld_tran = 1;
  sst_tran = 1;
  sld_req = 1;
  sst_req = 1;
  num_iter = 0;
  mlp = 0.0f;
  cycles = 0.0;

  if (tensorSplit.method == PackedSplit) {
    if (tensorSplit.splitRank < 0) return false;
    std::vector<int> dimSplit(dim, dim + rank);
    std::vector<int> dimSplitPlusOne(dim, dim + rank);
    cuDimMm = 1;
    cuDimMk = 1;
    dimSplit[tensorSplit.splitRank]        = tensorSplit.splitDim/tensorSplit.numSplit;
    dimSplitPlusOne[tensorSplit.splitRank] = tensorSplit.splitDim/tensorSplit.numSplit + 1;
    cuDimMm = cI.get(tensorSplit.splitRank);
    cuDimMk = cO.get(tensorSplit.splitRank);
    // Build MmkI = {q_1, ..., q_a}
    std::vector<int> MmkI(tensorSplit.sizeMmk);
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        MmkI[j] = i;
        j++;
      }
    }
    TensorC cMmkISplit(rank, tensorSplit.sizeMmk, MmkI.data(), dimSplit.data());
    TensorC cMmkISplitPlusOne(rank, tensorSplit.sizeMmk, MmkI.data(), dimSplitPlusOne.data());
    // Build MmkO = {q_t1, ..., q_ta}
    std::vector<int> MmkO(tensorSplit.sizeMmk);
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (isMm[pi] || isMk[pi]) {
        MmkO[j] = pi;
        j++;
      }
    }
    TensorC cMmkOSplit(rank, tensorSplit.sizeMmk, MmkO.data(), dimSplit.data());
    TensorC cMmkOSplitPlusOne(rank, tensorSplit.sizeMmk, MmkO.data(), dimSplitPlusOne.data());

    hostMmk.resize(tensorSplit.sizeMmk*2);
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Minor reading position
      int qi = MmkI[i];
      hostMmk[i].c_in                        = cMmkISplit.get(qi);
      hostMmk[i].d_in                        = dimSplit[qi];
      hostMmk[i].ct_in                       = cI.get(qi);
      hostMmk[i + tensorSplit.sizeMmk].c_in  = cMmkISplitPlusOne.get(qi);
      hostMmk[i + tensorSplit.sizeMmk].d_in  = dimSplitPlusOne[qi];
      hostMmk[i + tensorSplit.sizeMmk].ct_in = cI.get(qi);
      // Minor writing position
      int qti = MmkO[i];
      hostMmk[i].c_out                        = cMmkOSplit.get(qti);
      hostMmk[i].d_out                        = dimSplit[qti];
      hostMmk[i].ct_out                       = cO.get(qti);
      hostMmk[i + tensorSplit.sizeMmk].c_out  = cMmkOSplitPlusOne.get(qti);
      hostMmk[i + tensorSplit.sizeMmk].d_out  = dimSplitPlusOne[qti];
      hostMmk[i + tensorSplit.sizeMmk].ct_out = cO.get(qti);
    }

    hostMsh.resize(tensorSplit.sizeMmk*2);
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Shared memory reading position
      int qti = MmkO[i];
      hostMsh[i].c                        = cMmkOSplit.get(qti);
      hostMsh[i].d                        = dimSplit[qti];
      hostMsh[i].ct                       = cMmkISplit.get(qti);
      hostMsh[i + tensorSplit.sizeMmk].c  = cMmkOSplitPlusOne.get(qti);
      hostMsh[i + tensorSplit.sizeMmk].d  = dimSplitPlusOne[qti];
      hostMsh[i + tensorSplit.sizeMmk].ct = cMmkISplitPlusOne.get(qti);
    }
  }

  if (tensorSplit.method == Packed) {
    // Build MmkI = {q_1, ..., q_a}
    std::vector<int> MmkI(tensorSplit.sizeMmk);
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        MmkI[j] = i;
        j++;
      }
    }
    TensorC cMmkI(rank, tensorSplit.sizeMmk, MmkI.data(), dim);
    // Build MmkO = {q_t1, ..., q_ta}
    std::vector<int> MmkO(tensorSplit.sizeMmk);
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (isMm[pi] || isMk[pi]) {
        MmkO[j] = pi;
        j++;
      }
    }
    TensorC cMmkO(rank, tensorSplit.sizeMmk, MmkO.data(), dim);

    hostMmk.resize(tensorSplit.sizeMmk);
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Minor reading position
      int qi = MmkI[i];
      hostMmk[i].c_in  = cMmkI.get(qi);
      hostMmk[i].d_in  = dim[qi];
      hostMmk[i].ct_in = cI.get(qi);
      // Minor writing position
      int qti = MmkO[i];
      hostMmk[i].c_out  = cMmkO.get(qti);
      hostMmk[i].d_out  = dim[qti];
      hostMmk[i].ct_out = cO.get(qti);
    }

    hostMsh.resize(tensorSplit.sizeMmk);
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Shared memory reading position
      int qti = MmkO[i];
      hostMsh[i].c  = cMmkO.get(qti);
      hostMsh[i].d  = dim[qti];
      hostMsh[i].ct = cMmkI.get(qti);
    }
  }

  return true;
}

// #define COUNTCYCLE_CHECK

//
// Count the number of cycles using the MWP-CWP model
//
bool cuttPlan_t::countCycles(cudaDeviceProp& prop, const int numPosMbarSample) {

  // Number of elements that are loaded per memory transaction:
  // 128 bytes per transaction
  const int accWidth = 128/sizeofType;
  // L2 cache line width is 32 bytes
  const int cacheWidth = 32/sizeofType;

  if (tensorSplit.method == Tiled) {
    // Global memory
#ifdef ENABLE_NVTOOLS
    gpuRangeStart("countTiledGlTransactions");
#endif
    countTiledGlTransactions(false, numPosMbarSample, tensorSplit.volMm, tensorSplit.volMk, tensorSplit.volMbar,
      cuDimMk, cuDimMm, accWidth, cacheWidth, hostMbar, tensorSplit.sizeMbar,
      num_iter, mlp, gld_tran, gst_tran, gld_req, gst_req, cl_full_l2, cl_part_l2);
#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
#endif
    // Shared memory
    sld_tran = 1;
    sst_tran = 1;
    sld_req = 1;
    sst_req = 1;
  } else if (tensorSplit.method == TiledCopy) {
    // Global memory
#ifdef ENABLE_NVTOOLS
    gpuRangeStart("countTiledGlTransactions (copy)");
#endif
    countTiledGlTransactions(true, numPosMbarSample, tensorSplit.volMm, tensorSplit.volMkBar, tensorSplit.volMbar,
      cuDimMk, cuDimMm, accWidth, cacheWidth, hostMbar, tensorSplit.sizeMbar,
      num_iter, mlp, gld_tran, gst_tran, gld_req, gst_req, cl_full_l2, cl_part_l2);
#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
#endif
    // Shared memory
    sld_tran = 1;
    sst_tran = 1;
    sld_req = 1;
    sst_req = 1;
  } else if (tensorSplit.method == PackedSplit) {
    if (tensorSplit.splitRank < 0) return false;
#ifdef ENABLE_NVTOOLS
    gpuRangeStart("PackedSplit: init");
#endif
    num_iter = tensorSplit.volMbar*tensorSplit.numSplit;
    mlp = (float)launchConfig.numRegStorage;
    int dimSplit = tensorSplit.splitDim/tensorSplit.numSplit;
    // Number of splits that are "round up" i.e. "PlusOne"
    int num1 = tensorSplit.splitDim % tensorSplit.numSplit;
    int volMmk1 = (dimSplit + 1)*tensorSplit.volMmkUnsplit;
    // Number of splits that are "round down"
    int num0 = tensorSplit.numSplit - num1;
    int volMmk0 = dimSplit*tensorSplit.volMmkUnsplit;
    mlp = (float)(volMmk0*num0 + volMmk1*num1) / (float)(launchConfig.numthread.x*(num0 + num1));
    // Global memory
    gld_tran = 0;
    gst_tran = 0;
    gld_req = 0;
    gst_req = 0;
    cl_full_l2 = 0;
    cl_part_l2 = 0;
    cl_full_l1 = 0;
    cl_part_l1 = 0;
    // Random number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, tensorSplit.volMbar*tensorSplit.numSplit - 1);
    // Pre-compute posMmkIn and posMmkOut
    std::vector<int> posMmkIn0(volMmk0);
    std::vector<int> posMmkOut0(volMmk0);
#ifdef ENABLE_NVTOOLS
    gpuRangeStart("computePos");
#endif
    computePos0(volMmk0, hostMmk.data(), tensorSplit.sizeMmk, posMmkIn0.data(), posMmkOut0.data());
    // computePos(0, volMmk0 - 1, hostMmkFast.data(), tensorSplit.sizeMmk, posMmkIn0.data(), posMmkOut0.data());
#ifdef COUNTCYCLE_CHECK
    std::vector<int> posMmkIn0Ref(volMmk0);
    std::vector<int> posMmkOut0Ref(volMmk0);
    computePosRef(0, volMmk0 - 1, hostMmk.begin(), hostMmk.begin() + tensorSplit.sizeMmk,
      posMmkIn0Ref, posMmkOut0Ref);
    for (int i=0;i < volMmk0;i++) {
      if (posMmkIn0[i] != posMmkIn0Ref[i] || posMmkOut0[i] != posMmkOut0Ref[i]) {
        printf("%d %d | %d %d | i %d volMmk0 %d sizeMmk %d\n", posMmkIn0[i], posMmkIn0Ref[i], posMmkOut0[i], posMmkOut0Ref[i],
          i, volMmk0, tensorSplit.sizeMmk);
        for (int j=0;j < tensorSplit.sizeMmk;j++) {
          printf("%d %d %d %d %d %d\n", hostMmk[j].c_in, hostMmk[j].d_in, hostMmk[j].ct_in,
            hostMmk[j].c_out, hostMmk[j].d_out, hostMmk[j].ct_out);
        }
        return false;
      }
    }
#endif
#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
#endif
    std::vector<int> posMmkIn1(volMmk1);
    std::vector<int> posMmkOut1(volMmk1);
    if (num1 > 0) {
#ifdef ENABLE_NVTOOLS
      gpuRangeStart("computePos");
#endif
      computePos0(volMmk1, hostMmk.data() + tensorSplit.sizeMmk, tensorSplit.sizeMmk, posMmkIn1.data(), posMmkOut1.data());
      // computePos(0, volMmk1 - 1, hostMmkFast.data() + tensorSplit.sizeMmk, tensorSplit.sizeMmk,
      //   posMmkIn1.data(), posMmkOut1.data());
#ifdef COUNTCYCLE_CHECK
      std::vector<int> posMmkIn1Ref(volMmk1);
      std::vector<int> posMmkOut1Ref(volMmk1);
      computePosRef(0, volMmk1 - 1, hostMmk.begin() + tensorSplit.sizeMmk, hostMmk.begin() + tensorSplit.sizeMmk*2,
        posMmkIn1Ref, posMmkOut1Ref);
      for (int i=0;i < volMmk1;i++) {
        if (posMmkIn1[i] != posMmkIn1Ref[i] || posMmkOut1[i] != posMmkOut1Ref[i]) {
          printf("%d %d | %d %d | i %d volMmk1 %d sizeMmk %d\n", posMmkIn1[i], posMmkIn1Ref[i], posMmkOut1[i], posMmkOut1Ref[i],
            i, volMmk1, tensorSplit.sizeMmk);
          for (int j=0;j < tensorSplit.sizeMmk;j++) {
            printf("%d %d %d %d %d %d\n", hostMmk[j].c_in, hostMmk[j].d_in, hostMmk[j].ct_in,
              hostMmk[j].c_out, hostMmk[j].d_out, hostMmk[j].ct_out);
          }
          return false;
        }
      }
#endif
#ifdef ENABLE_NVTOOLS
      gpuRangeStop();
#endif
    }

    int num_ipos = (numPosMbarSample == 0) ? tensorSplit.volMbar*tensorSplit.numSplit : numPosMbarSample;

#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
    gpuRangeStart("PackedSplit: loop");
#endif

    std::vector<int> posTmp(num_ipos);
    int numRoundUp = 0;
    for (int ipos=0;ipos < num_ipos;ipos++) {
      posTmp[ipos] = (numPosMbarSample == 0) ? ipos : distribution(generator);
      int isplit = posTmp[ipos] % tensorSplit.numSplit;
      if (isplit < num1) numRoundUp++;
    }
    std::vector<int> pos(num_ipos);
    int indRoundUp = 0;
    int indRoundDown = numRoundUp;
    for (int ipos=0;ipos < num_ipos;ipos++) {
      int isplit = posTmp[ipos] % tensorSplit.numSplit;
      if (isplit < num1) {
        pos[indRoundUp++] = posTmp[ipos];
      } else {
        pos[indRoundDown++] = posTmp[ipos];        
      }
    }
    if (indRoundUp != numRoundUp || indRoundDown != num_ipos) {
      printf("cuttPlan_t::countCycles, fatal implemention bug\n");
      return false;
    }
    // Round up is in pos[0 ... numRoundUp - 1]
    // Round down is in pos[numRoundUp ... num_ipos - 1]

    // Round up splits
    for (int ipos=0;ipos < numRoundUp;ipos += INT_VECTOR_LEN) {
      int numPos = std::min(numRoundUp - ipos, INT_VECTOR_LEN);
      int posMbarIn[INT_VECTOR_LEN];
      int posMbarOut[INT_VECTOR_LEN];
      for (int i=0;i < numPos;i++) {
        int posMbar = pos[ipos + i] / tensorSplit.numSplit;
        int isplit  = pos[ipos + i] % tensorSplit.numSplit;
        int p0 = isplit*tensorSplit.splitDim/tensorSplit.numSplit;
        computePos(posMbar, posMbar, hostMbar.data(), tensorSplit.sizeMbar, &posMbarIn[i], &posMbarOut[i]);
        posMbarIn[i] += p0*cuDimMm;
        posMbarOut[i] += p0*cuDimMk;
      }
      for (int i=numPos;i < INT_VECTOR_LEN;i++) {
        posMbarIn[i]  = posMbarIn[numPos - 1];
        posMbarOut[i] = posMbarOut[numPos - 1];
      }

      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int gld_req_tmp = 0;
      int gst_req_tmp = 0;
      int cl_full_l2_tmp = 0;
      int cl_part_l2_tmp = 0;
      countPackedGlTransactions0(prop.warpSize, accWidth, cacheWidth, launchConfig.numthread.x,
        numPos, posMbarIn, posMbarOut, volMmk1, posMmkIn1.data(), posMmkOut1.data(),
        gld_tran_tmp, gst_tran_tmp, gld_req_tmp, gst_req_tmp,
        cl_full_l2_tmp, cl_part_l2_tmp, cl_full_l1, cl_part_l1);
      gld_tran += gld_tran_tmp;
      gst_tran += gst_tran_tmp;
      gld_req += gld_req_tmp;
      gst_req += gst_req_tmp;
      cl_full_l2 += cl_full_l2_tmp;
      cl_part_l2 += cl_part_l2_tmp;

#ifdef COUNTCYCLE_CHECK
      int gld_tran_ref = 0;
      int gst_tran_ref = 0;
      int gld_req_ref = 0;
      int gst_req_ref = 0;
      int cl_full_l2_ref = 0;
      int cl_part_l2_ref = 0;
      for (int i=0;i < numPos;i++) {
        countPackedGlTransactions(prop.warpSize, accWidth, cacheWidth, launchConfig.numthread.x,
          posMbarIn[i], posMbarOut[i], volMmk1, posMmkIn1, posMmkOut1,
          gld_tran_ref, gst_tran_ref, gld_req_ref, gst_req_ref,
          cl_full_l2_ref, cl_part_l2_ref, cl_full_l1, cl_part_l1);
      }
      if (gld_tran_tmp != gld_tran_ref || gst_tran_tmp != gst_tran_ref ||
        gld_req_tmp != gld_req_ref || gst_req_tmp != gst_req_ref) {
        printf("PackedSplit:countPackedGlTransactions0 ERROR\n");
        printf("tmp %d %d %d %d\n", gld_tran_tmp, gst_tran_tmp, gld_req_tmp, gst_req_tmp);
        printf("ref %d %d %d %d\n", gld_tran_ref, gst_tran_ref, gld_req_ref, gst_req_ref);
        return false;
      }
      if (cl_full_l2_tmp != cl_full_l2_ref || cl_part_l2_tmp != cl_part_l2_ref) {
        printf("PackedSplit:countPackedGlTransactions0 ERROR\n");
        printf("tmp %d %d\n", cl_full_l2_tmp, cl_part_l2_tmp);
        printf("ref %d %d\n",  cl_full_l2_ref, cl_part_l2_ref);
        return false;
      }
#endif
    }

    // Round down splits
    for (int ipos=numRoundUp;ipos < num_ipos;ipos += INT_VECTOR_LEN) {
      int numPos = std::min(num_ipos - ipos, INT_VECTOR_LEN);
      int posMbarIn[INT_VECTOR_LEN];
      int posMbarOut[INT_VECTOR_LEN];
      for (int i=0;i < numPos;i++) {
        int posMbar = pos[ipos + i] / tensorSplit.numSplit;
        int isplit  = pos[ipos + i] % tensorSplit.numSplit;
        int p0 = isplit*tensorSplit.splitDim/tensorSplit.numSplit;
        computePos(posMbar, posMbar, hostMbar.data(), tensorSplit.sizeMbar, &posMbarIn[i], &posMbarOut[i]);
        posMbarIn[i] += p0*cuDimMm;
        posMbarOut[i] += p0*cuDimMk;
      }
      for (int i=numPos;i < INT_VECTOR_LEN;i++) {
        posMbarIn[i]  = posMbarIn[numPos - 1];
        posMbarOut[i] = posMbarOut[numPos - 1];
      }

      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int gld_req_tmp = 0;
      int gst_req_tmp = 0;
      int cl_full_l2_tmp = 0;
      int cl_part_l2_tmp = 0;
      countPackedGlTransactions0(prop.warpSize, accWidth, cacheWidth, launchConfig.numthread.x,
        numPos, posMbarIn, posMbarOut, volMmk0, posMmkIn0.data(), posMmkOut0.data(),
        gld_tran_tmp, gst_tran_tmp, gld_req_tmp, gst_req_tmp,
        cl_full_l2_tmp, cl_part_l2_tmp, cl_full_l1, cl_part_l1);
      gld_tran += gld_tran_tmp;
      gst_tran += gst_tran_tmp;
      gld_req += gld_req_tmp;
      gst_req += gst_req_tmp;
      cl_full_l2 += cl_full_l2_tmp;
      cl_part_l2 += cl_part_l2_tmp;

#ifdef COUNTCYCLE_CHECK
      int gld_tran_ref = 0;
      int gst_tran_ref = 0;
      int gld_req_ref = 0;
      int gst_req_ref = 0;
      int cl_full_l2_ref = 0;
      int cl_part_l2_ref = 0;
      for (int i=0;i < numPos;i++) {
        countPackedGlTransactions(prop.warpSize, accWidth, cacheWidth, launchConfig.numthread.x,
          posMbarIn[i], posMbarOut[i], volMmk0, posMmkIn0, posMmkOut0,
          gld_tran_ref, gst_tran_ref, gld_req_ref, gst_req_ref,
          cl_full_l2_ref, cl_part_l2_ref, cl_full_l1, cl_part_l1);
      }
      if (gld_tran_tmp != gld_tran_ref || gst_tran_tmp != gst_tran_ref ||
        gld_req_tmp != gld_req_ref || gst_req_tmp != gst_req_ref) {
        printf("PackedSplit:countPackedGlTransactions0 ERROR\n");
        printf("tmp %d %d %d %d\n", gld_tran_tmp, gst_tran_tmp, gld_req_tmp, gst_req_tmp);
        printf("ref %d %d %d %d\n", gld_tran_ref, gst_tran_ref, gld_req_ref, gst_req_ref);
        return false;
      }
      if (cl_full_l2_tmp != cl_full_l2_ref || cl_part_l2_tmp != cl_part_l2_ref) {
        printf("PackedSplit:countPackedGlTransactions0 ERROR\n");
        printf("tmp %d %d\n", cl_full_l2_tmp, cl_part_l2_tmp);
        printf("ref %d %d\n",  cl_full_l2_ref, cl_part_l2_ref);
        return false;
      }
#endif
    }

#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
    gpuRangeStart("PackedSplit: shared");
#endif

    // Shared memory
    sld_tran = 0;
    sst_tran = 0;
    sld_req = 0;
    sst_req = 0;
    // Round down splits
    countPackedShTransactions0(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
      volMmk0, hostMsh.data(), tensorSplit.sizeMmk,
      sld_tran, sst_tran, sld_req, sst_req);
#ifdef COUNTCYCLE_CHECK
    {
      int sld_tran_ref = 0;
      int sst_tran_ref = 0;
      int sld_req_ref = 0;
      int sst_req_ref = 0;
      countPackedShTransactionsRef(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
        volMmk0, hostMsh.data(), tensorSplit.sizeMmk,
        sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
      if (sld_tran != sld_tran_ref || sst_tran != sst_tran_ref ||
        sld_req != sld_req_ref || sst_req != sst_req_ref) {
        printf("PackedSplit:countPackedShTransactions0 fails\n");
        printf("    %d %d %d %d\n", sld_tran, sst_tran, sld_req, sst_req);
        printf("ref %d %d %d %d\n", sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
        return false;
      }
    }
#endif
    sld_tran *= num0;
    sst_tran *= num0;
    sld_req *= num0;
    sst_req *= num0;

    // Round up splits
    if (num1 > 0) {
      int sld_tran_tmp = 0;
      int sst_tran_tmp = 0;
      int sld_req_tmp = 0;
      int sst_req_tmp = 0;
      countPackedShTransactions0(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
        volMmk1, hostMsh.data() + tensorSplit.sizeMmk, tensorSplit.sizeMmk,
        sld_tran_tmp, sst_tran_tmp, sld_req_tmp, sst_req_tmp);
#ifdef COUNTCYCLE_CHECK
      {
        int sld_tran_ref = 0;
        int sst_tran_ref = 0;
        int sld_req_ref = 0;
        int sst_req_ref = 0;
        countPackedShTransactionsRef(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
          volMmk1, hostMsh.data() + tensorSplit.sizeMmk, tensorSplit.sizeMmk,
          sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
        if (sld_tran_tmp != sld_tran_ref || sst_tran_tmp != sst_tran_ref ||
          sld_req_tmp != sld_req_ref || sst_req_tmp != sst_req_ref) {
          printf("PackedSplit:countPackedShTransactions0 fails\n");
          printf("    %d %d %d %d\n", sld_tran, sst_tran, sld_req, sst_req);
          printf("ref %d %d %d %d\n", sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
          return false;
        }
      }
#endif
      sld_tran += sld_tran_tmp*num1;
      sst_tran += sst_tran_tmp*num1;
      sld_req += sld_req_tmp*num1;
      sst_req += sst_req_tmp*num1;
    }
#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
#endif

  } else if (tensorSplit.method == Packed) {
#ifdef ENABLE_NVTOOLS
    gpuRangeStart("Packed: init");
#endif
    num_iter = tensorSplit.volMbar;
    // mlp = (float)launchConfig.numRegStorage;
    mlp = (float)(tensorSplit.volMmk) / (float)(launchConfig.numthread.x);
    // Global memory
    gld_tran = 0;
    gst_tran = 0;
    gld_req = 0;
    gst_req = 0;
    cl_full_l2 = 0;
    cl_part_l2 = 0;
    cl_full_l1 = 0;
    cl_part_l1 = 0;
    // Random number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, tensorSplit.volMbar - 1);
    // Pre-compute posMmkIn and posMmkOut
    std::vector<int> posMmkIn(tensorSplit.volMmk);
    std::vector<int> posMmkOut(tensorSplit.volMmk);

    computePos0(tensorSplit.volMmk, hostMmk.data(), tensorSplit.sizeMmk,
      posMmkIn.data(), posMmkOut.data());
    // computePos(0, tensorSplit.volMmk - 1, hostMmkFast.data(), tensorSplit.sizeMmk,
    //   posMmkIn.data(), posMmkOut.data());
#ifdef COUNTCYCLE_CHECK
    std::vector<int> posMmkInRef(tensorSplit.volMmk);
    std::vector<int> posMmkOutRef(tensorSplit.volMmk);
    computePosRef(0, tensorSplit.volMmk - 1, hostMmk.begin(), hostMmk.begin() + tensorSplit.sizeMmk,
      posMmkInRef, posMmkOutRef);
    for (int i=0;i < tensorSplit.volMmk;i++) {
      if (posMmkIn[i] != posMmkInRef[i] || posMmkOut[i] != posMmkOutRef[i]) {
        printf("computePos0 fails\n");
        printf("%d %d %d %d\n", posMmkIn[i], posMmkInRef[i], posMmkOut[i], posMmkOutRef[i]);
        return false;
      }
    }
#endif

    int num_ipos = (numPosMbarSample == 0) ? tensorSplit.volMbar : numPosMbarSample;

#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
    gpuRangeStart("Packed: loop");
#endif

    for (int iposMbar=0;iposMbar < num_ipos;iposMbar+=INT_VECTOR_LEN) {
      // int posMbar = (numPosMbarSample == 0) ? iposMbar : distribution(generator);
      int numPos = std::min(num_ipos - iposMbar, INT_VECTOR_LEN);
      int posMbar[INT_VECTOR_LEN];
      for (int i=0;i < numPos;i++) {
        posMbar[i] = (numPosMbarSample == 0) ? (iposMbar + i) : distribution(generator);
      }
      for (int i=numPos;i < INT_VECTOR_LEN;i++) {
        posMbar[i] = posMbar[numPos - 1];
      }

      int posMbarIn[INT_VECTOR_LEN];
      int posMbarOut[INT_VECTOR_LEN];
#ifdef ENABLE_NVTOOLS
      gpuRangeStart("computePos");
#endif
      for (int i=0;i < INT_VECTOR_LEN;i++) {
        computePos(posMbar[i], posMbar[i], hostMbar.data(), tensorSplit.sizeMbar, &posMbarIn[i], &posMbarOut[i]);
      }
      // computePosRef(posMbar, posMbar, hostMbar.begin(), hostMbar.begin() + tensorSplit.sizeMbar, posMbarInV, posMbarOutV);
      // int posMbarIn = posMbarInV[0];
      // int posMbarOut = posMbarOutV[0];

#ifdef ENABLE_NVTOOLS
      gpuRangeStop();
      gpuRangeStart("countPackedGlTransactions");
#endif

      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int gld_req_tmp = 0;
      int gst_req_tmp = 0;
      int cl_full_l2_tmp = 0;
      int cl_part_l2_tmp = 0;
      countPackedGlTransactions0(prop.warpSize, accWidth, cacheWidth, launchConfig.numthread.x,
        numPos, posMbarIn, posMbarOut, tensorSplit.volMmk, posMmkIn.data(), posMmkOut.data(),
        gld_tran_tmp, gst_tran_tmp, gld_req_tmp, gst_req_tmp,
        cl_full_l2_tmp, cl_part_l2_tmp, cl_full_l1, cl_part_l1);
      gld_tran += gld_tran_tmp;
      gst_tran += gst_tran_tmp;
      gld_req += gld_req_tmp;
      gst_req += gst_req_tmp;
      cl_full_l2 += cl_full_l2_tmp;
      cl_part_l2 += cl_part_l2_tmp;

#ifdef COUNTCYCLE_CHECK
      int gld_tran_ref = 0;
      int gst_tran_ref = 0;
      int gld_req_ref = 0;
      int gst_req_ref = 0;
      int cl_full_l2_ref = 0;
      int cl_part_l2_ref = 0;
      for (int i=0;i < numPos;i++) {
        countPackedGlTransactions(prop.warpSize, accWidth, cacheWidth, launchConfig.numthread.x,
          posMbarIn[i], posMbarOut[i], tensorSplit.volMmk, posMmkIn, posMmkOut,
          gld_tran_ref, gst_tran_ref, gld_req_ref, gst_req_ref,
          cl_full_l2_ref, cl_part_l2_ref, cl_full_l1, cl_part_l1);
      }
      if (gld_tran_tmp != gld_tran_ref || gst_tran_tmp != gst_tran_ref ||
        gld_req_tmp != gld_req_ref || gst_req_tmp != gst_req_ref) {
        printf("countPackedGlTransactions0 ERROR\n");
        printf("tmp %d %d %d %d\n", gld_tran_tmp, gst_tran_tmp, gld_req_tmp, gst_req_tmp);
        printf("ref %d %d %d %d\n", gld_tran_ref, gst_tran_ref, gld_req_ref, gst_req_ref);
        return false;
      }
      if (cl_full_l2_tmp != cl_full_l2_ref || cl_part_l2_tmp != cl_part_l2_ref) {
        printf("countPackedGlTransactions0 ERROR\n");
        printf("tmp %d %d\n", cl_full_l2_tmp, cl_part_l2_tmp);
        printf("ref %d %d\n",  cl_full_l2_ref, cl_part_l2_ref);
        return false;
      }
#endif

#ifdef ENABLE_NVTOOLS
      gpuRangeStop();
#endif
    }

#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
    gpuRangeStart("Packed: shared");
#endif

    // Shared memory
    sld_tran = 0;
    sst_tran = 0;
    sld_req = 0;
    sst_req = 0;
    countPackedShTransactions0(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
      tensorSplit.volMmk, hostMsh.data(), tensorSplit.sizeMmk,
      sld_tran, sst_tran, sld_req, sst_req);
#ifdef COUNTCYCLE_CHECK
    int sld_tran_ref = 0;
    int sst_tran_ref = 0;
    int sld_req_ref = 0;
    int sst_req_ref = 0;
    countPackedShTransactionsRef(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
      tensorSplit.volMmk, hostMsh.data(), tensorSplit.sizeMmk,
      sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
    if (sld_tran != sld_tran_ref || sst_tran != sst_tran_ref ||
      sld_req != sld_req_ref || sst_req != sst_req_ref) {
      printf("countPackedShTransactions0 fails\n");
      printf("    %d %d %d %d\n", sld_tran, sst_tran, sld_req, sst_req);
      printf("ref %d %d %d %d\n", sld_tran_ref, sst_tran_ref, sld_req_ref, sst_req_ref);
      return false;
    }
#endif
    // countPackedShTransactions(prop.warpSize, prop.warpSize, launchConfig.numthread.x, 
    //   tensorSplit.volMmk, hostMsh.data(), tensorSplit.sizeMmk,
    //   sld_tran, sst_tran, sld_req, sst_req);
#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
#endif
  } else if (tensorSplit.method == Trivial) {
    size_t vol = tensorSplit.volMmk*tensorSplit.volMbar;
    // Global memory
    gld_req = (vol - 1)/prop.warpSize + 1;
    gst_req = gld_req;
    gld_tran = (vol - 1)/accWidth + 1;
    gst_tran = gld_tran;
    cl_full_l2 = vol/cacheWidth;
    cl_part_l2 = ((vol % cacheWidth) > 0);
    // Shared memory
    sld_tran = 0;
    sst_tran = 0;
    sld_req = 0;
    sst_req = 0;
    // Cycles
    cycles = 0.0;
    return true;
  } else {
    return false;
  }


  int numthread = launchConfig.numthread.x*launchConfig.numthread.y*launchConfig.numthread.z;

  if (tensorSplit.method == Packed || tensorSplit.method == PackedSplit) {
    cycles = cyclesPacked(tensorSplit.method == PackedSplit, sizeofType, prop, numthread,
      numActiveBlock, launchConfig.numRegStorage, 
      gld_req, gst_req, gld_tran, gst_tran, sld_req, sst_req, sld_tran, sst_tran,
      num_iter, cl_full_l2, cl_part_l2);
  } else if (tensorSplit.method == Tiled || tensorSplit.method == TiledCopy) {
    cycles = cyclesTiled(tensorSplit.method == TiledCopy, sizeofType, prop, numthread,
      numActiveBlock, mlp, gld_req, gst_req, gld_tran, gst_tran,
      sld_req, sst_req, sld_tran, sst_tran,
      num_iter, cl_full_l2, cl_part_l2);
  }

  return true;
}

//
// Activates the plan: Allocates device memory buffers and copies data to them
//
void cuttPlan_t::activate() {

  if (tensorSplit.sizeMbar > 0) {
    if (Mbar == NULL) {
      allocate_device<TensorConvInOut>(&Mbar, tensorSplit.sizeMbar);
      copy_HtoD<TensorConvInOut>(hostMbar.data(), Mbar, tensorSplit.sizeMbar, stream);
    }
  }

  if (tensorSplit.method == Packed || tensorSplit.method == PackedSplit) {
    int MmkSize = (tensorSplit.method == Packed) ? tensorSplit.sizeMmk : tensorSplit.sizeMmk*2;
    if (Mmk == NULL) {
      allocate_device<TensorConvInOut>(&Mmk, MmkSize);
      copy_HtoD<TensorConvInOut>(hostMmk.data(), Mmk, MmkSize, stream);
    }
    if (Msh == NULL) {
      allocate_device<TensorConv>(&Msh, MmkSize);
      copy_HtoD<TensorConv>(hostMsh.data(), Msh, MmkSize, stream);
    }
  }

}

//
// Set device buffers to NULL
//
void cuttPlan_t::nullDevicePointers() {
  Mbar = NULL;
  Mmk = NULL;
  Msh = NULL;
  Mk = NULL;
  Mm = NULL;
}

cuttPlan_t::cuttPlan_t() {
  CUDA_CALL(cudaGetDevice(&deviceID));
  stream = 0;
  numActiveBlock = 0;
  nullDevicePointers();
}

cuttPlan_t::~cuttPlan_t() {
  // Deallocate device buffers
  if (Mbar != NULL) deallocate_device<TensorConvInOut>(&Mbar);
  if (Mmk != NULL) deallocate_device<TensorConvInOut>(&Mmk);
  if (Msh != NULL) deallocate_device<TensorConv>(&Msh);
  if (Mk != NULL) deallocate_device<TensorConv>(&Mk);
  if (Mm != NULL) deallocate_device<TensorConv>(&Mm);
}

void cuttPlan_t::setStream(cudaStream_t stream_in) {
  stream = stream_in;
}
