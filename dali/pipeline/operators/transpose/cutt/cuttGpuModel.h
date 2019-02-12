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

#ifndef DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUTTGPUMODEL_H
#define DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUTTGPUMODEL_H

#include <vector>
#include "dali/pipeline/operators/transpose/cutt/cuttTypes.h"
#include "dali/pipeline/operators/transpose/cutt/cuttplan.h"
#include "dali/pipeline/operators/transpose/cutt/int_vector.h"

void computePos(const int vol0, const int vol1,
  const TensorConvInOut* conv, const int numConv,
  int* posIn, int* posOut);

void computePos0(const int vol,
  const TensorConvInOut* conv, const int numConv,
  int* posIn, int* posOut);

void computePosRef(int vol0, int vol1,
  std::vector<TensorConvInOut>::iterator it0, std::vector<TensorConvInOut>::iterator it1,
  std::vector<int>& posIn, std::vector<int>& posOut);

void countPackedGlTransactions(const int warpSize, const int accWidth, const int cacheWidth,
  const int numthread, const int posMbarIn, const int posMbarOut, const int volMmk, 
  std::vector<int>& posMmkIn, std::vector<int>& posMmkOut,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1);

void countPackedGlTransactions0(const int warpSize, const int accWidth, const int cacheWidth,
  const int numthread, 
  const int numPos, const int posMbarIn[INT_VECTOR_LEN], const int posMbarOut[INT_VECTOR_LEN],
  const int volMmk,  const int* __restrict__ posMmkIn, const int* __restrict__ posMmkOut,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1);

void countPackedShTransactions(const int warpSize, const int bankWidth, const int numthread,
  const int volMmk, const TensorConv* msh, const int numMsh,
  int& sld_tran, int& sst_tran, int& sld_req, int& sst_req);

void countPackedShTransactions0(const int warpSize, const int bankWidth, const int numthread,
  const int volMmk, const TensorConv* msh, const int numMsh,
  int& sld_tran, int& sst_tran, int& sld_req, int& sst_req);

void countPackedShTransactionsRef(const int warpSize, const int bankWidth, const int numthread,
  const int volMmk, const TensorConv* msh, const int numMsh,
  int& sld_tran, int& sst_tran, int& sld_req, int& sst_req);

void countTiledGlTransactions(const bool leadVolSame,
  const int numPosMbarSample, const int volMm, const int volMk, const int volMbar,
  const int cIn, const int cOut, const int accWidth, const int cacheWidth,
  std::vector<TensorConvInOut>& hostMbar, const int sizeMbar,
  int& num_iter, float& mlp, int& gld_tran, int& gst_tran, int& gld_req, int& gst_req, int& cl_full, int& cl_part);

double cyclesPacked(const bool isSplit, const size_t sizeofType, cudaDeviceProp& prop,
  int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, int cl_full, int cl_part);

double cyclesTiled(const bool isCopy, const size_t sizeofType, cudaDeviceProp& prop,
  int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, int cl_full, int cl_part);

bool testCounters(const int warpSize, const int accWidth, const int cacheWidth);

#endif // DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_CUTTGPUMODEL_H