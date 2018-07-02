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


#include "dali/pipeline/operators/resize/new_resize.h"
#include <float.h>
#include <assert.h>
#include <string>
#include <vector>
#include <algorithm>

namespace dali {

//  Greatest Common Factor
int gcf(int a, int b) {
  int t;
  if (b > a) {
    t = a;
    a = b;
    b = t;
  }

  while (b) {
    t = a % b;
    a = b;
    b = t;
  }

  return a;
}

// Least Common Multiplier
int lcm(int a, int b) {
  return a / gcf(a, b) * b;
}

void DataDependentSetupCPU(const Tensor<CPUBackend> &input,
                           Tensor<CPUBackend> *output, const char *pOpName,
                           const uint8 **ppInRaster, uint8 **ppOutRaster,
                           vector<DALISize> *pSizes, const DALISize *out_size) {
  DALI_ENFORCE(input.ndim() == 3);
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expects input data in uint8.");

  const vector<Index> &shape = input.shape();
  const int C = shape[2];
  DALI_ENFORCE(C == 1 || C == 3,
               string(pOpName ? pOpName : "Operation") +
               " supports only hwc rgb & grayscale inputs.");

  if (out_size)
    output->Resize({out_size->height, out_size->width, C});
  else
    output->Resize(shape);

  output->set_type(input.type());

  if (!ppInRaster)
    return;

  *ppInRaster = input.template data<uint8>();
  if (ppOutRaster)
    *ppOutRaster = static_cast<uint8 *>(output->raw_mutable_data());

  if (pSizes) {
    (*pSizes)[0].height = shape[0];
    (*pSizes)[0].width = shape[1];
  }
}

bool DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> *output,
                           size_t batch_size, bool reshapeBatch, vector<const uint8 *> *inPtrs,
                           vector<uint8 *> *outPtrs, vector<DALISize> *pSizes,
                           ResizeParamDescr *pResizeDescr) {
  DALI_ENFORCE(IsType<uint8>(input.type()),
               "Expected input data stored in uint8.");

  auto pResize = pResizeDescr ? pResizeDescr->pResize_ : NULL;
  auto pResizeParam = pResizeDescr ? pResizeDescr->pResizeParam_ : NULL;
  auto pMirroring = pResizeDescr ? pResizeDescr->pMirroring_ : NULL;
  auto pTotalSize = pResizeDescr ? pResizeDescr->pTotalSize_ : NULL;

  // Set all elements to 0, if we will use them
  if (pTotalSize)
    memset(pTotalSize, 0, pResizeDescr->nBatchSlice_ * sizeof(pTotalSize[0]));

  bool newResize = false;
  vector<Dims> output_shape(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    // Verify the inputs
    const auto &input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
                 "Expects 3-dimensional image input.");

    DALI_ENFORCE(input_shape[2] == 1 || input_shape[2] == 3,
                 "Not valid color type argument (1 or 3)");

    // Collect the output shapes
    if (pResize) {
      // We are resizing
      const auto input_size = pResize->size(input_t, i);
      const auto out_size = pResize->size(output_t, i);
      pResize->SetSize(input_size, input_shape, pResize->newSizes(i), out_size);

      if (pResizeParam) {
        // NewResize is used
        const int H0 = input_size->height;
        const int W0 = input_size->width;
        const int H1 = out_size->height;
        const int W1 = out_size->width;

        int cropY, cropX;
        const bool doingCrop = pResize->CropNeeded(*out_size);
        if (doingCrop)
          pResize->DefineCrop(out_size, &cropX, &cropY);
        else
          cropY = cropX = 0;

        auto resizeParam = pResizeParam + i * (pMirroring ? N_GRID_PARAMS : 1);
        if (pMirroring) {
          const int lcmH = lcm(H0, H1);
          const int lcmW = lcm(W0, W1);

          const int sy0 = lcmH / H0;
          const int sy1 = lcmH / H1;
          const int sx0 = lcmW / W0;
          const int sx1 = lcmW / W1;

          if (!newResize) {
            newResize = resizeParam[0].x != sx0 || resizeParam[0].y != sy0 ||
                        resizeParam[1].x != sx1 || resizeParam[1].y != sy1 ||
                        resizeParam[2].x != cropX || resizeParam[2].y != cropY;
          }

          if (newResize) {
            resizeParam[0] = {sx0, sy0};
            resizeParam[1] = {sx1, sy1};
            resizeParam[2] = {cropX, cropY};
          }

          if (pTotalSize) {
            // We need to check for overflow
            const size_t idx = i % pResizeDescr->nBatchSlice_;
            if (pTotalSize[idx] < UINT_MAX - sx0 * sy0)
              pTotalSize[idx] += sx0 * sy0;
            else
              pTotalSize[idx] = UINT_MAX;
          }

          if (pMirroring)
            pResize->MirrorNeeded(pMirroring + i);
        } else {
          resizeParam[0] = {W1, H1};
        }
      }

      // Collect the output shapes
      output_shape[i] = {out_size->height, out_size->width, input_shape[2]};
    } else {
      output_shape[i] = input_shape;
    }

    if (pSizes) {
      (*pSizes)[i].height = input_shape[0];
      (*pSizes)[i].width = input_shape[1];
      if (reshapeBatch) {
        // When batch is reshaped: only one "image" will be used
        (*pSizes)[i].height *= batch_size;
        pSizes = NULL;
      }
    }
  }

  // Resize the output
  output->Resize(output_shape);
  output->set_type(input.type());

  CollectPointersForExecution(reshapeBatch ? 1 : batch_size, input, inPtrs, output, outPtrs);
  return newResize;
}

void CollectPointersForExecution(size_t batch_size,
                                 const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
                                 TensorList<GPUBackend> *output, vector<uint8 *> *outPtrs) {
  if (!inPtrs || !outPtrs)
    return;

  // Collect the pointers for execution
  for (size_t i = 0; i < batch_size; ++i) {
    (*inPtrs)[i] = input.template tensor<uint8>(i);
    (*outPtrs)[i] = output->template mutable_tensor<uint8>(i);
  }
}

typedef void (*allocMemoryFunction)(ResizeMappingPixDescrCPU *pntr, size_t nElem);

typedef void (*assignElemFunction)(ResizeMappingPixDescrCPU *pntr, size_t nElem,
                                   uint32_t addr, uint32_t area);

static void resizeVector(ResizeMappingPixDescrCPU *pntr, size_t nElem) {
  pntr->resize(nElem);
}

static void assignVectorElement(ResizeMappingPixDescrCPU *pntr, size_t elemIdx,
                                uint32_t addr, uint32_t area) {
  PixMapping &elem = PIX_MAPPING_CPU(*pntr)[elemIdx];
  elem.pixAddr = addr;
  elem.pixArea = area;
}

class PixMappingHelper {
 public:
  CC PixMappingHelper(uint32_t len, ResizeMapping *pMapping, MappingInfo *pMapInfo,
                      uint32_t resizedArea, ResizeMappingPixDescrCPU *pPixMapping = NULL,
                      allocMemoryFunction allocMemFunc = NULL,
                      assignElemFunction assignFunc = NULL);

  void CC constructTable(int C, int W0, size_t sx0, size_t sy0, size_t sx1, size_t sy1,
                         int stepW = 1, int stepH = 1, int startW = 0, int startH = 0);

  inline uint32_t CC numUsed() const { return numPixMapUsed_; }

 private:
  void CC AddPixel(uint32_t addr, uint32_t area, int crdX, int crdY);

  void CC UpdateMapping(int shift, int centerX, int centerY);

  inline float CC distance(float x, float y) const { return x * x + y * y; }

  uint32_t numPixMapMax_;     // length of the allocated PixMapping array
  uint32_t numPixMapUsed_;    // number of already used elements of pPixMapping
  ResizeMappingPixDescrCPU *pPixMapping_;
  ResizeMapping *const pMappingBase_;
  ResizeMapping *pMapping_;
  MappingInfo *const pMappingClosestBase_;
  MappingInfo *pMappingClosest_;

  const allocMemoryFunction allocMemFunc_;
  const assignElemFunction assignFunc_;

  const uint32_t area_;
  const uint32_t resizedArea_;
  float closestDist_;
  float centerX_, centerY_;
};

__global__ void InitiateResizeTables(int nTable, const ResizeGridParam *resizeDescr,
                      MappingInfo *mapPntr[], MappingInfo **mappingMem, size_t step) {
  size_t idx = blockIdx.x;
  mapPntr[idx] = mappingMem[idx];
  if (mapPntr[idx]) {
    for (size_t i = idx + step; i < nTable; idx = i, i += step)
      mapPntr[i] = mapPntr[idx] +
                   resizeDescr[idx * N_GRID_PARAMS].x * resizeDescr[idx * N_GRID_PARAMS].y;
  } else {
    // No resize tables for that batch slice will be constructed
    for (size_t i = idx + step; i < nTable; i += step)
      mapPntr[i] = NULL;
  }
}

__global__ void ConstructResizeTables(int C, const ResizeGridParam *resizeDescr,
                       const DALISize *in_sizes, int W0, MappingInfo *pResizeMapping[]) {
  const int imagIdx = blockIdx.x;
  auto resizeParam = resizeDescr + N_GRID_PARAMS * imagIdx;
  if (in_sizes)
    W0 = in_sizes[imagIdx].width;

  SET_RESIZE_PARAM();

  PixMappingHelper helper(sx0 * sy0, NULL, pResizeMapping[imagIdx], area);
  helper.constructTable(C, W0, sx0, sy0, sx1, sy1,
                        blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
}

#define RESIZE_GPU_PREAMBLE()         // empty macro
#define RESIZE_GPU_CORE(C)            RESIZE_CORE(C)

#define RESIZE_GPU_N_PREAMBLE()       // empty macro
#define RESIZE_GPU_N_CORE(C)          RESIZE_N_CORE(C)

__global__ void BatchedCongenericResizeKernel(
  int H0, int W0, const uint8 *img_in, int H, int W, uint8 *img_out, int C,
  const ResizeGridParam *resizeParam, const MirroringInfo *pMirrorInfo,
  MappingInfo *const ppMapping[], const ResizeMapping *pResizeMapping,
  const PixMapping *pPixMapping) {
  const int imagIdx = blockIdx.x;
  const bool mirrorHor = pMirrorInfo && (pMirrorInfo + imagIdx)->x != 0;
  const bool mirrorVert = pMirrorInfo && (pMirrorInfo + imagIdx)->y != 0;
  RESIZE_PREPARE();
  if (ppMapping || pResizeMapping) {
    const MappingInfo *pMapping = ppMapping ? ppMapping[0] : NULL;
    AUGMENT_RESIZE_GPU_CONGENERIC(H, W, C, img_in, img_out, RESIZE_GPU_N);
  } else {
    AUGMENT_RESIZE_GPU_CONGENERIC(H, W, C, img_in, img_out, RESIZE_GPU);
  }
}

DALIError_t BatchedCongenericResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                     const DALISize &sizeIn, const uint8 *in_batch, const DALISize &sizeOut,
                     uint8 *out_batch,
                     const ResizeGridParam *resizeDescr, const MirroringInfo *pMirrorInfo,
                     MappingInfo *ppMapping[], MappingInfo **mapMem,
                     const ResizeMapping *pResizeMapping, const PixMapping *pPixMapping,
                     bool newMapping) {
  if (ppMapping && newMapping) {
    CHECK_RESIZE_DESCR();

    InitiateResizeTables << < 1, 1, 0, stream >> >
         (1, resizeDescr, ppMapping, mapMem, 1);

    ConstructResizeTables << < 1, gridDim, 0, stream >> >
         (C, resizeDescr, NULL, sizeIn.width, ppMapping);
  }

  BatchedCongenericResizeKernel << < N, gridDim, 0, stream >> >
         (sizeIn.height, sizeIn.width, in_batch, sizeOut.height, sizeOut.width, out_batch, C,
          resizeDescr, pMirrorInfo, ppMapping, pResizeMapping, pPixMapping);

  return DALISuccess;
}

__global__ void BatchedResizeKernel(int C, const ResizeGridParam *resizeDescr,
                       MappingInfo *const ppMapping[], const MirroringInfo *pMirrorInfo,
                       const DALISize *in_sizes, const uint8 *const imgs_in[],
                       const DALISize *out_sizes, uint8 *const imgs_out[]) {
  const int imagIdx = blockIdx.x;
  auto resizeParam = resizeDescr + N_GRID_PARAMS * imagIdx;
  const int W0 = in_sizes[imagIdx].width;
  const int H0 = in_sizes[imagIdx].height;
  const int W = out_sizes[imagIdx].width;
  const int H = out_sizes[imagIdx].height;
  const bool mirrorHor = pMirrorInfo && (pMirrorInfo + imagIdx)->x != 0;
  const bool mirrorVert = pMirrorInfo && (pMirrorInfo + imagIdx)->y != 0;

  RESIZE_PREPARE();
  if (ppMapping) {
    const ResizeMapping *pResizeMapping = NULL;
    const PixMapping *pPixMapping = NULL;
    const MappingInfo *pMapping = ppMapping[imagIdx];
    AUGMENT_RESIZE_GPU_GENERIC(H, W, C, imgs_in[imagIdx], imgs_out[imagIdx], RESIZE_GPU_N);
  } else {
    AUGMENT_RESIZE_GPU_GENERIC(H, W, C, imgs_in[imagIdx], imgs_out[imagIdx], RESIZE_GPU);
  }
}

DALIError_t BatchedResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const ResizeGridParam *resizeDescr, const ImgSizeDescr sizes[],
                          const ImgRasterDescr raster[], MappingInfo *ppMapping[],
                          MappingInfo **mapMem, size_t nBatchSlice) {
  auto in_sizes = IMG_SIZES(sizes[input_t]);
  auto out_sizes = IMG_SIZES(sizes[output_t]);
  if (ppMapping) {
    InitiateResizeTables << < nBatchSlice, 1, 0, stream >> >
             (N, resizeDescr, ppMapping, mapMem, nBatchSlice);

    ConstructResizeTables << < N, gridDim, 0, stream >> >
             (C, resizeDescr, in_sizes, 0, ppMapping);
  }

  const uint8 *const *in = IMG_RASTERS(raster[input_t]);
  uint8 *const *out = IMG_RASTERS(raster[output_t]);

  const MirroringInfo *pMirrorInfo = resizeDescr + N_GRID_PARAMS * N;
  BatchedResizeKernel << < N, gridDim, 0, stream >> >
              (C, resizeDescr, ppMapping, pMirrorInfo, in_sizes, in, out_sizes, out);

  return DALISuccess;
}

PixMappingHelper::PixMappingHelper(uint32_t area, ResizeMapping *pMapping, MappingInfo *pMapInfo,
                         uint32_t resizedArea, ResizeMappingPixDescrCPU *pPixMapping,
                         allocMemoryFunction allocMemFunc, assignElemFunction assignFunc) :
                         area_(area), resizedArea_(resizedArea), allocMemFunc_(allocMemFunc),
                         assignFunc_(assignFunc), pMappingBase_(pMapping),
                         pMappingClosestBase_(pMapInfo), closestDist_(FLT_MAX) {
  numPixMapMax_ = 1;
  numPixMapUsed_ = 0;

  if (resizedArea == 0 && allocMemFunc_)
    (*allocMemFunc_)(pPixMapping_ = pPixMapping, numPixMapMax_ = 2 * area);
  else
    pPixMapping_ = NULL;
}

void PixMappingHelper::AddPixel(uint32_t addr, uint32_t area, int crdX, int crdY) {
  assert(area != 0);
  if (pPixMapping_) {
    if (numPixMapUsed_ == numPixMapMax_) {
      // Previously allocated array needs to be extended
      (*allocMemFunc_)(pPixMapping_, numPixMapMax_ <<= 1);
    }

    pMapping_->nPixels++;
    (*assignFunc_)(pPixMapping_, numPixMapUsed_++, addr, area);
  } else {
    const float newDist = distance((crdX << 1) - centerX_, (crdY << 1) - centerY_);
    if (closestDist_ > newDist) {
      closestDist_ = newDist;
      *pMappingClosest_ = addr;
    }
  }
}

void PixMappingHelper::UpdateMapping(int shift, int centerX, int centerY) {
  if (pPixMapping_) {
    (pMapping_ = pMappingBase_ + shift)->intersectInfoAddr = numUsed();
  } else {
    pMappingClosest_ = pMappingClosestBase_ + shift;
    centerX_ = centerX;
    centerY_ = centerY;
    closestDist_ = FLT_MAX;
  }
}

#define RUN_CHECK_1     0

void PixMappingHelper::constructTable(int C, int W0, size_t sx0, size_t sy0, size_t sx1,
                           size_t sy1, int stepW, int stepH, int startW, int startH) {
  // (x, y) pixel coordinate of PIX in resized image
  // 0 <= x < W1;  0 <= y < H1

  for (size_t y = startH; y < sy0; y += stepH) {
    for (size_t x = startW; x < sx0; x += stepW) {
      const size_t nX = x * sx1;
      const size_t nY = y * sy1;
      // The indices of the top-left pixel of the initial image, intersecting with PIX
      const size_t begIdx[2] = {nX / sx0, nY / sy0};

      // The indices of the bottom-right pixel of the initial image, intersecting with PIX
      size_t endIdx[2] = {(nX + sx1) / sx0, (nY + sy1) / sy0};

      // Intersection of the right (bottom) pixels with the PIX (could be equal to 0)
      const size_t extra[2] = {min((nX + sx1) % sx0, sx1), min((nY + sy1) % sy0, sy1)};

      // Length of the left (top) pixels intersecting with the PIX
      const size_t lenFirst[2] = {(sx0 - nX % sx0), (sy0 - nY % sy0)};

      // Doubled (x,y) coordinates of the pixel's center
      const size_t lenX = endIdx[0] + begIdx[0] - (extra[0] || endIdx[0] == begIdx[0]? 0 : 1);
      const size_t lenY = endIdx[1] + begIdx[1] - (extra[1] || endIdx[1] == begIdx[1]? 0 : 1);

      // Relative address to the first intersecting pixels
      UpdateMapping(((y * sy1) % sy0) * sx0 + (x * sx1) % sx0, lenX, lenY);

      endIdx[0] -= begIdx[0];
      endIdx[1] -= begIdx[1];
#if RUN_CHECK_1
      size_t check = 0;
#endif
      size_t rowMult = endIdx[1]? lenFirst[1] : extra[1];
      size_t y0 = 0;
      while (true) {
        size_t x0 = endIdx[0];

        // Relative address of the last pixel in row y0, intersecting with PIX
        uint32_t pixAddr = ((y0 * W0) + x0) * C;
        if (extra[0])
          AddPixel(pixAddr, extra[0] * rowMult, x0, y0);

        if (x0) {
          while (--x0 > 0)
            AddPixel(pixAddr -= C, sx0 * rowMult, x0, y0);

          AddPixel(pixAddr -= C, lenFirst[0] * rowMult, x0, y0);
        }

#if RUN_CHECK_1
        check += rowMult * ((endIdx[0]? sx0 * (endIdx[0] - 1) + lenFirst[0] : 0) + extra[0]);
#endif
        if (++y0 >= endIdx[1]) {
          if (y0 > endIdx[1] || !(rowMult = extra[1]))
            break;
        } else {
          rowMult = sy0;
        }
      }

#if RUN_CHECK_1
      assert(check == sx1 * sy1);
#endif
    }
  }
}

void ResizeMappingTable::initTable(int H0, int W0, int H1, int W1, int C,
                                   uint16_t xSize, uint16_t ySize, bool use_NN) {
  io_size[0] = {W0, H0};
  io_size[1] = {W1, H1};
  C_ = C;

  if (use_NN)
    resizeMappingSimpleCPU.resize({xSize * ySize});
  else
    resizeMappingCPU.resize({xSize * ySize});
}

void ResizeMappingTable::constructTable(int H0, int W0, int H1, int W1, int C, int resizeType) {
  // The table, which contains the information about correspondence of pixels of the initial
  // image to the pixels of the resized one.

  // Resizing from (H0, W0) to (H1, W1)
  // Main equations are:
  // H0 * sy0 = H1 * sy1
  // W0 * sx0 = W1 * sx1
  const size_t lcmH = lcm(H0, H1);
  const size_t lcmW = lcm(W0, W1);

  const size_t sy0 = lcmH / H0;
  const size_t sy1 = lcmH / H1;
  const size_t sx0 = lcmW / W0;
  const size_t sx1 = lcmW / W1;

  const bool use_NN = resizeType == DALI_INTERP_NN;
  initTable(H0, W0, H1, W1, C, sx0, sy0, use_NN);

  PixMappingHelper helper(sx0 * sy0, RESIZE_MAPPING_CPU(resizeMappingCPU),
                          RESIZE_MAPPING_CPU(resizeMappingSimpleCPU), use_NN ? sx1 * sy1 : 0,
                          &pixMappingCPU, resizeVector, assignVectorElement);

  helper.constructTable(C, W0, sx0, sy0, sx1, sy1);
  if (!use_NN)
    pixMappingCPU.resize(helper.numUsed());
}

template <>
void NewResize<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  const auto &output = ws->Output<GPUBackend>(idx);
  const bool use_NN = type_ == DALI_INTERP_NN;

  size_t resizeMemory[BATCH_SLICE_NUMB];
  ResizeGridParam *pResizeGrid = resizeParam_.data();
  MirroringInfo *pMirror = pResizeGrid + N_GRID_PARAMS * batch_size_;
  ResizeParamDescr resizeDescr(this, pResizeGrid, pMirror,
                             use_NN ? resizeMemory : NULL, BATCH_SLICE_NUMB);

  const bool newMapping = DataDependentSetupGPU(input, output, batch_size_, false,
                             inputImages(), outputImages(), NULL,
                             &resizeDescr);

  const int C = input.shape()[0][2];

  const auto sizeIn = size(input_t, 0);
  const auto sizeOut = size(output_t, 0);
  cudaStream_t s = ws->stream();

  const bool congenericBatch = BatchIsCongeneric(sizeIn, sizeOut, C);
  MappingInfo **mapPntr = NULL;
  if (use_NN) {
    if (newMapping) {
        if (congenericBatch)
            mapPntr = CopyResizeTableToGPU(resizeMemory, s);
        else
            mapPntr = CopyResizeTableToGPU(resizeMemory, s, batch_size_, BATCH_SLICE_NUMB);
    } else {
        mapPntr = mappingPntr_;
    }
  }

  if (congenericBatch) {
    if (newMapping) {
      // Copying the descriptor of operation into GPU
      resizeParamGPU_.Copy(vector<ResizeGridParam>(
              resizeParam_.begin(), resizeParam_.begin() + N_GRID_PARAMS), s);
    }

    mirrorParamGPU_.Copy(vector<ResizeGridParam>(
          resizeParam_.begin() + N_GRID_PARAMS * batch_size_, resizeParam_.end()), s);

    const ResizeMapping *pResizeMapping = NULL;
    const PixMapping *pPixMapping = NULL;
#if USE_RESIZE_TABLE_GPU
    if (!use_NN) {
      if (newMapping || !resizeTbl_.IsValid(sizeIn, sizeOut, C)) {
          resizeTbl_.constructTable(sizeIn.height, sizeIn.width,
                          sizeOut.height, sizeOut.width, C, type_);
          resizeTbl_.copyToGPU(s);
      }

      pResizeMapping = RESIZE_MAPPING_GPU(resizeTbl_.resizeMappingGPU);
      pPixMapping = PIX_MAPPING_GPU(resizeTbl_.pPixMappingGPU);
    }
#endif

  DALI_CALL(BatchedCongenericResize(batch_size_, dim3(32, 32), s, C,
                *sizeIn, input.template data<uint8>(),
                *sizeOut, static_cast<uint8 *>(output->raw_mutable_data()),
                RESIZE_PARAM(resizeParamGPU_), MIRRORING_PARAM(mirrorParamGPU_),
                mapPntr, mapMemGPU_, pResizeMapping, pPixMapping, newMapping));
  } else {
    resizeParamGPU_.Copy(resizeParam_, s);

    vector<uint8 *> *raster[] = {(vector<uint8 *> *)(inputImages()), outputImages()};

    for (int i = input_t; i <= output_t; i++) {
        TENSOR_COPY(sizesGPU_[i], sizes(static_cast<io_type >(i)), s);
        TENSOR_COPY(imgsGPU_[i], *(raster[i]), s);
    }

    DALI_CALL(BatchedResize(batch_size_, dim3(32, 32), s, C, RESIZE_PARAM(resizeParamGPU_),
                            sizesGPU_, imgsGPU_, mapPntr, mapMemGPU_, _countof(mapMem_)));
  }
}

template<>
void NewResize<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace* ws) {
  Resize::SetupSharedSampleParams(ws);
}

}  // namespace dali

