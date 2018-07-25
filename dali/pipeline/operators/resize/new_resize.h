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


#ifndef DALI_PIPELINE_OPERATORS_RESIZE_NEW_RESIZE_H_
#define DALI_PIPELINE_OPERATORS_RESIZE_NEW_RESIZE_H_

#include <npp.h>
#include <random>
#include <ctgmath>
#include <vector>
#include <algorithm>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/resize/resize.h"

namespace dali {

#define BATCH_SLICE_NUMB            32      // The number of slices composing the whole batch
#define USE_RESIZE_TABLE_GPU         0      // Use (1) or not (0) the resize tables on GPU
                                            // NOTE: As of 04/13/2018, the performance with
                                            // these tables is better on CPU, but not on GPU

#define GPU_BACKEND_PARAMS(x, type)         (x).template data<type>()
#define IMG_SIZES(x)                        GPU_BACKEND_PARAMS(x, DALISize)
#define IMG_RASTERS(x)                      GPU_BACKEND_PARAMS(x, uint8 *)
#define PIX_MAPPING_GPU(x)                  GPU_BACKEND_PARAMS(x, PixMapping)
#define RESIZE_MAPPING_GPU(x)               GPU_BACKEND_PARAMS(x, ResizeMapping)
#define RESIZE_PARAM(x)                     GPU_BACKEND_PARAMS(x, ResizeGridParam)
#define MIRRORING_PARAM(x)                  GPU_BACKEND_PARAMS(x, MirroringInfo)

#define TENSOR_COPY(x, y, s)                (x).Copy(y, s)

#define PIX_MAPPING_CPU(x)                  (x).data()
#define RESIZE_MAPPING_CPU(x)               (x).data()

#define N_GRID_PARAMS       3
#define _countof(x)         sizeof(x)/sizeof(x[0])

#define CUDA_CALLABLE __host__ __device__    // Calling from CUDA

typedef struct {
  uint16_t nPixels;               // number of the pixels, intersecting with the resulting pixel
  uint32_t intersectInfoAddr;     // address to the information for first intersecting pixel
} ResizeMapping;

typedef struct {
  uint32_t pixAddr;               // relative address to the pixels of the initial image
  uint32_t pixArea;               // area of its intersection with the resulting pixels
} PixMapping;

typedef NppiPoint ResizeGridParam;
typedef uint32_t MappingInfo;

typedef Tensor<GPUBackend> ImgSizeDescr, ImgRasterDescr, ResizeMappingPixDescrGPU, MirroringDescr,
            ResizeMappingTableGPU, ResizeGridDescr, ResizeMappingPntrGPU, ResizeMappingGPU;
typedef vector<ResizeMapping> ResizeMappingTableCPU;
typedef vector<MappingInfo> ResizeMappingCPU;
typedef vector<PixMapping> ResizeMappingPixDescrCPU;


#define nYoffset(W, C)          ((W) * (C))

#define SAME_SIZES(size1, size2)    ((size1)->height == (size2)->height && \
                                     (size1)->width == (size2)->width)
class ResizeMappingTable {
 public:
  DALISize io_size[2];
  int C_;
                                              // Pointers to:
  ResizeMappingTableCPU resizeMappingCPU;     //      ResizeMapping table on CPU
  ResizeMappingPixDescrCPU pixMappingCPU;     //      PixMapping arrays for CPU
  ResizeMappingCPU resizeMappingSimpleCPU;    //      simplified ResizeMapping table on CPU

#if USE_RESIZE_TABLE_GPU
  ResizeMappingTableGPU resizeMappingGPU;     //      ResizeMapping table on GPU
  ResizeMappingPixDescrGPU pPixMappingGPU;    //      PixMapping arrays for GPU

  ResizeMappingTable() {
    resizeMappingGPU.mutable_data<ResizeMapping>();
    pPixMappingGPU.mutable_data<PixMapping>();
  }

  bool IsValid(const DALISize &in, const DALISize &out, int C) const {
    if (!RESIZE_MAPPING_CPU(resizeMappingCPU) || C_ != C)
      return false;

    return SAME_SIZES(io_size, &in) && SAME_SIZES(io_size+1, &out);
  }

  inline void copyToGPU(cudaStream_t s) {
    TENSOR_COPY(resizeMappingGPU, resizeMappingCPU, s);
    TENSOR_COPY(pPixMappingGPU, pixMappingCPU, s);
  }
#endif

  void constructTable(int H0, int W0, int H1, int W1, int C, int resizeType);

 private:
  void initTable(int H0, int W0, int H1, int W1, int C,
                 uint16_t xSize, uint16_t ySize, bool use_NN);
};

CUDA_CALLABLE void ResizeFunc(int W0, int H0, const uint8 *img_in, int W, int H, uint8 *img_out,
                   int C, const ResizeGridParam *resizeParam, const MirroringInfo *pMirrorInfo,
                   int imgIdx = 0, int startW = 0, int stepW = 1, int startH = 0, int stepH = 1,
                   const MappingInfo *pMapping = NULL, const ResizeMapping *pResizeMapping = NULL,
                   const PixMapping *pPixMapping = NULL);

template <typename Backend>
class NewResize : public Resize<Backend> {
 public:
  inline explicit NewResize(const OpSpec &spec) : Resize<Backend>(spec) {
    mirrorParamGPU_.mutable_data<MirroringInfo>();
    resizeParam_.resize((N_GRID_PARAMS + 1) * batch_size_);
    mappingPntr_ = NULL;
    for (size_t i = 0; i < _countof(mapMem_); ++i) {
      mapMem_[i] = NULL;
      resizeMemory_[i] = 0;
    }
  }

  virtual inline ~NewResize() {
    for (size_t i = 0; i < _countof(mapMem_); ++i)
      CUDA_CALL(cudaFree(mapMem_[i]));

    CUDA_CALL(cudaFree(mappingPntr_));
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;
  void SetupSharedSampleParams(Workspace<Backend> *ws) override {
    Resize<Backend> ::SetupSharedSampleParams(ws);
  }
  uint ResizeInfoNeeded() const override { return t_crop + t_mirrorHor; }

 private:
  MappingInfo **CopyResizeTableToGPU(size_t resizeMemory[], cudaStream_t s,
                                      size_t nTable = 1, size_t nSliceNumb = 1) {
    MappingInfo *mapMem[BATCH_SLICE_NUMB];
    for (size_t i = 0; i < nSliceNumb; ++i) {
      if (resizeMemory[i] == UINT_MAX) {
        // Resize tables will not be constructed for that batch slice
        mapMem[i] = NULL;
        continue;
      }

      if (resizeMemory_[i] < resizeMemory[i]) {
        resizeMemory_[i] = resizeMemory[i];

        // Re-allocate memory needed for all resize tables of the i-th batch slice
        CUDA_CALL(cudaFree(mapMem_[i]));

        const size_t length = resizeMemory_[i] * sizeof(mapMem_[0]);
        CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(mapMem_ + i), length));
      }

      mapMem[i] = mapMem_[i];
    }


    if (!mappingPntr_) {
      const size_t length = nTable * sizeof(mappingPntr_[0]);
      CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&mappingPntr_), length));
    }

    const size_t len = nSliceNumb * sizeof(mappingPntr_[0]);
    CUDA_CALL(cudaMemcpyAsync(mappingPntr_, mapMem, len, cudaMemcpyHostToDevice, s));
    return mappingPntr_;
  }

  bool PrepareCropAndResize(const DALISize *input_size, DALISize *out_size, int idx, int C,
                           ResizeGridParam resizeParam[],
                           ResizeMappingTable *ppResizeTbl = NULL) const {
    const DALISize out_resize(*out_size);
    int cropY, cropX;
    ResizeAttr::DefineCrop(out_size, &cropX, &cropY, idx);
    resizeParam[2] = {cropX, cropY};

    return CreateResizeGrid(*input_size, out_resize, C, resizeParam, ppResizeTbl);
  }

  bool CreateResizeGrid(const DALISize &input_size, const DALISize &out_size, int C,
                        ResizeGridParam resizeParam[],
                        ResizeMappingTable *ppResizeTbl = NULL) const {
    const int H0 = input_size.height;
    const int H1 = out_size.height;
    const int W0 = input_size.width;
    const int W1 = out_size.width;

    int lcm(int a, int b);
    const int lcmH = lcm(H0, H1);
    const int lcmW = lcm(W0, W1);

    bool newResize = resizeParam[0].x != lcmW / W0 || resizeParam[0].y != lcmH / H0 ||
                    resizeParam[1].x != lcmW / W1 || resizeParam[1].y != lcmH / H1;

    if (newResize) {
      resizeParam[0] = {lcmW / W0, lcmH / H0};
      resizeParam[1] = {lcmW / W1, lcmH / H1};
    }

    if (newResize && ppResizeTbl)
      ppResizeTbl->constructTable(H0, W0, H1, W1, C, ResizeAttr::interp_type_);

    return newResize;
  }

  bool BatchIsCongeneric(const DALISize *sizeIn, const DALISize *sizeOut, int C) {
      // Check if all input sizes are the same
    const uint32_t imageSize = sizeOut->width * sizeOut->height * C;

    const auto pImages = *ResizeAttr::outputImages();
    const auto pFirstBatchImage = pImages[0];

    int i = batch_size_;
    while (--i > 0) {
      if (!SAME_SIZES(ResizeAttr::size(input_t, i), sizeIn))
        break;

      if (!SAME_SIZES(ResizeAttr::size(output_t, i), sizeOut))
        break;

      if (pImages[i] != pFirstBatchImage + i * imageSize)
        break;
    }

    return i == 0;
  }

  // Members used in RunBatchedGPU;
  ResizeMappingTable resizeTbl_;
                                           // Memory allocated for:
  vector<ResizeGridParam>resizeParam_;     //     Resizing grid AND mirroring parameters on CPU
  ResizeGridDescr resizeParamGPU_;         //                                            on GPU
  MirroringDescr mirrorParamGPU_;
  ImgSizeDescr sizesGPU_[2];               //     Input/Output images sizes on GPU
  ImgRasterDescr imgsGPU_[2];              //     Input/Output images rasters on GPU
  size_t resizeMemory_[BATCH_SLICE_NUMB];  //     total length of simplified resize tables
  MappingInfo *mapMem_[BATCH_SLICE_NUMB];  //     all resize tables of the batch slices on CPU
  MappingInfo **mappingPntr_;              //     all resize tables for batch images on GPU
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_NEW_RESIZE_H_
