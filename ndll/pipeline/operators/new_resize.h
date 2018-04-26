// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_

#include <npp.h>
#include <random>
#include <ctgmath>
#include <vector>

#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

#define BATCH_SLICE_NUMB            32      // The number of slices composing the whole batch
#define USE_RESIZE_TABLE_GPU         0      // Use (1) or not (0) the resize tables on GPU
                                            // NOTE: As of 04/13/2018, the performance with
                                            // these tables is better on CPU, but not on GPU

#define GPU_BACKEND_PARAMS(x, type)         (x).template data<type>()
#define IMG_SIZES(x)                        GPU_BACKEND_PARAMS(x, NDLLSize)
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

#define SET_RESIZE_PARAM()                      \
    const uint32_t sx0 = resizeParam[0].x;      \
    const uint32_t sy0 = resizeParam[0].y;      \
    const uint32_t sx1 = resizeParam[1].x;      \
    const uint32_t sy1 = resizeParam[1].y;      \
    const uint32_t area = sx1 * sy1;            \

#define RESIZE_PREPARE()                        \
    SET_RESIZE_PARAM()                          \
    const uint32_t cropX = resizeParam[2].x;    \
    const uint32_t cropY = resizeParam[2].y;


#define SET_PIXEL_COLOR()                                           \
    out[to] = (pixColor[0] + (area >> 1)) / area;                   \
    if (C > 1) {                                                    \
      out[to + 1] = (pixColor[1] + (area >> 1)) / area;             \
      out[to + 2] = (pixColor[2] + (area >> 1)) / area;             \
    }                                                               \

#define RESIZE_PREAMBLE()                                           \
    RESIZE_PREPARE()                                                \
    uint32_t extraColor[3] = {0, 0, 0};                             \
    uint32_t sumColor[3], pixColor[3];

#define RESIZE_CORE(C)                                                  \
    const uint32_t begIdx[2] = {nX / sx0, nY / sy0};                    \
    const uint32_t endIdx[2] = {(nX + sx1) / sx0, (nY + sy1) / sy0};    \
    const uint32_t extra[2] = {(nX + sx1) % sx0, (nY + sy1) % sy0};     \
    const uint32_t lenFirst[2] = {(sx0 - nX % sx0), (sy0 - nY % sy0)};  \
    uint32_t rowMult = lenFirst[1];                                     \
    pixColor[0] = pixColor[1] = pixColor[2] = 0;                        \
    uint32_t y0 = begIdx[1];                                            \
    while (true) {                                                      \
      size_t x0 = endIdx[0];                                            \
      const uint8 *pPix = in + ((y0 * W0) + x0) * C;                    \
      uint32_t len = extra[0];                                          \
      extraColor[0] = len * *pPix;                                      \
      if (C > 1) {                                                      \
        extraColor[1] = len * *(pPix + 1);                              \
        extraColor[2] = len * *(pPix + 2);                              \
      }                                                                 \
                                                                        \
      sumColor[0] = sumColor[1] = sumColor[2] = 0;                      \
      while (--x0 > begIdx[0]) {                                        \
        pPix -= C;                                                      \
        sumColor[0] += *pPix;                                           \
        if (C > 1) {                                                    \
          sumColor[1] += *(pPix + 1);                                   \
          sumColor[2] += *(pPix + 2);                                   \
        }                                                               \
      }                                                                 \
                                                                        \
      len = lenFirst[0];                                                \
      pixColor[0] += rowMult * (sumColor[0] * sx0 + len * *(pPix -= C) + extraColor[0]);  \
      if (C > 1) {                                                      \
        pixColor[1] += rowMult * (sumColor[1] * sx0 + len * *(pPix + 1) + extraColor[1]); \
        pixColor[2] += rowMult * (sumColor[2] * sx0 + len * *(pPix + 2) + extraColor[2]); \
      }                                                                 \
                                                                        \
      if (++y0  >= endIdx[1]) {                                         \
        if (y0 > endIdx[1] || !(rowMult = extra[1]))                    \
          break;                                                        \
      } else {                                                          \
        rowMult = sy0;                                                  \
      }                                                                 \
    }                                                                   \
    SET_PIXEL_COLOR()

#define CC __host__ __device__      // Calling from CUDA

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

class ResizeMappingTable {
 public:
  NDLLSize io_size[2];
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

  bool IsValid(const NDLLSize &in, const NDLLSize &out, int C) const {
    if (!RESIZE_MAPPING_CPU(resizeMappingCPU) || C_ != C)
      return false;

    return SAME_SIZES(io_size[0], in) && SAME_SIZES(io_size[1], out);
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


#define RESIZE_N_PREAMBLE()  RESIZE_PREPARE()

#define RESIZE_N_CORE(C)                                            \
  auto pBase = in + ((nY / sy0 * W0) + nX / sx0) * C;               \
  if (!pMapping) {                                                  \
    auto pResizePix = pResizeMapping + (nY % sy0) * sx0 + nX % sx0; \
    auto pPixMap = pPixMapping + pResizePix->intersectInfoAddr;     \
    int pixColor[3] = {0, 0, 0};                                    \
    for (int i = pResizePix->nPixels; i--;) {                       \
      auto pPix = pBase + (pPixMap + i)->pixAddr;                   \
      const int pixArea = (pPixMap + i)->pixArea;                   \
      pixColor[0] += *pPix * pixArea;                               \
      if (C > 1) {                                                  \
        pixColor[1] += *(pPix + 1) * pixArea;                       \
        pixColor[2] += *(pPix + 2) * pixArea;                       \
      }                                                             \
    }                                                               \
    SET_PIXEL_COLOR();                                              \
  } else {                                                          \
    auto pPix = pBase + pMapping[(nY % sy0) * sx0 + nX % sx0];      \
    out[to] = *pPix;                                                \
    if (C > 1) {                                                    \
      out[to + 1] = *(pPix +1);                                     \
      out[to + 2] = *(pPix +2);                                     \
    }                                                               \
  }

NDLLError_t BatchedCongenericResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
       const NDLLSize &sizeIn, const uint8 *in_batch, const NDLLSize &sizeOut, uint8 *out_batch,
       const ResizeGridParam *pResizeParam, const MirroringInfo *pMirrorParam,
       MappingInfo * pMapping[], MappingInfo **mapMem, const ResizeMapping *pResizeMapping,
       const PixMapping *pPixMapping, bool newMapping);

NDLLError_t BatchedResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const ResizeGridParam *pResizeParam,
                          const ImgSizeDescr sizes[], const ImgRasterDescr imgRasterGPU[],
                          MappingInfo *pMapping[], MappingInfo **mapMem, size_t nBatchSlice);

// Macros for  creation of the CPU/GPU augmentation methods:
#define AUGMENT_RESIZE(H, W, C, img_in, img_out,                  \
            AUGMENT_PREAMBLE, AUGMENT_CORE,                       \
            stepW, stepH, startW, startH, imgIdx, ...)            \
  AUGMENT_PREAMBLE();                                             \
  int outStep = C;                                                \
  const uint32_t offset = nYoffset(W, C);                         \
  int32_t shift = stepH * offset;                                 \
  const uint8 *in = img_in + H0 *nYoffset(W0, C) * imgIdx;        \
  uint8 *out = img_out + (H * imgIdx + startH) * offset - shift;  \
  if (mirrorVert)                                                 \
    out += (H - 2 * startH - 1) * offset - 2 * (shift *= -1);     \
  if (mirrorHor)                                                  \
    out += offset + (outStep = -C);                               \
  for (int y = startH; y < H; y += stepH) {                       \
    out += shift;                                                 \
    const uint32_t nY = (y + cropY) * sy1;                        \
    for (int x = startW; x < W; x += stepW) {                     \
      const uint32_t nX = (x + cropX) * sx1;                      \
      const int32_t to = x * outStep;                             \
      AUGMENT_CORE(C);                                            \
    }                                                             \
  }

#define AUGMENT_RESIZE_CPU(H, W, C, img_in, img_out, KIND)          \
        AUGMENT_RESIZE(H, W, C, img_in, img_out, KIND ## _PREAMBLE, \
        KIND ## _CORE, 1, 1, 0, 0, 0)

#define AUGMENT_RESIZE_GPU(H, W, C, img_in, img_out, KIND, imgID)   \
        AUGMENT_RESIZE(H, W, C, img_in, img_out, KIND ## _PREAMBLE, \
        KIND ## _CORE, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, imgID)

#define AUGMENT_RESIZE_GPU_CONGENERIC(H, W, C, img_in, img_out, KIND)   \
        AUGMENT_RESIZE_GPU(H, W, C, img_in, img_out, KIND, blockIdx.x)

#define AUGMENT_RESIZE_GPU_GENERIC(H, W, C, img_in, img_out, KIND)  \
        AUGMENT_RESIZE_GPU(H, W, C, img_in, img_out, KIND, 0)

#define nYoffset(W, C)          ((W) * (C))


#define SAME_SIZES(size1, size2)    (size1.height == size2.height && \
                                     size1.width == size2.width)
template <typename Backend>
class NewResize : public Resize<Backend> {
 public:
  inline explicit NewResize(const OpSpec &spec) : Resize<Backend>(spec) {
    mirrorParamGPU_.mutable_data<MirroringInfo>();
    resizeParam_.resize((N_GRID_PARAMS + 1) * batch_size_);
    mappingPntr_ = NULL;
    mapMemGPU_ = NULL;
    for (size_t i = 0; i < _countof(mapMem_); ++i) {
      mapMem_[i] = NULL;
      resizeMemory_[i] = 0;
    }
  }

  virtual inline ~NewResize() {
    for (size_t i = 0; i < _countof(mapMem_); ++i)
      CUDA_CALL(cudaFree(mapMem_[i]));

    CUDA_CALL(cudaFree(mappingPntr_));
    CUDA_CALL(cudaFree(mapMemGPU_));
  }

 protected:
  void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
    const auto &input = ws->Input<CPUBackend>(idx);
    const auto &output = ws->Output<CPUBackend>(idx);

    const auto &input_shape = input.shape();
    NDLLSize out_size, input_size;
    ResizeAttr::SetSize(&input_size, input_shape, ResizeAttr::resize(), &out_size);

    const int C = input_shape[2];

    ResizeGridParam resizeParam[N_GRID_PARAMS] = {};
    ResizeMappingTable resizeTbl;
    PrepareCropAndResize(&input_size, &out_size, C, resizeParam, &resizeTbl);

    const int H0 = input_size.height;
    const int W0 = input_size.width;
    const int H1 = out_size.height;
    const int W1 = out_size.width;
    bool mirrorHor, mirrorVert;
    ResizeAttr::MirrorNeeded(&mirrorHor, &mirrorVert);

    DataDependentSetupCPU(input, output, "NewResize", NULL, NULL, NULL, &out_size);
    const auto pResizeMapping = RESIZE_MAPPING_CPU(resizeTbl.resizeMappingCPU);
    const auto pMapping = RESIZE_MAPPING_CPU(resizeTbl.resizeMappingSimpleCPU);
    const auto pPixMapping = PIX_MAPPING_CPU(resizeTbl.pixMappingCPU);
    AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                     static_cast<uint8 *>(output->raw_mutable_data()), RESIZE_N);
  }

  void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
    const auto &input = ws->Input<GPUBackend>(idx);
    const auto &output = ws->Output<GPUBackend>(idx);
    const bool use_NN = ResizeAttr::type_ == NDLL_INTERP_NN;

    size_t resizeMemory[BATCH_SLICE_NUMB];
    ResizeGridParam *pResizeGrid = resizeParam_.data();
    MirroringInfo *pMirror = pResizeGrid + N_GRID_PARAMS * batch_size_;
    ResizeParamDescr resizeDescr(this, pResizeGrid, pMirror,
                               use_NN ? resizeMemory : NULL, BATCH_SLICE_NUMB);

    const bool newMapping = DataDependentSetupGPU(input, output, batch_size_, false,
                               ResizeAttr::inputImages(), ResizeAttr::outputImages(), NULL,
                               &resizeDescr);

    const int C = input.shape()[0][2];

    const auto sizeIn = *ResizeAttr::size(input_t, 0);
    const auto sizeOut = *ResizeAttr::size(output_t, 0);
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
                            sizeOut.height, sizeOut.width, C, ResizeAttr::type_);
            resizeTbl_.copyToGPU(s);
        }

        pResizeMapping = RESIZE_MAPPING_GPU(resizeTbl_.resizeMappingGPU);
        pPixMapping = PIX_MAPPING_GPU(resizeTbl_.pPixMappingGPU);
      }
#endif

    NDLL_CALL(BatchedCongenericResize(batch_size_, dim3(32, 32), s, C,
                  sizeIn, input.template data<uint8>(),
                  sizeOut, static_cast<uint8 *>(output->raw_mutable_data()),
                  RESIZE_PARAM(resizeParamGPU_), MIRRORING_PARAM(mirrorParamGPU_),
                  mapPntr, mapMemGPU_, pResizeMapping, pPixMapping, newMapping));
    } else {
      resizeParamGPU_.Copy(resizeParam_, s);

      vector<uint8 *> *raster[] = {(vector<uint8 *> *)(ResizeAttr::inputImages()),
                                   ResizeAttr::outputImages()};

      for (int i = input_t; i <= output_t; i++) {
          const auto &sizes = ResizeAttr::sizes(static_cast<io_type >(i));
          TENSOR_COPY(sizesGPU_[i], sizes, s);
          TENSOR_COPY(imgsGPU_[i], *(raster[i]), s);
      }

      NDLL_CALL(BatchedResize(batch_size_, dim3(32, 32), s, C, RESIZE_PARAM(resizeParamGPU_),
                              sizesGPU_, imgsGPU_, mapPntr, mapMemGPU_, _countof(mapMem_)));
    }
  }

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

    const size_t len = nSliceNumb * sizeof(mapMemGPU_[0]);
    if (!mappingPntr_) {
      const size_t length = nTable * sizeof(mappingPntr_[0]);
      CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&mappingPntr_), length));
      CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&mapMemGPU_), len));
    }

    CUDA_CALL(cudaMemcpyAsync(mapMemGPU_, mapMem, len, cudaMemcpyHostToDevice, s));
    return mappingPntr_;
  }

  bool PrepareCropAndResize(const NDLLSize *input_size, NDLLSize *out_size, int C,
                           ResizeGridParam resizeParam[],
                           ResizeMappingTable *ppResizeTbl = NULL) const {
    NDLLSize out_resize(*out_size);
    int cropY, cropX;
    const bool doingCrop = ResizeAttr::CropNeeded(*out_size);
    if (doingCrop)
     ResizeAttr::DefineCrop(out_size, &cropX, &cropY);
    else
     cropY = cropX = 0;

    resizeParam[2] = {cropX, cropY};

    return CreateResizeGrid(*input_size, out_resize, C, resizeParam, ppResizeTbl);
  }

  bool CreateResizeGrid(const NDLLSize &input_size, const NDLLSize &out_size, int C,
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
     ppResizeTbl->constructTable(H0, W0, H1, W1, C, ResizeAttr::type_);

    return newResize;
  }

  bool BatchIsCongeneric(const NDLLSize &sizeIn, const NDLLSize &sizeOut, int C) {
      // Check if all input sizes are the same
    const uint32_t imageSize = sizeOut.width * sizeOut.height * C;

    const auto pImages = *ResizeAttr::outputImages();
    const auto pFirstBatchImage = pImages[0];

    int i = batch_size_;
    while (--i > 0) {
      const auto inSize = *ResizeAttr::size(input_t, i);
      if (!SAME_SIZES(inSize, sizeIn))
       break;

      const auto outSize = *ResizeAttr::size(output_t, i);
      if (!SAME_SIZES(outSize, sizeOut))
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
  MappingInfo **mapMemGPU_;                //     all resize tables of the batch slices on GPU
  MappingInfo **mappingPntr_;              //     all resize tables for batch images on GPU
  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_
