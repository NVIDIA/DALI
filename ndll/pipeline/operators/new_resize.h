// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_

#include <npp.h>
#include <random>
#include <ctgmath>
#include <vector>
#include <nppdefs.h>

#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

#define KEEP_RESIZE_TABLE            0      // When 0, the ResizeTable is re-created every
                                            // time RunBatchedGPU is called
#define RESIZE_TABLE_ALLOC           1      // Allocate memory for all resize tables as one (1)
                                            // or multiple (0) pieces

#define GPU_BACKEND_PARAMS(x, type)         (x).template data<type>()
#define IMG_SIZES(x)                        GPU_BACKEND_PARAMS(x, NDLLSize)
#define IMG_RASTERS(x)                      GPU_BACKEND_PARAMS(x, uint8 *)
#define PIX_MAPPING_GPU(x)                  GPU_BACKEND_PARAMS(x, PixMapping)
#define RESIZE_MAPPING_GPU(x)               GPU_BACKEND_PARAMS(x, ResizeMapping)
#define RESIZE_MAPPING_S_GPU(x)             GPU_BACKEND_PARAMS(x, MappingInfo)
#define RESIZE_PARAM(x)                     GPU_BACKEND_PARAMS(x, ResizeGridParam)

#define TENSOR_COPY(x, y, s)                (x).Copy(y, s)
#define TENSOR_COPY_SIZES(x, y, s)          TENSOR_COPY(x, y, s)
#define TENSOR_COPY_RASTERS(x, y, s)        TENSOR_COPY(x, y, s)
#define TENSOR_COPY_PIX_MAPPING(x, y, s)    TENSOR_COPY(x, y, s)

#define PIX_MAPPING_CPU(x)                  (x).data()
#define RESIZE_MAPPING_CPU(x)               (x).data()

#define N_GRID_PARAMS       3

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
    const uint32_t to = x * C;                                      \
    out[to] = (pixColor[0] + (area >> 1)) / area;                   \
    if (C > 1) {                                                    \
        out[to + 1] = (pixColor[1] + (area >> 1)) / area;           \
        out[to + 2] = (pixColor[2] + (area >> 1)) / area;           \
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
    while (true) {                                                  \
        size_t x0 = endIdx[0];                                      \
        const uint8 *pPix = in + ((y0 * W0) + x0) * C;              \
        uint32_t len = extra[0];                                    \
        extraColor[0] = len * *pPix;                                \
        if (C > 1) {                                                \
            extraColor[1] = len * *(pPix + 1);                      \
            extraColor[2] = len * *(pPix + 2);                      \
        }                                                           \
                                                                    \
        sumColor[0] = sumColor[1] = sumColor[2] = 0;                \
        while (--x0 > begIdx[0]) {                                  \
            pPix -= C;                                              \
            sumColor[0] += *pPix;                                   \
            if (C > 1) {                                            \
                sumColor[1] += *(pPix + 1);                         \
                sumColor[2] += *(pPix + 2);                         \
            }                                                       \
        }                                                           \
                                                                    \
        len = lenFirst[0];                                          \
        pixColor[0] += rowMult * (sumColor[0] * sx0 + len * *(pPix -= C) + extraColor[0]);    \
        if (C > 1) {                                                \
            pixColor[1] += rowMult * (sumColor[1] * sx0 + len * *(pPix + 1) + extraColor[1]); \
            pixColor[2] += rowMult * (sumColor[2] * sx0 + len * *(pPix + 2) + extraColor[2]); \
        }                                                           \
                                                                    \
        if (++y0  >= endIdx[1]) {                                   \
            if (y0 > endIdx[1] || !(rowMult = extra[1]))            \
                break;                                              \
        } else {                                                    \
            rowMult = sy0;                                          \
        }                                                           \
    }                                                               \
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

typedef Tensor<GPUBackend> ImgSizeDescr, ImgRasterDescr, ResizeMappingPixDescrGPU,
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
    ResizeMappingTableGPU resizeMappingGPU;     //      ResizeMapping table on GPU
    ResizeMappingCPU resizeMappingSimpleCPU;    //      simplified ResizeMapping table on CPU
    ResizeMappingGPU resizeMappingSimpleGPU;    //      simpified ResizeMapping table on GPU
    ResizeMappingPixDescrCPU pixMappingCPU;     //      PixMapping arrays for CPU
    ResizeMappingPixDescrGPU pPixMappingGPU;    //      PixMapping arrays for GPU

    bool IsValid(int H0, int W0, int H1, int W1, int C) const;
    void constructTable(int H0, int W0, int H1, int W1, int C, int resizeType);
    inline void copyToGPU(cudaStream_t s, bool flag)   {
        if (flag) {
            resizeMappingGPU.Copy(resizeMappingCPU, s);
            pPixMappingGPU.mutable_data<PixMapping>();
            TENSOR_COPY_PIX_MAPPING(pPixMappingGPU, pixMappingCPU, s);
        } else {
            resizeMappingSimpleGPU.Copy(resizeMappingSimpleCPU, s);
        }
    }

 private:
    void initTable(int H0, int W0, int H1, int W1, int C,
                   uint16_t xSize, uint16_t ySize, bool use_NN);
};


#define RESIZE_N_PREAMBLE()  RESIZE_PREPARE()

#define RESIZE_N_CORE(C)                                                \
    auto pBase = in + ((nY / sy0 * W0) + nX / sx0) * C;                 \
    if (!pMapping) {                                                    \
        auto pResizePix = pResizeMapping + (nY % sy0) * sx0 + nX % sx0; \
        auto pPixMap = pPixMapping + pResizePix->intersectInfoAddr;     \
        int pixColor[3] = {0, 0, 0};                                    \
        for (int i = pResizePix->nPixels; i--;) {                       \
            auto pPix = pBase + (pPixMap + i)->pixAddr;                 \
            const int pixArea = (pPixMap + i)->pixArea;                 \
            pixColor[0] += *pPix * pixArea;                             \
            if (C > 1) {                                                \
                pixColor[1] += *(pPix + 1) * pixArea;                   \
                pixColor[2] += *(pPix + 2) * pixArea;                   \
            }                                                           \
        }                                                               \
        SET_PIXEL_COLOR();                                              \
    } else {                                                            \
        const uint32_t to = x * C;                                      \
        auto pPix = pBase + pMapping[(nY % sy0) * sx0 + nX % sx0];      \
        out[to] = *pPix;                                                \
        if (C > 1) {                                                    \
            out[to + 1] = *(pPix +1);                                   \
            out[to + 2] = *(pPix +2);                                   \
        }                                                               \
    }

NDLLError_t BatchedCongenericResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
       const NDLLSize &sizeIn, const uint8 *in_batch, const NDLLSize &sizeOut, uint8 *out_batch,
       const ResizeGridParam *pResizeParam, const MappingInfo *pMapping,
       const ResizeMapping *pResizeMapping, const PixMapping *pPixMapping = NULL);

NDLLError_t BatchedResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const ResizeGridParam *pResizeParam,
                          const ImgSizeDescr sizes[], const ImgRasterDescr imgRasterGPU[],
                          MappingInfo *pResizeMapping[], MappingInfo *mappingMem);

// Macros for  creation of the CPU/GPU augmentation methods:
#define AUGMENT_RESIZE(H, W, C, img_in, img_out,                    \
            AUGMENT_PREAMBLE, AUGMENT_CORE,                         \
            stepW, stepH, startW, startH, imgIdx, ...)              \
    AUGMENT_PREAMBLE();                                             \
    const uint32_t offset = nYoffset(W, C);                         \
    const uint32_t shift = stepH * offset;                          \
    const uint8 *in = img_in + H0 *nYoffset(W0, C) * imgIdx;        \
    uint8 *out = img_out + (H * imgIdx + startH) * offset - shift;  \
    for (int y = startH; y < H; y += stepH) {                       \
        out += shift;                                               \
        const uint32_t nY = (y + cropY) * sy1;                      \
        for (int x = startW; x < W; x += stepW) {                   \
            const uint32_t nX = (x + cropX) * sx1;                  \
            AUGMENT_CORE(C);                                        \
        }                                                           \
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


#define SAME_SIZES(size1, size2)    (size1->height == size2->height && \
                                     size1->width == size2->width)
template <typename Backend>
class NewResize : public Resize<Backend> {
 public:
    inline explicit NewResize(const OpSpec &spec) : Resize<Backend>(spec) {
        resizeParam_.resize(N_GRID_PARAMS * batch_size_);
        mappingPntr_ = NULL;
        mappingMem_ = NULL;
        resizeMemory_ = 0;
    }

    virtual inline ~NewResize()         {
        CUDA_CALL(cudaFree(mappingMem_));
        CUDA_CALL(cudaFree(mappingPntr_));
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

        size_t resizeMemory = 0;
        const bool newMapping = DataDependentSetupGPU(input, output, batch_size_, false,
              ResizeAttr::inputImages(), ResizeAttr::outputImages(), NULL, this,
              resizeParam_.data(), use_NN && RESIZE_TABLE_ALLOC? &resizeMemory : NULL);

        const auto &shape = input.shape();
        const int C = shape[0][2];

        const auto sizeIn = ResizeAttr::size(input_t, 0);
        const auto sizeOut = ResizeAttr::size(output_t, 0);
        cudaStream_t s = ws->stream();

        if (BatchIsCongeneric(sizeIn, sizeOut, C)) {
#if !KEEP_RESIZE_TABLE
            ResizeMappingTable resizeTbl_;
#endif
            if (newMapping) {
                resizeTbl_.constructTable(sizeIn->height, sizeIn->width,
                                          sizeOut->height, sizeOut->width, C, ResizeAttr::type_);

                // Copying the descriptor of operation into GPU
                resizeParamGPU_.Copy(vector<ResizeGridParam>(
                        resizeParam_.begin(), resizeParam_.begin() + N_GRID_PARAMS), s);

                resizeTbl_.copyToGPU(s, !use_NN);
            }

            auto pMapping = use_NN? RESIZE_MAPPING_S_GPU(resizeTbl_.resizeMappingSimpleGPU) : NULL;
            auto pResizeMapping = use_NN? NULL : RESIZE_MAPPING_GPU(resizeTbl_.resizeMappingGPU);
            auto pPixMapping = use_NN? NULL : PIX_MAPPING_GPU(resizeTbl_.pPixMappingGPU);
            NDLL_CALL(BatchedCongenericResize(batch_size_, dim3(32, 32), s, C,
                        *sizeIn, input.template data<uint8>(),
                        *sizeOut, static_cast<uint8 *>(output->raw_mutable_data()),
                        RESIZE_PARAM(resizeParamGPU_), pMapping, pResizeMapping, pPixMapping));
        } else {
            resizeParamGPU_.Copy(resizeParam_, s);

            vector<uint8 *> *raster[] = {(vector<uint8 *> *)(ResizeAttr::inputImages()),
                                         ResizeAttr::outputImages()};

            for (int i = input_t; i <= output_t; i++) {
                const auto &sizes = ResizeAttr::sizes(static_cast<io_type >(i));
                TENSOR_COPY_SIZES(sizesGPU_[i], sizes, s);
                TENSOR_COPY_RASTERS(imgsGPU_[i], *(raster[i]), s);
            }

            MappingInfo **mapPntr = NULL;
            if (use_NN && (!RESIZE_TABLE_ALLOC || resizeMemory)) {
                if (!mappingPntr_) {
                    const size_t len = batch_size_ * sizeof(mappingPntr_[0]);
                    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&mappingPntr_), len));
                }

#if RESIZE_TABLE_ALLOC
                if (resizeMemory_ < resizeMemory) {
                    // We need to allocate more memory
                    if (resizeMemory_)
                        CUDA_CALL(cudaFree(mappingPntr_[0]));

                    resizeMemory_ = resizeMemory;

                    // Allocate memory needed for all resize tables
                    const size_t len = resizeMemory_* sizeof(MappingInfo);
                    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&mappingMem_), len));
                }
#endif
                mapPntr = mappingPntr_;
            }

            NDLL_CALL(BatchedResize(batch_size_, dim3(32, 32), s, C,
                      RESIZE_PARAM(resizeParamGPU_), sizesGPU_, imgsGPU_, mapPntr, mappingMem_));
        }
    }

 private:
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

    bool BatchIsCongeneric(const NDLLSize *sizeIn, const NDLLSize *sizeOut, int C) {
        // Check if all input sizes are the same
        const uint32_t imageSize = sizeOut->width * sizeOut->height * C;

        const auto pImages = *ResizeAttr::outputImages();
        const auto pFirstBatchImage = pImages[0];

        int i = batch_size_;
        while (--i > 0) {
            const auto inSize = ResizeAttr::size(input_t, i);
            if (!SAME_SIZES(inSize, sizeIn))
                break;

            const auto outSize = ResizeAttr::size(output_t, i);
            if (!SAME_SIZES(outSize, sizeOut))
                break;

            if (pImages[i] != pFirstBatchImage + i * imageSize)
                break;
        }

        return i == 0;
    }

    // Members used in RunBatchedGPU;
#if KEEP_RESIZE_TABLE
    ResizeMappingTable resizeTbl_;
#endif
    vector<ResizeGridParam>resizeParam_;
    ResizeGridDescr resizeParamGPU_;
    ImgSizeDescr sizesGPU_[2];     // Input/Output images sizes on GPU
    ImgRasterDescr imgsGPU_[2];    // Input/Output images rasters on GPU
    MappingInfo **mappingPntr_;    // Memory allocated for pointers of resizeMapping tables.
    MappingInfo *mappingMem_;      // Memory allocated for all resized tables
    size_t resizeMemory_;          // Memory allocated for ALL simplified resize tables.

    USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_
