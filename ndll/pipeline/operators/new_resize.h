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

#define CUDA_MALLOC(x, len)     CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&x), len))
#define CUDA_FREE(x)            { CUDA_CALL(cudaFree(x)); x = NULL; }
#define CUDA_MEMCPY(x, y, len)  CUDA_CALL(cudaMemcpy(x, y, len, cudaMemcpyHostToDevice))

struct ResizeGridParam {
    int nX;
    int nY;
};

#define N_GRID_PARAMS       3

#define RESIZE_PREPARE()                   \
    const int sx0 = resizeParam[0].nX;     \
    const int sy0 = resizeParam[0].nY;     \
    const int sx1 = resizeParam[1].nX;     \
    const int sy1 = resizeParam[1].nY;     \
    const int cropX = resizeParam[2].nX;   \
    const int cropY = resizeParam[2].nY;   \
    const int area = sx1 * sy1;

#define SET_PIXEL_COLOR()                                           \
    const uint32_t to = x * C;                                      \
    out[to] = (pixColor[0] + (area >> 1)) / area;                   \
    if (C > 1) {                                                    \
        out[to + 1] = (pixColor[1] + (area >> 1)) / area;           \
        out[to + 2] = (pixColor[2] + (area >> 1)) / area;           \
    }                                                               \

#define RESIZE_PREAMBLE(H, W, C)                \
    RESIZE_PREPARE()                            \
    uint32_t extraColor[3] = {0, 0, 0};         \
    uint32_t sumColor[3], pixColor[3];

#define RESIZE_CORE(C)                                              \
    const int begIdx[2] = {nX / sx0, nY / sy0};                     \
    const int endIdx[2] = {(nX + sx1) / sx0, (nY + sy1) / sy0};     \
    const int extra[2] = {(nX + sx1) % sx0, (nY + sy1) % sy0};      \
    const int lenFirst[2] = {(sx0 - nX % sx0), (sy0 - nY % sy0)};   \
    int rowMult = lenFirst[1];                                      \
    pixColor[0] = pixColor[1] = pixColor[2] = 0;                    \
    int y0 = begIdx[1];                                             \
    while (true) {                                                  \
        int x0 = endIdx[0];                                         \
        const uint8 *pPix = in + ((y0 * W0) + x0) * C;              \
        int len = extra[0];                                         \
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

typedef struct {
    uint16_t nPixels;               // number of the pixels, intersecting with the resulting pixel
    uint32_t intersectInfoAddr;     // address to the information for first intersecting pixel
} ResizeMapping;

typedef struct {
    uint32_t pixAddr;               // relative address to the pixels of the initial image
    uint32_t pixArea;               // area of its intersection with the resulting pixels
    void Init(uint32_t addr, uint32_t area) { pixAddr = addr; pixArea = area; }
} PixMapping;

typedef struct ResizeMappingTable {
    NDLLSize io_size[2];
    int C_;
    uint32_t tableLength;               // sizes of the ResizeMapping
    ResizeMapping *pResizeMapping[2];   // pointer to the ResizeMapping table for CPU/GPU
    PixMapping *pPixMapping[2];         // pointer to the PixMapping arrays  for CPU/GPU
    uint32_t pixMappingLen;             // length of array pPixMapping in bytes

    ResizeMappingTable(int H0, int W0, int H1, int W1, int C, uint16_t xSize, uint16_t ySize);
    ~ResizeMappingTable();
    inline uint32_t getMappingTableLength() const   { return tableLength; }
    bool IsValid(int H0, int W0, int H1, int W1) const;
    void CopyCongenericResizeParam();
 private:
    void releaseCudaResizeMapingTable();
} ResizeMappingTable;


#define RESIZE_N_PREAMBLE(H, W, C)                                      \
    RESIZE_PREPARE()                                                    \
    const ResizeMapping *pResizePix;                                    \
    const PixMapping *pPixMap;


#define RESIZE_N_CORE(C)                                                \
    pResizePix = pResizeMapping + (nY % sy0) * sx0 + nX % sx0;          \
    const uint8 *pBase = in + ((nY / sy0 * W0) + nX / sx0) * C;         \
    if (pPixMapping) {                                                  \
        pPixMap = pPixMapping + pResizePix->intersectInfoAddr;          \
        int pixColor[3] = {0, 0, 0};                                    \
        for (int i = pResizePix->nPixels; i--;) {                       \
            const uint8 *pPix = pBase + (pPixMap + i)->pixAddr;         \
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
        const uint8 *pPix = pBase + pResizePix->intersectInfoAddr;      \
        out[to] = *pPix;                                                \
        if (C > 1) {                                                    \
            out[to + 1] = *(pPix +1);                                   \
            out[to + 2] = *(pPix +2);                                   \
        }                                                               \
    }

NDLLError_t BatchedCongenericResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const NDLLSize &sizeIn, const uint8 *in_batch,
                          const NDLLSize &sizeOut, uint8 *out_batch,
                          const ResizeGridParam *pResizeParam, const ResizeMappingTable *pTbl);

NDLLError_t BatchedResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const NppiRect *resizeDescr,
                          const NDLLSize * const sizes[], uint8 ** const imgRasterGPU[]);

ResizeMappingTable *createResizeMappingTable(int H0, int W0, int H1, int W1, int C, bool closest);

#define SAME_SIZES(size1, size2)    (size1->height == size2->height && \
                                     size1->width == size2->width)
template <typename Backend>
class NewResize : public Resize<Backend> {
 public:
    inline explicit NewResize(const OpSpec &spec) : Resize<Backend>(spec) {
        resizeDescr_.resize(batch_size_);
        resizeParamGPU_ = NULL;
        for (int i = 0; i < 2; i++) {
            sizesGPU_[i] = NULL;
            imgsGPU_[i] = NULL;
        }

        resizeDescrGPU_ = NULL;
        setBatchSizeMax(0);
    }

    virtual inline ~NewResize()             { releaseCudaResizeParameter(); }

 protected:
    void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        const auto output = ws->Output<CPUBackend>(idx);

        const vector <Index> &input_shape = input.shape();
        NDLLSize out_size, input_size;
        ResizeAttr::SetSize(&input_size, input_shape, ResizeAttr::resize(), &out_size);

        const vector<Index> &shape = input.shape();
        const int C = shape[2];

        ResizeGridParam resizeParam[N_GRID_PARAMS] = {};
        ResizeMappingTable *pResizeTbl = NULL;
        PrepareCropAndResize(&input_size, &out_size, C, resizeParam, &pResizeTbl);

        const int H0 = input_size.height;
        const int W0 = input_size.width;
        const int H1 = out_size.height;
        const int W1 = out_size.width;

        DataDependentSetupCPU(input, output, "NewResize", NULL, NULL, NULL, &out_size);

        if (pResizeTbl) {
            const ResizeMapping *pResizeMapping = pResizeTbl->pResizeMapping[0];
            const PixMapping *pPixMapping = pResizeTbl->pPixMapping[0];
            AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                               static_cast<uint8 *>(output->raw_mutable_data()), RESIZE_N);
        } else {
            AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                               static_cast<uint8 *>(output->raw_mutable_data()), RESIZE);
        }
    }

    void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<GPUBackend>(idx);
        const auto output = ws->Output<GPUBackend>(idx);

        DataDependentSetupGPU(input, output, batch_size_, false,
              ResizeAttr::inputImages(), ResizeAttr::outputImages(), NULL, this, &resizeDescr_);

        const auto &shape = input.shape();
        const int C = shape[0][2];

        const NDLLSize *sizeIn = ResizeAttr::size(input_t, 0);
        const NDLLSize *sizeOut = ResizeAttr::size(output_t, 0);

        if (BatchIsCongeneric(sizeIn, sizeOut, C)) {
            NDLLSize out_size = {resizeDescr_[0].width, resizeDescr_[0].height};
            ResizeGridParam resizeParam[N_GRID_PARAMS];
            ResizeMappingTable *pResizeTbl = NULL;
            const bool newMapping = PrepareCropAndResize(sizeIn, &out_size, C,
                                                         resizeParam, &pResizeTbl);
            if (newMapping) {
                if (!resizeParamGPU_)
                    CUDA_MALLOC(resizeParamGPU_, sizeof(resizeParam));

                // Copying the descriptor of operation into __constant__ memory
                CUDA_MEMCPY(resizeParamGPU_, resizeParam, sizeof(resizeParam));

                if (pResizeTbl)
                    pResizeTbl->CopyCongenericResizeParam();
            }

            NDLL_CALL(BatchedCongenericResize(batch_size_, dim3(32, 32), ws->stream(), C,
                        *sizeIn, input.template data<uint8>(),
                        *sizeOut, static_cast<uint8 *>(output->raw_mutable_data()),
                        resizeParamGPU_, pResizeTbl));

            delete pResizeTbl;
        } else {
            const size_t len1 = batch_size_ * sizeof(sizesGPU_[0][0]);
            const size_t len2 = batch_size_ * sizeof(imgsGPU_[0][0]);
            const size_t len3 = batch_size_ * sizeof(resizeDescrGPU_[0]);
            if (BatchSizeMax() < batch_size_) {
                releaseCudaResizeParameter();
                for (int i = input_t; i <= output_t; i++) {
                    CUDA_MALLOC(sizesGPU_[i], len1);
                    CUDA_MALLOC(imgsGPU_[i], len2);
                }

                CUDA_MALLOC(resizeDescrGPU_, len3);
                setBatchSizeMax(batch_size_);
            }

            vector<uint8 *> *raster[] = {(vector<uint8 *> *)(ResizeAttr::inputImages()),
                                         ResizeAttr::outputImages()};

            for (int i = input_t; i <= output_t; i++) {
                const vector<NDLLSize> &sizes =  ResizeAttr::sizes(static_cast<io_type >(i));
                CUDA_MEMCPY(sizesGPU_[i], sizes.data(), len1);
                CUDA_MEMCPY(imgsGPU_[i], raster[i]->data(), len2);
            }

            CUDA_MEMCPY(resizeDescrGPU_, resizeDescr_.data(), len3);

            NDLL_CALL(BatchedResize(batch_size_, dim3(32, 32), ws->stream(),
                                    C, resizeDescrGPU_, sizesGPU_, imgsGPU_));
        }
    }

 private:
    bool PrepareCropAndResize(const NDLLSize *input_size, NDLLSize *out_size, int C,
                              ResizeGridParam resizeParam[],
                              ResizeMappingTable **ppResizeTbl = NULL) const {
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
                          ResizeMappingTable **ppResizeTbl = NULL) const {
        const int H0 = input_size.height;
        const int H1 = out_size.height;
        const int W0 = input_size.width;
        const int W1 = out_size.width;

        int lcm(int a, int b);
        const int lcmH = lcm(H0, H1);
        const int lcmW = lcm(W0, W1);

        bool newResize = resizeParam[0].nX != lcmW / W0 || resizeParam[0].nY != lcmH / H0 ||
                         resizeParam[1].nX != lcmW / W1 || resizeParam[1].nY != lcmH / H1;

        if (newResize) {
            resizeParam[0] = {lcmW / W0, lcmH / H0};
            resizeParam[1] = {lcmW / W1, lcmH / H1};
        }

        if (ppResizeTbl) {
            ResizeMappingTable *pResizeTbl = *ppResizeTbl;
            if (pResizeTbl && !pResizeTbl->IsValid(H0, W0, H1, W1)) {
                delete pResizeTbl;
                pResizeTbl = NULL;
            }

            if (!pResizeTbl) {
                *ppResizeTbl = createResizeMappingTable(H0, W0, H1, W1, C,
                                                       ResizeAttr::type_ == NDLL_INTERP_NN);
                newResize = true;
            }
        }

        return newResize;
    }

    bool BatchIsCongeneric(const NDLLSize *sizeIn, const NDLLSize *sizeOut, int C) {
        // Check if all input sizes are the same
        const uint32_t imageSize = sizeOut->width * sizeOut->height * C;

        const auto pImages = *ResizeAttr::outputImages();
        const auto pFirstBatchImage = pImages[0];

        int i = batch_size_;
        while (--i > 0) {
            const NDLLSize *inSize = ResizeAttr::size(input_t, i);
            if (!SAME_SIZES(inSize, sizeIn))
                break;

            const NDLLSize *outSize = ResizeAttr::size(output_t, i);
            if (!SAME_SIZES(outSize, sizeOut))
                break;

            if (pImages[i] != pFirstBatchImage + i * imageSize)
                break;
        }

        return i == 0;
    }

    inline void setBatchSizeMax(int value)      { nBatchSizeMax_ = value; }
    inline int BatchSizeMax() const             { return nBatchSizeMax_; }
    void releaseCudaResizeParameter() {
        for (int i = input_t; i <= output_t; i++) {
            CUDA_FREE(sizesGPU_[i]);
            CUDA_FREE(imgsGPU_[i]);
        }

        CUDA_FREE(resizeDescrGPU_);
        setBatchSizeMax(0);
    }

    // Members used in RunBatchedGPU;
    ResizeGridParam *resizeParamGPU_;
    vector<NppiRect> resizeDescr_;
    NDLLSize *sizesGPU_[2];
    uint8 ** imgsGPU_[2];
    NppiRect *resizeDescrGPU_;
    int nBatchSizeMax_;

    USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_
