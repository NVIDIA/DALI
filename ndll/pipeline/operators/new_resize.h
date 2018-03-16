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

#define USE_CPU_BACKEND_MALLOC      1
#define PINNED                      false

#define BACKEND_MALLOC(x, len)      x.pntr = Backend::New(x.length = len, PINNED)
#define BACKEND_FREE(x)             { if (x.pntr)       \
                                        Backend::Delete(x.pntr, x.length, PINNED); x.reset(); }

#if USE_CPU_BACKEND_MALLOC
#define CPU_BACKEND_MALLOC(x, len, reset, type)             \
                                    (x).setMemory(CPUBackend::New(len, PINNED), len, reset);
#define CPU_BACKEND_FREE(x, type)   \
            { if ((x).pntr) {CPUBackend::Delete((x).pntr, (x).length, PINNED); (x).reset(); } }
#else
#define CPU_BACKEND_MALLOC(x, len, reset, type)             \
                                    (x).setMemory(new type[len/sizeof(type)], len, reset)
#define CPU_BACKEND_FREE(x, type)   { delete [] POINTER_OF_TYPE(type, x); (x).reset(); }
#endif

#define CUDA_MEMCPY(x, y, len, s)       \
            CUDA_CALL(cudaMemcpyAsync(x, y, len, cudaMemcpyHostToDevice, s));

#define CUDA_COPY_ELEMENTS(x, y, n, s)  CUDA_MEMCPY(x.pntr, y, n * sizeof(y[0]), s)

#define CUDA_COPY(x, y, len, s) if (y)  \
            { BACKEND_MALLOC(x, len); CUDA_MEMCPY(x.pntr, y, x.length, s); }

#define COPY_TO_DEVICE(x, y, s) CUDA_COPY(x, y.pntr, y.length, s);


struct ResizeGridParam {
    int nX;
    int nY;
};

struct ClassHandle {
    void *pntr;
    size_t length;
    inline ClassHandle()                                              { reset(); }
    inline void reset()                                               { setMemory(NULL, 0); }
    inline void setMemory(void *p, size_t len, bool initMem = false)  { length = len;
                                                                        if ((pntr = p) && initMem)
                                                                            memset(pntr, 0, len);
                                                                      }
};

#define POINTER_OF_TYPE(type, x)    static_cast<type *>((x).pntr)
#define MAPPING_TABLE(x)            POINTER_OF_TYPE(ResizeMappingTable, x)
#define RESIZE_MAPPING(x)           POINTER_OF_TYPE(ResizeMapping, x)
#define PIX_MAPPING(x)              POINTER_OF_TYPE(PixMapping, x)
#define RESIZE_PARAM(x)             POINTER_OF_TYPE(ResizeGridParam, x)
#define CROP_PARAMS(x)              POINTER_OF_TYPE(NppiRect, x)
#define IMG_SIZES(x)                POINTER_OF_TYPE(NDLLSize, x)
#define IMG_RASTERS(x)              POINTER_OF_TYPE(uint8 *, x)

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

#define RESIZE_PREAMBLE(H, W, C)                                    \
    RESIZE_PREPARE()                                                \
    uint32_t extraColor[3] = {0, 0, 0};                             \
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


class ResizeMappingTable {
 public:
    NDLLSize io_size[2];
    int C_;
    ClassHandle pResizeMapping[2];   // pointer to the ResizeMapping table for CPU/GPU
    ClassHandle pPixMapping[2];      // pointer to the PixMapping arrays  for CPU/GPU

    ResizeMappingTable()            {}
    ~ResizeMappingTable()           { closeTable(); }
    bool IsValid(int H0, int W0, int H1, int W1) const;
    void constructTable(int H0, int W0, int H1, int W1, int C, bool use_NN);
    void closeTable();
 private:
    void initTable(int H0, int W0, int H1, int W1, int C, uint16_t xSize, uint16_t ySize);
};


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

void CollectPointersForExecution(size_t batch_size,
                                 const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
                                 TensorList<GPUBackend> *output, vector<uint8 *> *outPtrs);

NDLLError_t BatchedCongenericResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const NDLLSize &sizeIn, const uint8 *in_batch,
                          const NDLLSize &sizeOut, uint8 *out_batch,
                          const ResizeGridParam *pResizeParam, const ResizeMappingTable *pTbl);

NDLLError_t BatchedResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const NppiRect *resizeDescr, const ClassHandle sizes[],
                          const ClassHandle imgRasterGPU[]);

// Macros for  creation of the CPU/GPU augmentation methods:
#define AUGMENT_RESIZE(H, W, C, img_in, img_out,                    \
            AUGMENT_PREAMBLE, AUGMENT_CORE,                         \
            stepW, stepH, startW, startH, imgIdx, ...)              \
    AUGMENT_PREAMBLE(H, W, C);                                      \
    const uint32_t offset = nYoffset(W, C);                         \
    const uint32_t shift = stepH * offset;                          \
    const uint8 *in = img_in + H0 *nYoffset(W0, C) * imgIdx;        \
    uint8 *out = img_out + (H * imgIdx + startH) * offset - shift;  \
    for (int y = startH; y < H; y += stepH) {                       \
        out += shift;                                               \
        const int nY = (y + cropY) * sy1;                           \
        for (int x = startW; x < W; x += stepW) {                   \
            const int nX = (x + cropX) * sx1;                       \
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
        resizeDescr_.resize(batch_size_);

        for (int i = input_t; i <= output_t; i++) {
            BACKEND_MALLOC(sizesGPU_[i], batch_size_ * sizeof(NDLLSize));
            BACKEND_MALLOC(imgsGPU_[i], batch_size_ * sizeof(uint8 *));
        }

        BACKEND_MALLOC(resizeDescrGPU_, batch_size_ * sizeof(NppiRect));
    }

    virtual inline ~NewResize()             { releaseCudaResizeParameter(); }

 protected:
    void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        const auto &output = ws->Output<CPUBackend>(idx);

        const vector <Index> &input_shape = input.shape();
        NDLLSize out_size, input_size;
        ResizeAttr::SetSize(&input_size, input_shape, ResizeAttr::resize(), &out_size);

        const vector<Index> &shape = input.shape();
        const int C = shape[2];

        ResizeGridParam resizeParam[N_GRID_PARAMS] = {};
        ClassHandle resizeTbl;
        PrepareCropAndResize(&input_size, &out_size, C, resizeParam, &resizeTbl);

        const int H0 = input_size.height;
        const int W0 = input_size.width;
        const int H1 = out_size.height;
        const int W1 = out_size.width;

        DataDependentSetupCPU(input, output, "NewResize", NULL, NULL, NULL, &out_size);

        ResizeMappingTable *pResizeTbl = MAPPING_TABLE(resizeTbl);
        if (pResizeTbl) {
            const ResizeMapping *pResizeMapping = RESIZE_MAPPING(pResizeTbl->pResizeMapping[0]);
            const PixMapping *pPixMapping = PIX_MAPPING(pResizeTbl->pPixMapping[0]);
            AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                               static_cast<uint8 *>(output->raw_mutable_data()), RESIZE_N);

            pResizeTbl->closeTable();
            CPU_BACKEND_FREE(resizeTbl, ResizeMappingTable);
        } else {
            AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                               static_cast<uint8 *>(output->raw_mutable_data()), RESIZE);
        }
    }

    void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<GPUBackend>(idx);
        const auto &output = ws->Output<GPUBackend>(idx);

        DataDependentSetupGPU(input, output, batch_size_, false,
              ResizeAttr::inputImages(), ResizeAttr::outputImages(), NULL, this, &resizeDescr_);

        const auto &shape = input.shape();
        const int C = shape[0][2];

        const NDLLSize *sizeIn = ResizeAttr::size(input_t, 0);
        const NDLLSize *sizeOut = ResizeAttr::size(output_t, 0);
        cudaStream_t s = ws->stream();

        if (BatchIsCongeneric(sizeIn, sizeOut, C)) {
            NDLLSize out_size = {resizeDescr_[0].width, resizeDescr_[0].height};
            ResizeGridParam resizeParam[N_GRID_PARAMS], *pResize = resizeParam;
            ClassHandle resizeTbl;
            const bool newMapping = PrepareCropAndResize(sizeIn, &out_size, C,
                                                         resizeParam, &resizeTbl);

            ResizeMappingTable *pResizeTbl = MAPPING_TABLE(resizeTbl);
            if (newMapping) {
                // Copying the descriptor of operation into __constant__ memory
                CUDA_COPY(resizeParamGPU_, pResize, sizeof(resizeParam), s);
                if (pResizeTbl)
                    CopyCongenericResizeParam(pResizeTbl, s);
            }

            NDLL_CALL(BatchedCongenericResize(batch_size_, dim3(32, 32), s, C,
                        *sizeIn, input.template data<uint8>(),
                        *sizeOut, static_cast<uint8 *>(output->raw_mutable_data()),
                        RESIZE_PARAM(resizeParamGPU_), pResizeTbl));

            pResizeTbl->closeTable();
            CPU_BACKEND_FREE(resizeTbl, ResizeMappingTable);
        } else {
            vector<uint8 *> *raster[] = {(vector<uint8 *> *)(ResizeAttr::inputImages()),
                                         ResizeAttr::outputImages()};

            for (int i = input_t; i <= output_t; i++) {
                const vector<NDLLSize> &sizes =  ResizeAttr::sizes(static_cast<io_type >(i));
                CUDA_COPY_ELEMENTS(sizesGPU_[i], sizes.data(), batch_size_, s);
                CUDA_COPY_ELEMENTS(imgsGPU_[i], raster[i]->data(), batch_size_, s);
            }

            CUDA_COPY_ELEMENTS(resizeDescrGPU_, resizeDescr_.data(), batch_size_, s);

            NDLL_CALL(BatchedResize(batch_size_, dim3(32, 32), s, C,
                                    CROP_PARAMS(resizeDescrGPU_), sizesGPU_, imgsGPU_));
        }
    }

 private:
    bool PrepareCropAndResize(const NDLLSize *input_size, NDLLSize *out_size, int C,
                              ResizeGridParam resizeParam[], ClassHandle *ppResizeTbl) const {
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
                          ResizeGridParam resizeParam[], ClassHandle *ppResizeTbl) const {
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
            ResizeMappingTable *pResizeTbl = MAPPING_TABLE(*ppResizeTbl);
            if (pResizeTbl && !pResizeTbl->IsValid(H0, W0, H1, W1)) {
                delete pResizeTbl;
                pResizeTbl = NULL;
            }

            if (!pResizeTbl) {
                const size_t lenClass = sizeof(ResizeMappingTable);
                CPU_BACKEND_MALLOC(*ppResizeTbl, lenClass, true, ResizeMappingTable);
                pResizeTbl = MAPPING_TABLE(*ppResizeTbl);
                pResizeTbl->constructTable(H0, W0, H1, W1, C,
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

    void CopyCongenericResizeParam(ResizeMappingTable *pResizeTbl, cudaStream_t s) {
        releaseCudaResizeMapingTable(pResizeTbl);
        COPY_TO_DEVICE(pResizeTbl->pResizeMapping[1], pResizeTbl->pResizeMapping[0], s);
        COPY_TO_DEVICE(pResizeTbl->pPixMapping[1], pResizeTbl->pPixMapping[0], s);
    }

    void releaseCudaResizeMapingTable(ResizeMappingTable *pResizeTbl) {
        BACKEND_FREE(pResizeTbl->pResizeMapping[1]);
        BACKEND_FREE(pResizeTbl->pPixMapping[1]);
    }

    void releaseCudaResizeParameter() {
        for (int i = input_t; i <= output_t; i++) {
            BACKEND_FREE(sizesGPU_[i]);
            BACKEND_FREE(imgsGPU_[i]);
        }

        BACKEND_FREE(resizeDescrGPU_);
    }

    // Members used in RunBatchedGPU;
    vector<NppiRect> resizeDescr_;
    ClassHandle resizeParamGPU_;

    ClassHandle sizesGPU_[2];
    ClassHandle imgsGPU_[2];

    ClassHandle resizeDescrGPU_;

    USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_NEW_RESIZE_H_
