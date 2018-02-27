// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_MY_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_MY_RESIZE_H_

#include <random>
#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

#define USE_FAST_RESIZE   1

struct ResizeGridParam {
    uint32_t nX;
    uint32_t nY;
    void Init(uint32_t sx = 0, uint32_t sy = 0) { nX = sx; nY = sy; }
};

#define RESIZE_PREPARE()                        \
    const ResizeGridParam &w0 = resizeParam[0]; \
    const ResizeGridParam &w1 = resizeParam[1]; \
    const uint32_t sy0 = w0.nY;                 \
    const uint32_t sy1 = w1.nY;                 \
    const uint32_t sx0 = w0.nX;                 \
    const uint32_t sx1 = w1.nX;                 \
    const uint32_t area = sx1 * sy1;

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

#define RESIZE_CORE(C)                                                  \
    const uint32_t nX = x * sx1;                                        \
    const uint32_t nY = y * sy1;                                        \
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
        if (++y0  < endIdx[1])                                      \
            rowMult = sy0;                                          \
        else {                                                      \
            if (y0 > endIdx[1] || !(rowMult = extra[1]))            \
                break;                                              \
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
    uint16_t size[2];                   // (x, y) sizes of the ResizeMapping
    ResizeMapping *pResizeMapping;      // pointer to the ResizeMapping table
    PixMapping *pPixMapping;            // pointer to the PixMapping array
    uint16_t pixMappingLen;             // length of array pPixMapping in bites
    ResizeMappingTable(int H0, int W0, int H1, int W1, int C, uint16_t xSize, uint16_t ySize);
    ~ResizeMappingTable();
    inline uint32_t getMappingTableLength() const   { return size[0] * size[1]; }
    bool IsValid(int H0, int W0, int H1, int W1) const;
} ResizeMappingTable;


#define RESIZE_N_PREAMBLE(H, W, C)    RESIZE_PREPARE()

#define RESIZE_N_CORE(C)                                                \
    const uint32_t nX = x * sx1;                                        \
    const uint32_t nY = y * sy1;                                        \
    const uint32_t begIdx[2] = {nX / sx0, nY / sy0};                    \
    const uint32_t idx[2] =    {nX % sx0, nY % sy0};                    \
    ResizeMapping *pResizePix = pResizeMapping + idx[1] * sy0 + idx[0]; \
    PixMapping *pPixMap = pPixMapping + pResizePix->intersectInfoAddr;  \
    const uint8 *pBase = in + ((begIdx[1] * W0) + begIdx[0]) * C;       \
    uint32_t pixColor[3] = {0, 0, 0};                                   \
    for (uint16_t i = pResizePix->nPixels; i--;) {                      \
        const uint8 *pPix = pBase + (pPixMap + i)->pixAddr;             \
        const uint32_t pixArea = (pPixMap + i)->pixArea;                \
        pixColor[0] += *pPix * pixArea;                                 \
        if (C > 1) {                                                    \
            pixColor[1] += *(pPix + 1) * pixArea;                       \
            pixColor[2] += *(pPix + 2) * pixArea;                       \
        }                                                               \
    }                                                                   \
    SET_PIXEL_COLOR();

NDLLError_t BatchedResize(const uint8 *in_batch, int N,
                          const NDLLSize &sizeIn, const NDLLSize &outImgSise, int C,
                          uint8 *out_batch, const dim3 &gridDim, cudaStream_t stream,
                          const ResizeGridParam *resizeParam, const ResizeMappingTable *pTbl);

ResizeMappingTable *createResizeMappingTable(int H0, int W0, int H1, int W1, int C);
void releaseCudaResizeMapingTable();

template <typename Backend>
class MyResize : public Resize<Backend> {
 public:
    inline explicit MyResize(const OpSpec &spec) : Resize<Backend>(spec) {
        pResizeMappingTable = NULL;
        for (int i = 0; i < 2; i++)
            resizeParam_[i].Init(0, 0);
    }

    virtual inline ~MyResize()          {
        delete pResizeMappingTable;
        releaseCudaResizeMapingTable();
    };

 protected:

    void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        const auto output = ws->Output<CPUBackend>(idx);

        const vector <Index> &input_shape = input.shape();
        NDLLSize out_size, input_size;
        ResizeAttr::SetSize(input_size, input_shape, ResizeAttr::resize(), out_size);

        const int H0 = input_size.height;
        const int W0 = input_size.width;
        const int H1 = out_size.height;
        const int W1 = out_size.width;

        const vector<Index> &shape = input.shape();
        DataDependentSetupCPU(input, output, "MyResize", NULL, NULL, NULL, &out_size);

        const int C = shape[2];
        ResizeGridParam resizeParam[2];
        CreateResizeGrid(input_size, out_size, C, resizeParam);

        if (USE_FAST_RESIZE) {
            ResizeMapping *pResizeMapping = pResizeMappingTable->pResizeMapping;
            PixMapping *pPixMapping = pResizeMappingTable->pPixMapping;
            AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                               static_cast<uint8*>(output->raw_mutable_data()), RESIZE_N)
/*            int startH = 0;
            int startW = 0;
            int stepH = 1;
            int stepW = 1;
            RESIZE_PREPARE();
            const uint32_t offset = nYoffset(W1, C);                 \
            const uint32_t strideOut = H1 * offset * 0;         \
            const uint32_t strideIn = H0 *nYoffset(W0, C) * 0; \
            const uint32_t shift = 1 * offset;                  \
            const uint8 *in = input.template data<uint8>() + strideIn;
            uint8 * img_out = static_cast<uint8*>(output->raw_mutable_data()); \
            uint8 *out = img_out + strideOut + startH * offset - shift;\

            for (int y = startH; y < H1; y += stepH) {               \
                out += shift;                                       \
                for (int x = startW; x < W1; x += stepW) {           \

                    const uint32_t nX = x * sx1;                                    \
                    const uint32_t nY = y * sy1;                                    \
                    const uint32_t begIdx[2] = {nX / sx0, nY / sy0};                \
                    const uint32_t idx[2] =    {nX % sx0, nY % sy0};                \
                    ResizeMapping *pResizePix = pResizeMapping + idx[1] * sy0 + idx[0]; \
                    PixMapping *pPixMap = pPixMapping + pResizePix->intersectInfoAddr; \
                    const uint8 *pBase = in + ((begIdx[1] * W0) + begIdx[0]) * C;   \
                    uint32_t pixColor[3];                                           \
                    pixColor[0] = pixColor[1] = pixColor[2] = 0;                    \
                    for (uint16_t i = pResizePix->nPixels; i--;) {                  \
                        const uint8 *pPix = pBase + (pPixMap + i)->pixAddr;         \
                        const uint32_t pixArea = (pPixMap + i)->pixArea;            \
                        pixColor[0] += *pPix * pixArea;                             \
                        if (C > 1) {                                                \
                            pixColor[1] += *(pPix + 1) * pixArea;                   \
                            pixColor[2] += *(pPix + 2) * pixArea;                   \
                        }                                                           \
                    }

                    SET_PIXEL_COLOR();
                }                                                   \
            }
*/
        } else {
            AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                               static_cast<uint8 *>(output->raw_mutable_data()), RESIZE);
        }
    }

    void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<GPUBackend>(idx);
        auto output = ws->Output<GPUBackend>(idx);

        DataDependentSetupGPU(input, output, batch_size_, false,
                              ResizeAttr::inputImages(), ResizeAttr::outputImages(), NULL, this);

        const auto &shape = input.shape();

        const NDLLSize &sizeIn = ResizeAttr::size(input_t, 0);
        const NDLLSize &sizeOut = ResizeAttr::size(output_t, 0);
        const int C = shape[0][2];
        const bool newMapping = CreateResizeGrid(sizeIn, sizeOut, C, resizeParam_);
        NDLL_CALL(BatchedResize(
                input.template data<uint8>(),
                batch_size_, sizeIn, sizeOut, C,
                static_cast<uint8*>(output->raw_mutable_data()),
                dim3(32, 32), ws->stream(), newMapping? resizeParam_ : NULL,
                newMapping? pResizeMappingTable : NULL));
    }

    bool CreateResizeGrid(const NDLLSize &input_size, const NDLLSize &out_size, int C,
                          ResizeGridParam *resizeParam) {
        const int H0 = input_size.height;
        const int H1 = out_size.height;
        const int W0 = input_size.width;
        const int W1 = out_size.width;

        int lcm(int a, int b);
        const size_t lcmH = lcm(H0, H1);
        const size_t lcmW = lcm(W0, W1);

        bool newResize;
        if (resizeParam == resizeParam_) {
            newResize = resizeParam_[0].nX != lcmW / W0 || resizeParam_[0].nY != lcmH / H0 ||
                        resizeParam_[1].nX != lcmW / W1 || resizeParam_[1].nY != lcmH / H1;
        } else
            newResize = true;

        if (newResize) {
            resizeParam[0].Init(lcmW / W0, lcmH / H0);
            resizeParam[1].Init(lcmW / W1, lcmH / H1);
        }

        if (USE_FAST_RESIZE) {
            if (pResizeMappingTable && !pResizeMappingTable->IsValid(H0, W0, H1, W1)) {
                delete pResizeMappingTable;
                pResizeMappingTable = NULL;
            }

            if (!pResizeMappingTable) {
                pResizeMappingTable = createResizeMappingTable(H0, W0, H1, W1, C);
                newResize = true;
            }
        }

        return newResize;
    }

private:
    const ResizeMappingTable *pResizeMappingTable;
    ResizeGridParam resizeParam_[2];
    USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_MY_RESIZE_H_
