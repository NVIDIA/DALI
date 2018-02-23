// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#ifndef NDLL_PIPELINE_OPERATORS_MY_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_MY_RESIZE_H_

#include <random>
#include <ctgmath>
#include <vector>
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

struct ResizeGridParam {
    size_t nX;
    size_t nY;
    void Init(size_t sx, size_t sy)         { nX = sx; nY = sy; }
};

#define RESIZE_PREAMBLE(H, W, C)                \
    const ResizeGridParam &w0 = resizeParam[0]; \
    const ResizeGridParam &w1 = resizeParam[1]; \
    const uint32_t sy0 = w0.nY;                 \
    const uint32_t sy1 = w1.nY;                 \
    const uint32_t sx0 = w0.nX;                 \
    const uint32_t sx1 = w1.nX;                 \
    const uint32_t area = sx1 * sy1;            \
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
        if (C > 1) {                \
            pixColor[1] += rowMult * (sumColor[1] * sx0 + len * *(pPix + 1) + extraColor[1]); \
            pixColor[2] += rowMult * (sumColor[2] * sx0 + len * *(pPix + 2) + extraColor[2]); \
        }                                   \
                                            \
        if (++y0  < endIdx[1])                                      \
            rowMult = sy0;                                          \
        else {                                                      \
            if (y0 > endIdx[1] || !(rowMult = extra[1]))            \
                break;                                              \
        }                                                           \
    }                                                               \
                                                                    \
    const uint32_t to = x * C;                                      \
    out[to] = (pixColor[0] + (area >> 1)) / area;                   \
    if (C > 1) {                                                    \
        out[to + 1] = (pixColor[1] + (area >> 1)) / area;           \
        out[to + 2] = (pixColor[2] + (area >> 1)) / area;           \
    }                                                               \

NDLLError_t BatchedResize(const uint8 *in_batch, int N,
                          const NDLLSize &sizeIn, const NDLLSize &outImgSise, int C,
                          uint8 *out_batch, const dim3 &gridDim, cudaStream_t stream,
                          ResizeGridParam *resizeParam);

template <typename Backend>
class MyResize : public Resize<Backend> {
 public:
    inline explicit MyResize(const OpSpec &spec) : Resize<Backend>(spec) {}

    virtual inline ~MyResize() = default;

 protected:

    void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        const auto output = ws->Output<CPUBackend>(idx);

        const vector <Index> &input_shape = input.shape();
        NDLLSize out_size, input_size;
        ResizeAttr::SetSize(input_size, input_shape, ResizeAttr::resize(), out_size);

        ResizeGridParam resizeParam[2];
        CreateResizeGrid(input_size, out_size, resizeParam);

        const int H0 = input_size.height;
        const int W0 = input_size.width;
        const int H1 = out_size.height;
        const int W1 = out_size.width;

        const vector<Index> &shape = input.shape();
        DataDependentSetupCPU(input, output, "MyResize", NULL, NULL, NULL, &out_size);
        AUGMENT_RESIZE_CPU(H1, W1, shape[2], input.template data<uint8>(),
                              static_cast<uint8*>(output->raw_mutable_data()), RESIZE);
    }

    void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<GPUBackend>(idx);
        auto output = ws->Output<GPUBackend>(idx);

        DataDependentSetupGPU(input, output, batch_size_, false,
                              ResizeAttr::inputImages(), ResizeAttr::outputImages(), NULL, this);

        const auto &shape = input.shape();

        const NDLLSize &sizeIn = ResizeAttr::size(input_t, 0);
        const NDLLSize &sizeOut = ResizeAttr::size(output_t, 0);
        ResizeGridParam resizeParam[2];
        CreateResizeGrid(sizeIn, sizeOut, resizeParam);

        NDLL_CALL(BatchedResize(
                input.template data<uint8>(),
                batch_size_, sizeIn, sizeOut, shape[0][2],
                static_cast<uint8*>(output->raw_mutable_data()),
                dim3(32, 32), ws->stream(), resizeParam));
    }

    void CreateResizeGrid(const NDLLSize &input_size, const NDLLSize &out_size,
                         ResizeGridParam resizeParam[2]) {
        const int H0 = input_size.height;
        const int H1 = out_size.height;
        const int W0 = input_size.width;
        const int W1 = out_size.width;

        int lcm(int a, int b);
        const size_t lcmH = lcm(H0, H1);
        const size_t lcmW = lcm(W0, W1);

        resizeParam[0].Init(lcmW / W0, lcmH / H0);
        resizeParam[1].Init(lcmW / W1, lcmH / H1);
    }

private:
    USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_MY_RESIZE_H_
