// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <nppdefs.h>
#include <npp.h>
#include "ndll/pipeline/operators/my_resize.h"

namespace ndll {

void DataDependentSetupCPU(const Tensor<CPUBackend> &input,
                            Tensor<CPUBackend> *output, const char *pOpName,
                            vector<const uint8 *> *inPtrs, vector<uint8 *> *outPtrs,
                            vector<NDLLSize> *pSizes, const NDLLSize *out_size) {
     NDLL_ENFORCE(input.ndim() == 3);
     NDLL_ENFORCE(IsType<uint8>(input.type()),
                  "Expects input data in uint8.");

     const vector <Index> &shape = input.shape();
     const int C = shape[2];
     NDLL_ENFORCE(C == 1 || C == 3,
                  string(pOpName? pOpName : "Operation") +
                  " supports only hwc rgb & grayscale inputs.");

     if (out_size)
         output->Resize({out_size->height, out_size->width, C});
     else
         output->Resize(shape);

     output->set_type(input.type());

     if (!inPtrs)
         return;

     (*inPtrs)[0] = input.template data<uint8>();
     if (outPtrs)
         (*outPtrs)[0] = static_cast<uint8*>(output->raw_mutable_data());

     if (pSizes) {
         (*pSizes)[0].height = shape[0];
         (*pSizes)[0].width = shape[1];
     }
 }

void DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> *output,
                           size_t batch_size, bool reshapeBatch, vector<const uint8 *> *inPtrs,
                           vector<uint8 *> *outPtrs, vector<NDLLSize> *pSizes, ResizeAttr *pntr) {
    NDLL_ENFORCE(IsType<uint8>(input.type()),
                 "Expected input data stored in uint8.");

    vector<Dims> output_shape(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        // Verify the inputs
        const auto &input_shape = input.tensor_shape(i);
        NDLL_ENFORCE(input_shape.size() == 3,
                     "Expects 3-dimensional image input.");

        NDLL_ENFORCE(input_shape[2] == 1 || input_shape[2] == 3,
                     "Not valid color type argument (1 or 3)");

        // Collect the output shapes
        if (pntr) {
            // We are resizing
            NDLLSize &out_size = pntr->size(output_t, i);
            pntr->SetSize(pntr->size(input_t, i), input_shape,
                          pntr->newSizes(i), out_size);

            // Collect the output shapes
            output_shape[i] = {out_size.height, out_size.width, input_shape[2]};
        } else
            output_shape[i] = input_shape;

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

    CollectPointersForExecution(reshapeBatch? 1 : batch_size, input, inPtrs, output, outPtrs);
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

__constant__ ResizeGridParam resizeParam[2];


__global__ void BatchedResizeKernel(const uint8 *img_in, int H0, int W0,
                                   int H, int W, int C, uint8 *img_out) {
    AUGMENT_RESIZE_GPU(H, W, C, img_in, img_out, RESIZE);
/*
    const int stepW  = blockDim.x;
    const int stepH = blockDim.y;
    const int startW = threadIdx.x;
    const int startH = threadIdx.y;
    const int imgIdx = blockIdx.x;

    RESIZE_PREAMBLE(H, W, C);

    const uint32_t offset = nYoffset(W, C);                 \
    const uint32_t strideOut = H * offset * imgIdx;         \
    const uint32_t strideIn = H0 *nYoffset(W0, C) * imgIdx; \
    const uint32_t shift = stepH * offset;                  \
    const uint8 *in = img_in + strideIn;                    \
    uint8 *out = img_out + strideOut + startH * offset - shift;\

    for (int y = startH; y < H; y += stepH) {               \
        out += shift;                                       \
        for (int x = startW; x < W; x += stepW) {
//            RESIZE_CORE(C);

            const uint32_t nX = x * sx1;
            const uint32_t nY = y * sy1;
            // The indices of the top-left pixel of the initial image, intersecting with PIX
            const uint32_t begIdx[2] = {nX / sx0, nY / sy0};

            // The indices of the bottom-right pixel of the initial image, intersecting with PIX
            const uint32_t endIdx[2] = {(nX + sx1) / sx0, (nY + sy1) / sy0};

            // Intersection of the right (bottom) pixels with the PIX (could be equal to 0)
            const uint32_t extra[2] = {(nX + sx1) % sx0, (nY + sy1) % sy0};

            // Length of the left (top) pixels intersecting with the PIX
            const uint32_t lenFirst[2] = {(sx0 - nX % sx0), (sy0 - nY % sy0)};
            uint32_t rowMult = lenFirst[1];

            pixColor[0] = pixColor[1] = pixColor[2] = 0;
            size_t y0 = begIdx[1];

            while (true) {
                uint32_t x0 = endIdx[0];

                // Last pixel in row y0, intersecting with PIX
                const uint8 *pPix = in + ((y0 * W0) + x0) * C;

                uint32_t len = extra[0];
                extraColor[0] = len * *pPix;
                if (C > 1) {
                    extraColor[1] = len * *(pPix + 1);
                    extraColor[2] = len * *(pPix + 2);
                }

                sumColor[0] = sumColor[1] = sumColor[2] = 0;
                while (--x0 > begIdx[0]) {
                    sumColor[0] += *(pPix -= C);
                    if (C > 1) {
                        sumColor[1] += *(pPix + 1);
                        sumColor[2] += *(pPix + 2);
                    }
                }

                len = lenFirst[0];

                pixColor[0] += rowMult * (sumColor[0] * sx0 + len * *(pPix -= C) + extraColor[0]);
                if (C > 1) {
                    pixColor[1] += rowMult * (sumColor[1] * sx0 + len * *(pPix + 1) + extraColor[1]);
                    pixColor[2] += rowMult * (sumColor[2] * sx0 + len * *(pPix + 2) + extraColor[2]);
                }

                if (++y0  < endIdx[1])
                    rowMult = sy0;
                else {
                    if (y0 > endIdx[1] || !(rowMult = extra[1]))
                        break;
                }
            }

            const int to = x * C;
            out[to] = (pixColor[0] + (area >> 1)) / area;
            if (C > 1) {
                out[to + 1] = (pixColor[1] + (area >> 1)) / area;
                out[to + 2] = (pixColor[2] + (area >> 1)) / area;
            }
        }
    } */
}

NDLLError_t BatchedResize(const uint8 *in_batch, int N,
                          const NDLLSize &inImg, const NDLLSize &outImg, int C,
                          uint8 *out_batch, const dim3 &gridDim, cudaStream_t stream,
                          ResizeGridParam *pResizeParam) {
     // Copying the descriptor of operation into __constant__ memory
    cudaMemcpyToSymbol(resizeParam, pResizeParam, sizeof(resizeParam));

    BatchedResizeKernel<<<N, gridDim, 0, stream>>>
                     (in_batch, inImg.height, inImg.width, outImg.height, outImg.width, C, out_batch);
    return NDLLSuccess;
}


//  Greatest Common Factor
int gcf (int a, int b) {
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

#include <assert.h>

// Least Common Multiplier
int lcm (int a, int b) {
    return a / gcf (a, b) * b;
}

void defineColor(int W0, int C, const uint8 *img_in, size_t x, size_t y, size_t sx0, size_t sx1, size_t sy0, size_t sy1, uint8 *pix_out)
{
    // Resising from (H0, W0) to (H1, W1)
    // Main equations are:
    // H0 * sy0 = H1 * sy1
    // W0 * sx0 = W1 * sx1
    //
    // (x, y) pixel coordinate of PIX in resized image
    // 0 <= x < W1;  0 <= y < H1
    const size_t nX = x * sx1;
    const size_t nY = y * sy1;
    // The indices of the top-left pixel of the initial image, intersecting with PIX
    const size_t begIdx[2] = { nX / sx0, nY / sy0 };

    // The indices of the bottom-right pixel of the initial image, intersecting with PIX
    size_t endIdx[2] = { (nX + sx1) / sx0, (nY + sy1) / sy0 };

    // Intersection of the right (bottom) pixels with the PIX (could be equal to 0)
    const size_t extra[2] = { (nX + sx1) % sx0, (nY + sy1) % sy0 };

    // Length of the left (top) pixels intersecting with the PIX
    const size_t lenFirst[2] = { (sx0 - nX % sx0),   (sy0 - nY % sy0)};

#define RUN_CHECK   0
#define MAKE_OUTPUT 0
#if RUN_CHECK

    size_t check = 0;
#if MAKE_OUTPUT
    FILE *file = x == 0? fopen("aaa.txt", y == 0? "w" : "a") : NULL;
    if (file) {
        fprintf(file, "(x, y) = (%3ld %3ld), sx = (%3ld %3ld),  sy = (%3ld %3ld)\n", x, y, sx0, sx1, sy0, sy1);
        fprintf(file, "begIdx = (%3ld, %3ld)  endIdx = (%3ld, %3ld)\n", begIdx[0], begIdx[1], endIdx[0], endIdx[1]);
        fprintf(file, "extra = (%3ld, %3ld)  lenFirst = (%3ld, %3ld)\n", extra[0], extra[1], lenFirst[0], lenFirst[1]);
    }
#endif
#endif

    size_t sumColor[3], extraColor[3], pixColor[3];
    size_t rowMult =  lenFirst[1];

    pixColor[0] = pixColor[1] = pixColor[2] = 0;
    size_t y0 = begIdx[1];
    while (true) {
        size_t x0 = endIdx[0];

        // Last pixel in row y0, intersecting with PIX
        const uint8 *pPix = img_in + ((y0 * W0) + x0) * C;

        size_t len = extra[0];
        extraColor[0] = len * *pPix;
        if (C > 1) {
            extraColor[1] = len * *(pPix + 1);
            extraColor[2] = len * *(pPix + 2);
        }

        sumColor[0] = sumColor[1] = sumColor[2] = 0;
        while (--x0 > begIdx[0]) {
            sumColor[0] += *(pPix -= C);
            if (C > 1) {
                sumColor[1] += *(pPix + 1);
                sumColor[2] += *(pPix + 2);
            }
        }

        len = lenFirst[0];
        pixColor[0] += rowMult * (sumColor[0] * sx0 + len * *(pPix -= C) + extraColor[0]);
        if (C > 1) {
            pixColor[1] += rowMult * (sumColor[1] * sx0 + len * *(pPix + 1) + extraColor[1]);
            pixColor[2] += rowMult * (sumColor[2] * sx0 + len * *(pPix + 2) + extraColor[2]);
        }

#if RUN_CHECK
            check += rowMult * (sx0 * (endIdx[0] - begIdx[0] - 1) + len + extra[0]);
#if MAKE_OUTPUT
        if (file)
            fprintf(file, "check = %ld ==> %ld * (%ld * (%ld - %ld - 1) + %ld - %ld)\n", check, rowMult, sx0, endIdx[0], begIdx[0], len, extra[0]);
#endif
#endif

            if (++y0  < endIdx[1])
                rowMult = sy0;
            else {
                if (y0 > endIdx[1] || !(rowMult = extra[1]))
                    break;
            }
        }

#if RUN_CHECK
        #if MAKE_OUTPUT
    if (file)
        fclose(file);
#endif
    assert(check == sx1 * sy1);
#endif

    const size_t area = sx1 * sy1;
    pix_out[0] = (pixColor[0] + (area >> 1)) / area;
    if (C > 1) {
        pix_out[1] = (pixColor[1] + (area >> 1)) / area;
        pix_out[2] = (pixColor[2] + (area >> 1)) / area;
    }
}

void resize(int H0, int W0, int H1, int W1, int C, const uint8 *img_in, uint8 *img_out)
{
    const size_t lcmH = lcm(H0, H1);
    const size_t lcmW = lcm(W0, W1);

    const size_t sy0 = lcmH / H0;
    const size_t sy1 = lcmH / H1;
    const size_t sx0 = lcmW / W0;
    const size_t sx1 = lcmW / W1;
    uint8 color[3];
    for (int y = 0; y < H1; ++y) {
        for (int x = 0; x < W1; ++x) {
            defineColor(W0, C, img_in, x, y, sx0, sx1, sy0, sy1, color);
        }
    }
}

void resize_test()
{
//    const int H0 = 480, W0 = 360; //240; 270;
//    const int H1 = 240, W1 = 180; //120;  130
    const int H0 = 224, W0 = 224; //240; 270;
    const int H1 = 220, W1 = 224; //120;  130
    const int C = 1;

    const size_t len = H0 * W0;
    uint8 *img_in = new uint8[len];
    memset(img_in, 1, len);

    resize(H0, W0, H1, W1, C, img_in, NULL);
    delete[] img_in;
}

}  // namespace ndll

