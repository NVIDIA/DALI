// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/new_resize.h"
#include <float.h>
#include <assert.h>
#include <string>
#include <vector>
#include <nppdefs.h>
#include <npp.h>

namespace ndll {

void DataDependentSetupCPU(const Tensor<CPUBackend> &input,
                            Tensor<CPUBackend> *output, const char *pOpName,
                            vector<const uint8 *> *inPtrs, vector<uint8 *> *outPtrs,
                            vector<NDLLSize> *pSizes, const NDLLSize *out_size) {
    NDLL_ENFORCE(input.ndim() == 3);
    NDLL_ENFORCE(IsType<uint8>(input.type()), "Expects input data in uint8.");

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
                           vector<uint8 *> *outPtrs, vector<NDLLSize> *pSizes, ResizeAttr *pResize,
                           vector<NppiRect>  *pOutResize) {
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
        if (pResize) {
            // We are resizing
            NDLLSize *out_size = pResize->size(output_t, i);
            pResize->SetSize(pResize->size(input_t, i), input_shape,
                pResize->newSizes(i), out_size);

            if (pOutResize) {
                NppiRect &outResize = (*pOutResize)[i];
                outResize.height = out_size->height;
                outResize.width = out_size->width;

                const bool doingCrop = pResize->CropNeeded(*out_size);
                if (doingCrop)
                    pResize->DefineCrop(out_size, &outResize.x, &outResize.y);
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

__global__ void BatchedCongenericResizeKernel(
                    int H0, int W0, const uint8 *img_in, int H, int W, uint8 *img_out,
                    int C, const ResizeGridParam *resizeParam,
                    const ResizeMapping *pResizeMapping, const PixMapping *pPixMapping) {
    if (pResizeMapping && pPixMapping) {
        AUGMENT_RESIZE_GPU_CONGENERIC(H, W, C, img_in, img_out, RESIZE_N);
    } else {
        AUGMENT_RESIZE_GPU_CONGENERIC(H, W, C, img_in, img_out, RESIZE);
    }
}

NDLLError_t BatchedCongenericResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const NDLLSize &sizeIn, const uint8 *in_batch,
                          const NDLLSize &sizeOut, uint8 *out_batch,
                          const ResizeGridParam *pResizeParam, const ResizeMappingTable *pTbl) {
    BatchedCongenericResizeKernel<<<N, gridDim, 0, stream>>>
          (sizeIn.height, sizeIn.width, in_batch, sizeOut.height, sizeOut.width, out_batch, C,
           pResizeParam, pTbl? pTbl->pResizeMapping[1] : NULL, pTbl? pTbl->pPixMapping[1] : NULL);

    return NDLLSuccess;
}

//  Greatest Common Factor
int __host__ __device__ gcf(int a, int b) {
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
int __host__ __device__ lcm(int a, int b) {
    return a / gcf (a, b) * b;
}

__global__ void BatchedResizeKernel(int C, const NppiRect *resizeDescr,
                                    const NDLLSize *in_sizes, const uint8 *const imgs_in[],
                                    const NDLLSize *out_sizes, uint8 *const imgs_out[]) {
    const int H0 = in_sizes[blockIdx.x].height;
    const int W0 = in_sizes[blockIdx.x].width;

    const int H1 = resizeDescr[blockIdx.x].height;
    const int W1 = resizeDescr[blockIdx.x].width;
    const int H = out_sizes[blockIdx.x].height;
    const int W = out_sizes[blockIdx.x].width;

    const int lcmH = lcm(H0, H1);
    const int lcmW = lcm(W0, W1);
    ResizeGridParam resizeParam[3] = {
            {lcmW / W0, lcmH / H0},
            {lcmW / W1, lcmH / H1},
            {resizeDescr[blockIdx.x].x, resizeDescr[blockIdx.x].y}
    };

    AUGMENT_RESIZE_GPU_GENERIC(H, W, C, imgs_in[blockIdx.x], imgs_out[blockIdx.x], RESIZE);
}

NDLLError_t BatchedResize(int N, const dim3 &gridDim, cudaStream_t stream, int C,
                          const NppiRect *resizeDescr,
                          const NDLLSize * const sizes[], uint8 ** const raster[]) {
    const uint8 * const* in = raster[input_t];
    uint8 * const *out = raster[output_t];

    BatchedResizeKernel<<<N, gridDim, 0, stream>>>(C, resizeDescr,
            sizes[input_t], in, sizes[output_t], out);

    return NDLLSuccess;
}

ResizeMappingTable::ResizeMappingTable(int H0, int W0, int H1, int W1, int C,
             uint16_t xSize, uint16_t ySize) {
    io_size[0] = {W0, H0};
    io_size[1] = {W1, H1};
    C_ = C;

    pResizeMapping[0] = new ResizeMapping[xSize * ySize];
    pResizeMapping[1] = NULL;

    tableLength = xSize * ySize * sizeof(pResizeMapping[0][0]);
    memset(pResizeMapping[0], 0, tableLength);
    pPixMapping[0] = pPixMapping[1] = NULL;
}

ResizeMappingTable::~ResizeMappingTable() {
    delete [] pPixMapping[0];
    delete [] pResizeMapping[0];
    releaseCudaResizeMapingTable();
}

bool ResizeMappingTable::IsValid(int H0, int W0, int H1, int W1) const {
    if (!pResizeMapping[0])
        return false;

    return io_size[0].height == H0 && io_size[0].width == W0 &&
           io_size[1].height == H1 && io_size[1].width == W1;
}

void ResizeMappingTable::CopyCongenericResizeParam() {
    releaseCudaResizeMapingTable();
    CUDA_MALLOC(pResizeMapping[1], getMappingTableLength());
    CUDA_MEMCPY(pResizeMapping[1], pResizeMapping[0], getMappingTableLength());
    if (pPixMapping[0]) {
        CUDA_MALLOC(pPixMapping[1], pixMappingLen);
        CUDA_MEMCPY(pPixMapping[1], pPixMapping[0], pixMappingLen);
    }
}

void ResizeMappingTable::releaseCudaResizeMapingTable() {
    CUDA_FREE(pResizeMapping[1]);
    CUDA_FREE(pPixMapping[1]);
}

class PixMappingHelper {
 public:
    PixMappingHelper(uint32_t len, ResizeMapping *pMapping, uint32_t resizedArea = 0);
    void AddPixel(uint32_t addr, uint32_t area, int crdX, int crdY);
    void UpdateMapping(int shift, int centerX, int centerY);
    inline PixMapping *getPixMapping() const        { return pPixMapping_; }
    inline uint32_t numUsed() const                 { return numPixMapUsed_; }
 private:
    inline float distance(float x, float y) const   { return x * x + y * y; }
    uint32_t numPixMapMax_;     // length of the allocated PixMapping array
    uint32_t numPixMapUsed_;    // number of already used elements of pPixMapping
    PixMapping *pPixMapping_;
    ResizeMapping *pMappingBase_;
    ResizeMapping *pMapping_;

    const uint32_t area_;
    const uint32_t resizedArea_;
    float closestDist_;
    float centerX_, centerY_;
};

PixMappingHelper::PixMappingHelper(uint32_t area, ResizeMapping *pMapping, uint32_t resizedArea) :
        area_(area), resizedArea_(resizedArea) {
    numPixMapMax_ = 1;
    numPixMapUsed_ = 0;
    pPixMapping_ = resizedArea == 0? new PixMapping[numPixMapMax_ = 2 * area] : NULL;
    pMappingBase_ = pMapping;
}

void PixMappingHelper::AddPixel(uint32_t addr, uint32_t area, int crdX, int crdY) {
    assert(area != 0);
    if (numPixMapUsed_ == numPixMapMax_) {
        // Previously allocated array needs to be extended
        PixMapping *pPixMappingNew = new PixMapping[numPixMapMax_ <<= 1];
        memcpy(pPixMappingNew, pPixMapping_, numPixMapUsed_ * sizeof(pPixMappingNew[0]));
        pPixMapping_ = pPixMappingNew;
    }

    if (resizedArea_ == 0) {
        pMapping_->nPixels++;
        pPixMapping_[numPixMapUsed_++].Init(addr, area);
    } else {
       const float newDist = distance((crdX << 1) - centerX_, (crdY << 1) - centerY_);
       if (closestDist_ > newDist) {
           closestDist_ = newDist;
           pMapping_->intersectInfoAddr = addr;
       }
    }
}

void PixMappingHelper::UpdateMapping(int shift, int centerX, int centerY) {
    pMapping_ = pMappingBase_ + shift;
    pMapping_->intersectInfoAddr = resizedArea_? 0 : numUsed();

    centerX_ = centerX;
    centerY_ = centerY;
    closestDist_ = FLT_MAX;
}

#define RUN_CHECK_1     0

ResizeMappingTable *createResizeMappingTable(int H0, int W0, int H1, int W1, int C, bool use_NN) {
    // The table, which contains the information about correspondence of pixels of the initial
    // image to the pixels of the resized one.

    // Resizing from (H0, W0) to (H1, W1)
    // Main equations are:
    // H0 * sy0 = H1 * sy1
    // W0 * sx0 = W1 * sx1
    const size_t lcmH = lcm(H0, H1);
    const size_t lcmW = lcm(W0, W1);

    const int sy0 = lcmH / H0;
    const int sy1 = lcmH / H1;
    const int sx0 = lcmW / W0;
    const int sx1 = lcmW / W1;

    ResizeMappingTable *pTable = new ResizeMappingTable(H0, W0, H1, W1, C, sx0, sy0);
    PixMappingHelper helper(sx0 * sy0, pTable->pResizeMapping[0], use_NN? sx1 * sy1 : 0);

    // (x, y) pixel coordinate of PIX in resized image
    // 0 <= x < W1;  0 <= y < H1

    for (int y = 0; y < sy0; ++y) {
        for (int x = 0; x < sx0; ++x) {
            const int nX = x * sx1;
            const int nY = y * sy1;
            // The indices of the top-left pixel of the initial image, intersecting with PIX
            const int begIdx[2] = { nX / sx0, nY / sy0 };

            // The indices of the bottom-right pixel of the initial image, intersecting with PIX
            int endIdx[2] = { (nX + sx1) / sx0, (nY + sy1) / sy0 };

            // Intersection of the right (bottom) pixels with the PIX (could be equal to 0)
            const int extra[2] = { (nX + sx1) % sx0, (nY + sy1) % sy0 };

            // Length of the left (top) pixels intersecting with the PIX
            const int lenFirst[2] = { (sx0 - nX % sx0),   (sy0 - nY % sy0) };

            // Doubled (x,y) coordinates of the pixel's center
            const int lenX = endIdx[0] + begIdx[0] - (extra[0] ? 0 : 1);
            const int lenY = endIdx[1] + begIdx[1] - (extra[1] ? 0 : 1);

            // Relative address to the first intersecting pixels
            helper.UpdateMapping(((y * sy1) % sy0) * sx0 + (x * sx1) % sx0, lenX, lenY);

            endIdx[0] -= begIdx[0];
            endIdx[1] -= begIdx[1];
#if RUN_CHECK_1
            size_t check = 0;
#endif
            size_t rowMult = lenFirst[1];
            int y0 = 0;
            while (true) {
                int x0 = endIdx[0];

                // Relative address of the last pixel in row y0, intersecting with PIX
                uint32_t pixAddr = ((y0 * W0) + x0) * C;
                if (extra[0])
                    helper.AddPixel(pixAddr, extra[0] * rowMult, x0, y0);

                while (--x0 > 0)
                    helper.AddPixel(pixAddr -= C, sx0 * rowMult, x0, y0);

                helper.AddPixel(pixAddr -= C, lenFirst[0] * rowMult, x0, y0);

#if RUN_CHECK_1
                check += rowMult * (sx0 * (endIdx[0] - 1) + lenFirst[0] + extra[0]);
#endif
                if (++y0  >= endIdx[1]) {
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

    pTable->pPixMapping[0] = helper.getPixMapping();
    pTable->pixMappingLen = helper.numUsed() * sizeof(pTable->pPixMapping[0][0]);
    return pTable;
}

}  // namespace ndll

