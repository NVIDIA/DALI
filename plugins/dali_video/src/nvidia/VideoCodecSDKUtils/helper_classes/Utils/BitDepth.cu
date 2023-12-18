/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2010-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static __global__ void ConvertUInt8ToUInt16Kernel(uint8_t *dpUInt8, uint16_t *dpUInt16, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }
    int destStrideInPixels = nDestPitch / (sizeof(uint16_t));
    *(uchar2 *)&dpUInt16[y * destStrideInPixels + x] = uchar2{ 0, dpUInt8[y * nSrcPitch + x] };
}

static __global__ void ConvertUInt16ToUInt8Kernel(uint16_t *dpUInt16, uint8_t *dpUInt8, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }
    int srcStrideInPixels = nSrcPitch / (sizeof(uint16_t));
    dpUInt8[y * nDestPitch + x] = ((uchar2 *)&dpUInt16[y * srcStrideInPixels + x])->y;
}

void ConvertUInt8ToUInt16(uint8_t *dpUInt8, uint16_t *dpUInt16, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(((uint32_t)nWidth + blockSize.x - 1) / blockSize.x, ((uint32_t)nHeight + blockSize.y - 1) / blockSize.y, 1);
    ConvertUInt8ToUInt16Kernel <<< gridSize, blockSize >>>(dpUInt8, dpUInt16, nSrcPitch, nDestPitch, nWidth, nHeight);
}

void ConvertUInt16ToUInt8(uint16_t *dpUInt16, uint8_t *dpUInt8, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(((uint32_t)nWidth + blockSize.x - 1) / blockSize.x, ((uint32_t)nHeight + blockSize.y - 1) / blockSize.y, 1);
    ConvertUInt16ToUInt8Kernel <<<gridSize, blockSize >>>(dpUInt16, dpUInt8, nSrcPitch, nDestPitch, nWidth, nHeight);
}
