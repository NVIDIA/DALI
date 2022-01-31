/*
* Copyright 2017-2021 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "ColorSpace.h"

typedef struct {
    uint8_t r, g, b;
} Rgb;


__constant__ float matYuv2Rgb[3][3] = {
    1.164383f,  0.0f,       1.596027f,
    1.164383f, -0.391762f, -0.812968f,
    1.164383f,  2.017232f,  0.0f
};

__device__ static uint8_t Clamp(float x, float lower, float upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

__device__ inline Rgb YuvToRgbForPixel(uint8_t y, uint8_t u, uint8_t v) {
    const int 
        low = 1 << (sizeof(uint8_t) * 8 - 4),
        mid = 1 << (sizeof(uint8_t) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(uint8_t) * 8) - 1.0f;

    return Rgb { 
        Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf)};
}

__global__ static void YuvToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(uint8_t) + y * nYuvPitch;

    uint8_t *pDst1 = pRgb + x * sizeof(Rgb) + y * nRgbPitch;
    uint8_t *pDst2 = pRgb + x * sizeof(Rgb) + (y+1) * nRgbPitch;

    uint8_t luma1 = *pSrc;
    uint8_t luma2 = *(pSrc + nYuvPitch);
    uint8_t *chroma = (pSrc + (nHeight - y / 2) * nYuvPitch);

    Rgb rgb = YuvToRgbForPixel(luma1, chroma[0], chroma[1]);
    Rgb rgb2 = YuvToRgbForPixel(luma2, chroma[0], chroma[1]);

    pDst1[0] = rgb.r;
    pDst1[1] = rgb.g;
    pDst1[2] = rgb.b;

    pDst1[3] = rgb.r;
    pDst1[4] = rgb.g;
    pDst1[5] = rgb.b;

    pDst2[0] = rgb2.r;
    pDst2[1] = rgb2.g;
    pDst2[2] = rgb2.b;

    pDst2[3] = rgb2.r;
    pDst2[4] = rgb2.g;
    pDst2[5] = rgb2.b;

}

void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix, cudaStream_t stream) {
    YuvToRgbKernel
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}
