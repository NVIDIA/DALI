/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <vector>
#include <stdint.h>
#include <mutex>
#include <cuda.h>
#include "NvEncoder.h"

#define CUDA_DRVAPI_CALL( call )                                                                                                 \
    do                                                                                                                           \
    {                                                                                                                            \
        CUresult err__ = call;                                                                                                   \
        if (err__ != CUDA_SUCCESS)                                                                                               \
        {                                                                                                                        \
            const char *szErrName = NULL;                                                                                        \
            cuGetErrorName(err__, &szErrName);                                                                                   \
            std::ostringstream errorLog;                                                                                         \
            errorLog << "CUDA driver API error " << szErrName ;                                                                  \
            throw NVENCException::makeNVENCException(errorLog.str(), NV_ENC_ERR_GENERIC, __FUNCTION__, __FILE__, __LINE__);      \
        }                                                                                                                        \
    }                                                                                                                            \
    while (0)

/**
*  @brief Encoder for CUDA device memory.
*/
class NvEncoderCuda : public NvEncoder
{
public:
    NvEncoderCuda(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
        uint32_t nExtraOutputDelay = 3, bool bMotionEstimationOnly = false, bool bOPInVideoMemory = false, bool bUseIVFContainer = true);
    virtual ~NvEncoderCuda();

    /**
    *  @brief This is a static function to copy input data from host memory to device memory.
    *  This function assumes YUV plane is a single contiguous memory segment.
    */
    static void CopyToDeviceFrame(CUcontext device,
        void* pSrcFrame,
        uint32_t nSrcPitch,
        CUdeviceptr pDstFrame,
        uint32_t dstPitch,
        int width,
        int height,
        CUmemorytype srcMemoryType,
        NV_ENC_BUFFER_FORMAT pixelFormat,
        const uint32_t dstChromaOffsets[],
        uint32_t numChromaPlanes,
        bool bUnAlignedDeviceCopy = false,
        CUstream stream = NULL,
        const uint32_t srcChromaOffsets[] = NULL);

    /**
    *  @brief This is a static function to copy input data from host memory to device memory.
    *  Application must pass a seperate device pointer for each YUV plane.
    */
    static void CopyToDeviceFrame(CUcontext device,
        void* pSrcFrame,
        uint32_t nSrcPitch,
        CUdeviceptr pDstFrame,
        uint32_t dstPitch,
        int width,
        int height,
        CUmemorytype srcMemoryType,
        NV_ENC_BUFFER_FORMAT pixelFormat,
        CUdeviceptr dstChromaPtr[],
        uint32_t dstChromaPitch,
        uint32_t numChromaPlanes,
        bool bUnAlignedDeviceCopy = false);

    NV_ENCODE_API_FUNCTION_LIST GetApi() const { return m_nvenc;}

    void*                       GetEncoder() const { return m_hEncoder;}
    /**
    *  @brief This function sets input and output CUDA streams
    */
    void SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream);

protected:
    /**
    *  @brief This function is used to release the input buffers allocated for encoding.
    *  This function is an override of virtual function NvEncoder::ReleaseInputBuffers().
    */
    virtual void ReleaseInputBuffers() override;

private:
    /**
    *  @brief This function is used to allocate input buffers for encoding.
    *  This function is an override of virtual function NvEncoder::AllocateInputBuffers().
    */
    virtual void AllocateInputBuffers(int32_t numInputBuffers) override;

private:
    /**
    *  @brief This is a private function to release CUDA device memory used for encoding.
    */
    void ReleaseCudaResources();

protected:
    CUcontext m_cuContext;

private:
    size_t m_cudaPitch = 0;
};
