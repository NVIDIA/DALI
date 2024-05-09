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

#include "NvEncoder/NvEncoderCuda.h"


NvEncoderCuda::NvEncoderCuda(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
    uint32_t nExtraOutputDelay, bool bMotionEstimationOnly, bool bOutputInVideoMemory, bool bUseIVFContainer):
    NvEncoder(NV_ENC_DEVICE_TYPE_CUDA, cuContext, nWidth, nHeight, eBufferFormat, nExtraOutputDelay, bMotionEstimationOnly, bOutputInVideoMemory, false, bUseIVFContainer),
    m_cuContext(cuContext)
{
    if (!m_hEncoder) 
    {
        NVENC_THROW_ERROR("Encoder Initialization failed", NV_ENC_ERR_INVALID_DEVICE);
    }

    if (!m_cuContext)
    {
        NVENC_THROW_ERROR("Invalid Cuda Context", NV_ENC_ERR_INVALID_DEVICE);
    }
}

NvEncoderCuda::~NvEncoderCuda()
{
    ReleaseCudaResources();
}

void NvEncoderCuda::AllocateInputBuffers(int32_t numInputBuffers)
{
    if (!IsHWEncoderInitialized())
    {
        NVENC_THROW_ERROR("Encoder intialization failed", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
    }

    // for MEOnly mode we need to allocate seperate set of buffers for reference frame
    int numCount = m_bMotionEstimationOnly ? 2 : 1;

    for (int count = 0; count < numCount; count++)
    {
        CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
        std::vector<void*> inputFrames;
        for (int i = 0; i < numInputBuffers; i++)
        {
            CUdeviceptr pDeviceFrame;
            uint32_t chromaHeight = GetNumChromaPlanes(GetPixelFormat()) * GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
            if (GetPixelFormat() == NV_ENC_BUFFER_FORMAT_YV12 || GetPixelFormat() == NV_ENC_BUFFER_FORMAT_IYUV)
                chromaHeight = GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
            CUDA_DRVAPI_CALL(cuMemAllocPitch((CUdeviceptr *)&pDeviceFrame,
                &m_cudaPitch,
                GetWidthInBytes(GetPixelFormat(), GetMaxEncodeWidth()),
                GetMaxEncodeHeight() + chromaHeight, 16));
            inputFrames.push_back((void*)pDeviceFrame);
        }
        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

        RegisterInputResources(inputFrames,
            NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            GetMaxEncodeWidth(),
            GetMaxEncodeHeight(),
            (int)m_cudaPitch,
            GetPixelFormat(),
            (count == 1) ? true : false);
    }
}

void NvEncoderCuda::SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream)
{
    //NVENC_API_CALL(m_nvenc.nvEncSetIOCudaStreams(m_hEncoder, inputStream, outputStream));
}

void NvEncoderCuda::ReleaseInputBuffers()
{
    ReleaseCudaResources();
}

void NvEncoderCuda::ReleaseCudaResources()
{
    if (!m_hEncoder)
    {
        return;
    }

    if (!m_cuContext)
    {
        return;
    }

    UnregisterInputResources();

    cuCtxPushCurrent(m_cuContext);

    for (uint32_t i = 0; i < m_vInputFrames.size(); ++i)
    {
        if (m_vInputFrames[i].inputPtr)
        {
            cuMemFree(reinterpret_cast<CUdeviceptr>(m_vInputFrames[i].inputPtr));
        }
    }
    m_vInputFrames.clear();

    for (uint32_t i = 0; i < m_vReferenceFrames.size(); ++i)
    {
        if (m_vReferenceFrames[i].inputPtr)
        {
            cuMemFree(reinterpret_cast<CUdeviceptr>(m_vReferenceFrames[i].inputPtr));
        }
    }
    m_vReferenceFrames.clear();

    cuCtxPopCurrent(NULL);
    m_cuContext = nullptr;
}

void NvEncoderCuda::CopyToDeviceFrame(CUcontext device,
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
    bool bUnAlignedDeviceCopy,
    CUstream stream,
    const uint32_t _srcChromaOffsets[])
{
    if (srcMemoryType != CU_MEMORYTYPE_HOST && srcMemoryType != CU_MEMORYTYPE_DEVICE)
    {
        NVENC_THROW_ERROR("Invalid source memory type for copy", NV_ENC_ERR_INVALID_PARAM);
    }

    NVTX_SCOPED_RANGE("CopyToDeviceFrame_aligned")

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(device));

    uint32_t srcPitch = nSrcPitch ? nSrcPitch : NvEncoder::GetWidthInBytes(pixelFormat, width);
    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = srcMemoryType;
    if (srcMemoryType == CU_MEMORYTYPE_HOST)
    {
        m.srcHost = pSrcFrame;
    }
    else
    {
        m.srcDevice = (CUdeviceptr)pSrcFrame;
    }
    m.srcPitch = srcPitch;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = pDstFrame;
    m.dstPitch = dstPitch;
    m.WidthInBytes = NvEncoder::GetWidthInBytes(pixelFormat, width);
    m.Height = height;
    if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
    {
        CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
    }
    else
    {
        CUDA_DRVAPI_CALL(stream == NULL? cuMemcpy2D(&m) : cuMemcpy2DAsync(&m, stream));
    }

    std::vector<uint32_t> srcChromaOffsets;
    if(_srcChromaOffsets)
    {
        for(int plane=0; plane<numChromaPlanes; ++plane)
        {
            srcChromaOffsets.push_back(_srcChromaOffsets[plane]);
        }
    }
    else
    {
        NvEncoder::GetChromaSubPlaneOffsets(pixelFormat, srcPitch, height, srcChromaOffsets);
    }
    uint32_t chromaHeight = NvEncoder::GetChromaHeight(pixelFormat, height);
    uint32_t destChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, dstPitch);
    uint32_t srcChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, srcPitch);
    uint32_t chromaWidthInBytes = NvEncoder::GetChromaWidthInBytes(pixelFormat, width);

    for (uint32_t i = 0; i < numChromaPlanes; ++i)
    {
        if (chromaHeight)
        {
            if (srcMemoryType == CU_MEMORYTYPE_HOST)
            {
                m.srcHost = ((uint8_t *)pSrcFrame + srcChromaOffsets[i]);
            }
            else
            {
                m.srcDevice = (CUdeviceptr)((uint8_t *)pSrcFrame + srcChromaOffsets[i]);
            }
            m.srcPitch = srcChromaPitch;

            m.dstDevice = (CUdeviceptr)((uint8_t *)pDstFrame + dstChromaOffsets[i]);
            m.dstPitch = destChromaPitch;
            m.WidthInBytes = chromaWidthInBytes;
            m.Height = chromaHeight;
            if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
            {
                CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
            }
            else
            {
                CUDA_DRVAPI_CALL(stream == NULL? cuMemcpy2D(&m) : cuMemcpy2DAsync(&m, stream));
            }
        }
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
}

void NvEncoderCuda::CopyToDeviceFrame(CUcontext device,
    void* pSrcFrame,
    uint32_t nSrcPitch,
    CUdeviceptr pDstFrame,
    uint32_t dstPitch,
    int width,
    int height,
    CUmemorytype srcMemoryType,
    NV_ENC_BUFFER_FORMAT pixelFormat,
    CUdeviceptr dstChromaDevicePtrs[],
    uint32_t dstChromaPitch,
    uint32_t numChromaPlanes,
    bool bUnAlignedDeviceCopy)
{
    if (srcMemoryType != CU_MEMORYTYPE_HOST && srcMemoryType != CU_MEMORYTYPE_DEVICE)
    {
        NVENC_THROW_ERROR("Invalid source memory type for copy", NV_ENC_ERR_INVALID_PARAM);
    }

    NVTX_SCOPED_RANGE("CopyToDeviceFrame_unaligned")

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(device));

    uint32_t srcPitch = nSrcPitch ? nSrcPitch : NvEncoder::GetWidthInBytes(pixelFormat, width);
    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = srcMemoryType;
    if (srcMemoryType == CU_MEMORYTYPE_HOST)
    {
        m.srcHost = pSrcFrame;
    }
    else
    {
        m.srcDevice = (CUdeviceptr)pSrcFrame;
    }
    m.srcPitch = srcPitch;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = pDstFrame;
    m.dstPitch = dstPitch;
    m.WidthInBytes = NvEncoder::GetWidthInBytes(pixelFormat, width);
    m.Height = height;
    if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
    {
        CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
    }
    else
    {
        CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
    }

    std::vector<uint32_t> srcChromaOffsets;
    NvEncoder::GetChromaSubPlaneOffsets(pixelFormat, srcPitch, height, srcChromaOffsets);
    uint32_t chromaHeight = NvEncoder::GetChromaHeight(pixelFormat, height);
    uint32_t srcChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, srcPitch);
    uint32_t chromaWidthInBytes = NvEncoder::GetChromaWidthInBytes(pixelFormat, width);

    for (uint32_t i = 0; i < numChromaPlanes; ++i)
    {
        if (chromaHeight)
        {
            if (srcMemoryType == CU_MEMORYTYPE_HOST)
            {
                m.srcHost = ((uint8_t *)pSrcFrame + srcChromaOffsets[i]);
            }
            else
            {
                m.srcDevice = (CUdeviceptr)((uint8_t *)pSrcFrame + srcChromaOffsets[i]);
            }
            m.srcPitch = srcChromaPitch;

            m.dstDevice = dstChromaDevicePtrs[i];
            m.dstPitch = dstChromaPitch;
            m.WidthInBytes = chromaWidthInBytes;
            m.Height = chromaHeight;
            if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
            {
                CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
            }
            else
            {
                CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
            }
        }
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
}
